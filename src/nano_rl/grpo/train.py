import random
from dataclasses import dataclass
from collections import deque

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_scheduler,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from ..environment import Environment, DigitEnvironment


@dataclass(frozen=True)
class TrainConfig:
    # The number of rollouts per step
    rollout_count: int
    # The number of experiences to contribute to a single gradient step
    group_size: int
    # The maximum number of response tokens that the model can generate during rollouts
    max_rollout_tokens: int
    # The number of training steps; we only do rollouts once per train step
    train_steps: int
    # The number of gradient update steps per train step. If this is greater than 1, we will go off-policy
    grad_updates_per_step: int
    # The maximum number of elements in the replay buffer. The larger this is, the more off-policy the training can go
    replay_buffer_capacity: int
    # The epsilon term in the GRPO objective
    epsilon: float


@dataclass(frozen=True)
class Experience:
    # A 1-dimensional tensor of the tokens of the combined prompt and response
    completion: torch.Tensor
    # A boolean mask of the same length as `self.completion` that marks prompt
    # tokens with `False` and response tokens with `True`.
    prompt_mask: list[bool]
    # The sum of the log probs of the response at generation time (i.e.,
    # according to the policy that the response was generated with)
    log_probs_sum: float
    # A float reward for the response.
    reward: float

    @property
    def response(self) -> torch.Tensor:
        return self.completion[self.prompt_mask]


class ReplayBuffer:
    """
    The replay buffer is a ring-buffer for experiences. It stores up to
    `capacity` experiences. When new experiences are pushed, the oldest
    experiences are discarded to satisfy the capacity. You can access these
    experiences by calling `sample`; this samples experiences uniformly at
    random from the whole buffer.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, experiences: list[Experience]) -> None:
        self.buffer.extend(experiences)

    def sample(self, count: int) -> list[Experience]:
        if len(self.buffer) < count:
            raise ValueError("Failed to sample; insufficient buffer entries")
        return random.sample(list(self.buffer), count)

    def __len__(self) -> int:
        return len(self.buffer)


def rollout(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: Environment,
    max_response_tokens: int,
) -> Experience:
    messages = environment.prompt()
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(device)  # type: ignore
    model_output: GenerateDecoderOnlyOutput = model.generate(
        inputs,
        max_new_tokens=max_response_tokens,
        temperature=1,
        # both required to return the completion and the response logits
        output_logits=True,
        return_dict_in_generate=True,
    )  # type: ignore

    # squeeze out the batch dimension
    completion = model_output.sequences.squeeze(0)

    prompt_length = len(inputs[0])
    prompt_mask = [False] * prompt_length + [True] * (len(completion) - prompt_length)
    assert len(prompt_mask) == len(completion)

    response = completion[prompt_mask]
    # squeeze out the batch dimension
    response_logits = torch.stack(model_output.logits).squeeze(1)  # T, C
    # pluck out the item as we don't need grads
    log_probs_sum = sum_log_probs(response, response_logits).item()

    return Experience(
        completion,
        prompt_mask,
        log_probs_sum,
        reward(tokenizer, environment, response),
    )


def sum_log_probs(
    response: torch.Tensor, response_logits: torch.Tensor
) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)  # T, C
    # pluck out the actual generated token from the vocab dimension
    response_log_probs = log_probs[torch.arange(response.size(0)), response]  # T
    return response_log_probs.sum(dim=-1)  # 1


def reward(
    tokenizer: PreTrainedTokenizer,
    environment: Environment,
    response: torch.Tensor,
) -> float:
    response_text = tokenizer.decode(response, skip_special_tokens=True)
    reward = environment.reward(response_text)
    print(f"Reward: {reward:.3f}; Response: {response_text}")
    return reward


def response_logits(
    model: PreTrainedModel,
    experience: Experience,
) -> torch.Tensor:
    # these are the logits at each sequence position for the whole completion
    # (i.e., including the prompt)
    completion_logits: torch.Tensor = model(
        experience.completion.unsqueeze(0)
    ).logits.squeeze(0)
    # the completion logits include a prediction past the end of sequence, so we
    # strip this
    sequence_logits = completion_logits[:-1, :]
    # the completion does not include logits for the first token as there is no
    # prior context, so we strip the first element in the mask
    causal_prompt_mask = experience.prompt_mask[1:]
    # we mask to only consider the response tokens for the objective
    return sequence_logits[causal_prompt_mask, :]


def objective(
    model: PreTrainedModel,
    group: list[Experience],
    epsilon: float,
) -> torch.Tensor:
    rewards = torch.Tensor([experience.reward for experience in group])
    mean_reward = rewards.mean().item()
    std_reward = rewards.std().item()

    def experience_objective(experience: Experience, reward: float) -> torch.Tensor:
        # We add a small epsilon to the standard deviation to prevent
        # zero-divides when all samples in the group have the same reward.
        # In these cases, there is no useful training signal.
        advantage = (reward - mean_reward) / (std_reward + 1e-6)

        logits = response_logits(model, experience)
        log_probs_sum = sum_log_probs(experience.response, logits)

        # we exponentiate as the importance is the ratio between the policy
        # probabilities, not the log probs. We operate in log space for
        # numerical stability.
        importance = torch.exp(log_probs_sum - experience.log_probs_sum)
        clipped_importance = torch.clip(importance, 1 - epsilon, 1 + epsilon)

        return torch.min(importance * advantage, clipped_importance * advantage)

    objectives = [
        experience_objective(experience, reward)
        for experience, reward in zip(group, rewards.tolist())
    ]
    return torch.stack(objectives).mean()


def train(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: Environment,
    config: TrainConfig,
) -> None:
    model.to(device)  # type: ignore
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, maximize=True)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=config.train_steps * config.grad_updates_per_step,
    )
    replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

    for step in range(config.train_steps):
        # Populate the replay buffer with rollouts. In "real" implementations,
        # this would happen in parallel with training, though we keep things
        # sequential here for simplicity.
        print("-- Generating rollouts")
        model.eval()
        with torch.no_grad():
            experiences = [
                rollout(
                    device, model, tokenizer, environment, config.max_rollout_tokens
                )
                for _ in range(config.rollout_count)
            ]
            replay_buffer.push(experiences)

        # Perform gradient steps from samples in the replay buffer. For each
        # gradient step in this loop, the training will stray further
        # off-policy.
        print("-- Training")
        model.train()
        for grad_step in range(config.grad_updates_per_step):
            # In a real implementation, we would batch multiple groups into a
            # single gradient step for efficiency by summing their objectives.
            group = replay_buffer.sample(config.group_size)
            group_objective = objective(model, group, config.epsilon)
            group_objective.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(
                f"Grad step {grad_step + 1}/{config.grad_updates_per_step} "
                f"in train step {step + 1}/{config.train_steps} completed. "
                f"Objective: {group_objective.item():.6f}"
            )


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    environment = DigitEnvironment()
    train_config = TrainConfig(
        rollout_count=25,
        group_size=5,
        max_rollout_tokens=10,
        train_steps=5,
        grad_updates_per_step=5,
        replay_buffer_capacity=50,
        epsilon=0.2,
    )

    train(device, model, tokenizer, environment, train_config)


if __name__ == "__main__":
    main()
