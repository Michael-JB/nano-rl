import random
from collections import deque
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from ..environment import Environment, DigitEnvironment


@dataclass(frozen=True)
class TrainConfig:
    # The number of experiences to sample
    group_size: int
    # The maximum number of response tokens that the model can generate during rollouts
    max_rollout_tokens: int
    # The number of training steps; we only do rollouts once per train step
    train_steps: int
    # The number of groups to process in a single gradient step
    train_batch_size: int
    # The number of groups the dynamic sampling buffer can hold
    dynamic_sampling_buffer_capacity: int
    # The epsilon_low term in the DAPO objective
    epsilon_low: float
    # The epsilon_high term in the DAPO objective
    epsilon_high: float

    def __post_init__(self) -> None:
        if self.dynamic_sampling_buffer_capacity % self.train_batch_size != 0:
            raise ValueError(
                "Dynamic sampling buffer capacity must be a multiple of train batch size"
            )


@dataclass(frozen=True)
class Experience:
    # A 1-dimensional tensor of the tokens of the combined prompt and response
    completion: torch.Tensor
    # A boolean mask of the same length as `self.completion` that marks prompt
    # tokens with `False` and response tokens with `True`.
    prompt_mask: list[bool]
    # A 1-dimensional tensor of the log probabilities of the response tokens.
    # We don't need gradients here.
    log_probs: torch.Tensor
    # A float reward for the response.
    reward: float

    @property
    def response(self) -> torch.Tensor:
        return self.completion[self.prompt_mask]


class DynamicSamplingBuffer:
    """
    The dynamic sampling buffer holds groups of experiences. It can hold up to
    `capacity` groups.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[list[Experience]] = deque()

    def push(self, experiences: list[Experience]) -> None:
        if self.full():
            raise RuntimeError("Dynamic sampling buffer is full")
        self.buffer.append(experiences)

    def pop(self) -> list[Experience]:
        return self.buffer.pop()

    def full(self) -> bool:
        return len(self.buffer) == self.capacity

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

    return Experience(
        completion,
        prompt_mask,
        response_log_probs(response, response_logits),
        reward(tokenizer, environment, response),
    )


def response_log_probs(
    response: torch.Tensor, response_logits: torch.Tensor
) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)  # T, C
    # pluck out the actual generated token from the vocab dimension
    response_log_probs = log_probs[torch.arange(response.size(0)), response]  # T
    return response_log_probs  # T


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
    epsilon_low: float,
    epsilon_high: float,
) -> torch.Tensor:
    rewards = torch.Tensor([experience.reward for experience in group])
    mean_reward = rewards.mean().item()
    std_reward = rewards.std().item()

    def experience_objective(experience: Experience) -> torch.Tensor:
        # DAPO dynamic sampling guarantees that std_reward will never be 0
        advantage = (experience.reward - mean_reward) / std_reward

        logits = response_logits(model, experience)
        log_probs = response_log_probs(experience.response, logits)

        # we exponentiate as the importance is the ratio between the policy
        # probabilities, not the log probs. We operate in log space for
        # numerical stability.
        importance = torch.exp(log_probs - experience.log_probs)  # T
        clip_importance = torch.clip(importance, 1 - epsilon_low, 1 + epsilon_high)  # T

        return torch.minimum(importance * advantage, clip_importance * advantage)  # T

    objectives = [experience_objective(experience) for experience in group]
    # concatenate the group to compute the token-level mean
    return torch.cat(objectives).mean()


def train(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: Environment,
    config: TrainConfig,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, maximize=True)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=config.train_steps
        * config.dynamic_sampling_buffer_capacity
        // config.train_batch_size,
    )
    dynamic_sampling_buffer = DynamicSamplingBuffer(
        config.dynamic_sampling_buffer_capacity
    )

    for step in range(config.train_steps):
        # Populate the dynamic sampling buffer ith rollout groups. In "real"
        # implementations, this would happen in parallel with training, though
        # we keep things sequential here for simplicity.
        print("-- Generating rollouts")
        model.eval()
        with torch.no_grad():
            while not dynamic_sampling_buffer.full():
                group = [
                    rollout(
                        device, model, tokenizer, environment, config.max_rollout_tokens
                    )
                    for _ in range(config.group_size)
                ]
                # Dynamic sampling: we only add groups that have reward
                # diversity. Note: as we don't have a binary reward, we check
                # for all-equal rewards rather than for all-1 rewards as the
                # DAPO objective describes.
                if len(set([experience.reward for experience in group])) == 1:
                    print("Dynamic sampling: rejecting group with no reward diversity.")
                    continue
                print("Dynamic sampling: accepting group.")
                dynamic_sampling_buffer.push(group)

        # Perform batched gradient steps from groups in the dynamic sampling
        # buffer. For each gradient step in this loop, the training will stray
        # further off-policy.
        print("-- Training")
        model.train()
        while len(dynamic_sampling_buffer) > 0:
            # while we could backpropagate groups individually, the DAPO paper
            # batches multiple groups into a single gradient step so we demonstrate
            # that here.
            groups = [
                dynamic_sampling_buffer.pop() for _ in range(config.train_batch_size)
            ]
            batch_objectives = [
                objective(model, group, config.epsilon_low, config.epsilon_high)
                for group in groups
            ]
            batch_objective = torch.stack(batch_objectives).sum()
            batch_objective.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(
                f"Grad step for group batch in train step {step + 1}/{config.train_steps} completed. "
                f"Batch objective: {batch_objective.item():.6f}. "
                f"{len(dynamic_sampling_buffer)}/{dynamic_sampling_buffer.capacity} groups remain in dynamic sampling buffer."
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
        group_size=6,
        max_rollout_tokens=4,
        train_steps=4,
        train_batch_size=2,
        dynamic_sampling_buffer_capacity=6,
        epsilon_low=0.2,
        epsilon_high=0.28,
    )

    train(device, model, tokenizer, environment, train_config)


if __name__ == "__main__":
    main()
