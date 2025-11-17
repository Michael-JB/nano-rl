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

from environment import Environment

MODEL_NAME = "Qwen/Qwen3-0.6B"
ROLLOUT_COUNT = 20  # Number of valid samples to add to the replay buffer per step (implementation detail)
GROUP_SIZE = 5  # The number of experiences to contribute to a single gradient step
MAX_ROLLOUT_TOKENS = 4  # The maximum number of response tokens that the model can generate during rollouts
TRAIN_STEPS = 6  # The number of training steps; we only do rollouts once per train step
GRAD_UPDATES_PER_STEP = 5  # The number of gradient update steps per train step. If this is greater than 1, we will go off-policy
REPLAY_BUFFER_CAPACITY = 50  # The maximum number of elements in the replay buffer. The larger this is, the more off-policy the training can go
EPSILON_LOW = 0.2  # The epsilon_low term in the DAPO objective
EPSILON_HIGH = 0.28  # The epsilon_high term in the DAPO objective


@dataclass(frozen=True)
class Experience:
    # A 1-dimensional tensor of the tokens of the combined prompt and response
    completion: torch.Tensor
    # A boolean mask of the same length as `self.completion` that marks prompt
    # tokens with `False` and response tokens with `True`.
    prompt_mask: list[bool]
    # A 1-dimensional tensor of the log probabilities of the response tokens.
    log_probs: torch.Tensor
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
        max_new_tokens=MAX_ROLLOUT_TOKENS,
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
) -> torch.Tensor | None:
    rewards = torch.Tensor([experience.reward for experience in group])
    mean_reward = rewards.mean().item()
    std_reward = rewards.std().item()

    if std_reward == 0:
        # we get no signal from this group; throw it away. Note: due to dynamic
        # sampling, this should be very unlikely in DAPO, unless you've got a
        # discrete reward (as we do in this example).
        return None

    def experience_objective(experience: Experience) -> torch.Tensor:
        advantage = (experience.reward - mean_reward) / std_reward

        logits = response_logits(model, experience)
        log_probs = response_log_probs(experience.response, logits)

        # we exponentiate as the importance is the ratio between the policy
        # probabilities, not the log probs. We operate in log space for
        # numerical stability.
        importance = torch.exp(log_probs - experience.log_probs)  # T
        clip_importance = torch.clip(importance, 1 - EPSILON_LOW, 1 + EPSILON_HIGH)  # T

        return torch.minimum(importance * advantage, clip_importance * advantage)  # T

    objectives = [experience_objective(experience) for experience in group]
    # concatenate the group to compute the token-level mean
    return torch.cat(objectives).mean()


def train(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: Environment,
) -> None:
    model.to(device)  # type: ignore
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, maximize=True)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=TRAIN_STEPS * GRAD_UPDATES_PER_STEP,
    )
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    for step in range(TRAIN_STEPS):
        # Populate the replay buffer with rollouts using DAPO's dynamic
        # sampling: over-sample and discard rewards that might lead to a zero
        # group advantage.`ROLLOUT_COUNT` is just here as we're doing things
        # sequentially. In "real" implementations, this would happen in parallel
        # with training, though we keep things sequential here for simplicity.
        model.eval()
        with torch.no_grad():
            new_experiences: list[Experience] = []
            while len(new_experiences) < ROLLOUT_COUNT:
                experience = rollout(device, model, tokenizer, environment)
                if experience.reward == 0 or experience.reward == 1:
                    continue
                new_experiences.append(experience)
            replay_buffer.push(new_experiences)

        # Perform gradient steps from samples in the replay buffer. For each
        # gradient step in this loop, the training will stray further
        # off-policy.
        model.train()
        for grad_step in range(GRAD_UPDATES_PER_STEP):
            group = replay_buffer.sample(GROUP_SIZE)
            group_objective = objective(model, group)
            if not group_objective:
                print(
                    f"-- No signal in grad step {grad_step + 1}/{GRAD_UPDATES_PER_STEP} "
                    f"in train step {step + 1}/{TRAIN_STEPS}; skipping grad update --"
                )
                continue

            group_objective.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(
                f"-- Grad step {grad_step + 1}/{GRAD_UPDATES_PER_STEP} "
                f"in train step {step + 1}/{TRAIN_STEPS} completed. "
                f"Objective: {group_objective.item():.6f} --"
            )


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    environment = Environment()

    train(device, model, tokenizer, environment)


if __name__ == "__main__":
    main()
