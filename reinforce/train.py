from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_scheduler,
)

from environment import Environment

MODEL_NAME = "Qwen/Qwen3-0.6B"
ROLLOUT_COUNT = 10
MAX_ROLLOUT_TOKENS = 10
TRAIN_STEPS = 10


@dataclass(frozen=True)
class Experience:
    # A 1-dimensional tensor of the tokens of the combined prompt and response
    completion: torch.Tensor
    # A boolean mask of the same length as `self.completion` that marks prompt
    # tokens with `False` and response tokens with `True`.
    prompt_mask: list[bool]

    @property
    def response(self) -> torch.Tensor:
        return self.completion[self.prompt_mask]


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

    # We squeeze as we're working with batch size 1
    completion = (
        model.generate(inputs, max_new_tokens=MAX_ROLLOUT_TOKENS, temperature=1)  # type: ignore
        .squeeze(0)
        .to(device)
    )
    prompt_length = len(inputs[0])
    prompt_mask = [False] * prompt_length + [True] * (len(completion) - prompt_length)
    assert len(prompt_mask) == len(completion)

    return Experience(completion, prompt_mask)


def log_probs_sum(model: PreTrainedModel, experience: Experience) -> torch.Tensor:
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
    logits = sequence_logits[causal_prompt_mask, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # T, C
    # pluck out the actual generated token from the vocab dimension
    response_log_probs = log_probs[
        torch.arange(experience.response.size(0)), experience.response
    ]  # T
    return response_log_probs.sum(dim=-1)  # 1


def objective(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: Environment,
    experience: Experience,
) -> torch.Tensor:
    # Detokenize the response. We skip special tokens to strip the EOS token
    response = tokenizer.decode(experience.response, skip_special_tokens=True)
    reward = environment.reward(response)
    print(f"Reward: {reward:.4f}; Response: {response}")
    return log_probs_sum(model, experience) * reward


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
        num_training_steps=TRAIN_STEPS,
    )

    for step in range(TRAIN_STEPS):
        model.eval()
        with torch.no_grad():
            experiences = [
                rollout(device, model, tokenizer, environment)
                for _ in range(ROLLOUT_COUNT)
            ]

        model.train()
        objectives = [
            objective(model, tokenizer, environment, experience)
            for experience in experiences
        ]
        mean_objective = torch.stack(objectives).mean()
        mean_objective.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print(
            f"-- Train step {step + 1}/{TRAIN_STEPS} completed. "
            f"Objective: {mean_objective.item():.6f} --"
        )


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    environment = Environment()

    train(device, model, tokenizer, environment)


if __name__ == "__main__":
    main()
