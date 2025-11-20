import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from environment import Environment
from train import TrainConfig, rollout, train


def model_accuracy(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: Environment,
    rollout_count: int = 10,
) -> float:
    rollouts = [
        rollout(device, model, tokenizer, environment, 4) for _ in range(rollout_count)
    ]
    responses = [
        tokenizer.decode(rollout.response, skip_special_tokens=True)
        for rollout in rollouts
    ]
    correct_rollouts = sum(
        1 for r in responses if r.strip() == str(environment.target_digit)
    )
    return correct_rollouts / rollout_count


def test_train() -> None:
    # Given
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    environment = Environment()
    train_config = TrainConfig(
        rollout_count=10,
        max_rollout_tokens=4,
        train_steps=10,
    )
    assert (
        model_accuracy(
            device=device,
            model=model,
            tokenizer=tokenizer,
            environment=environment,
        )
        < 0.2
    )

    # When
    train(device, model, tokenizer, environment, train_config)

    # Then
    assert (
        model_accuracy(
            device=device,
            model=model,
            tokenizer=tokenizer,
            environment=environment,
        )
        >= 0.8
    )
