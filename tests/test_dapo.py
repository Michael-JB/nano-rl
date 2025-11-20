import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from nano_rl.environment import DigitEnvironment
from nano_rl.dapo.train import TrainConfig, rollout, train


def model_accuracy(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: DigitEnvironment,
    rollout_count: int = 10,
) -> float:
    rollouts = [
        rollout(device, model, tokenizer, environment, 4) for _ in range(rollout_count)
    ]
    responses = [
        tokenizer.decode(rollout.response, skip_special_tokens=True)
        for rollout in rollouts
    ]
    correct_rollouts = sum(1 for r in responses if r == str(environment.target_digit))
    return correct_rollouts / rollout_count


def test_train(device: torch.device) -> None:
    # Given
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
