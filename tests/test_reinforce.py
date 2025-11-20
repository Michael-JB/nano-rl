import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from nano_rl.environment import DigitEnvironment
from nano_rl.reinforce.train import TrainConfig, train

from digit_environment import model_accuracy


def test_train(
    device: torch.device, qwen3_0_6b: tuple[PreTrainedModel, PreTrainedTokenizer]
) -> None:
    # Given
    model, tokenizer = qwen3_0_6b
    environment = DigitEnvironment()
    assert model_accuracy(device, model, tokenizer, environment) < 0.2
    train_config = TrainConfig(
        rollout_count=30,
        max_rollout_tokens=2,
        train_steps=10,
    )

    # When
    train(device, model, tokenizer, environment, train_config)

    # Then
    assert model_accuracy(device, model, tokenizer, environment) >= 0.8
