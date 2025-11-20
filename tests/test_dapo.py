import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from nano_rl.environment import DigitEnvironment
from nano_rl.dapo.train import TrainConfig, train

from digit_environment import model_accuracy


def test_train(
    device: torch.device, qwen3_0_6b: tuple[PreTrainedModel, PreTrainedTokenizer]
) -> None:
    # Given
    model, tokenizer = qwen3_0_6b
    environment = DigitEnvironment()
    assert model_accuracy(device, model, tokenizer, environment) < 0.2
    train_config = TrainConfig(
        group_size=6,
        max_rollout_tokens=4,
        train_steps=4,
        train_batch_size=2,
        dynamic_sampling_buffer_capacity=6,
        epsilon_low=0.2,
        epsilon_high=0.28,
    )

    # When
    train(device, model, tokenizer, environment, train_config)

    # Then
    assert model_accuracy(device, model, tokenizer, environment) >= 0.8
