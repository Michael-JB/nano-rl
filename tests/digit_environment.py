import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from nano_rl.environment import DigitEnvironment

from inference import rollout


def model_accuracy(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    environment: DigitEnvironment,
    rollout_count: int = 10,
) -> float:
    responses = [
        rollout(device, model, tokenizer, environment.prompt(), 4)
        for _ in range(rollout_count)
    ]
    correct_rollouts = sum(1 for r in responses if r == str(environment.target_digit))
    return correct_rollouts / rollout_count
