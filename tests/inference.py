import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)


def rollout(
    device: torch.device,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: list[dict[str, str]],
    max_response_tokens: int,
) -> str:
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(device)  # type: ignore

    # We squeeze as we're working with batch size 1
    completion = (
        model.generate(inputs, max_new_tokens=max_response_tokens, temperature=1)  # type: ignore
        .squeeze(0)
        .to(device)
    )
    response = completion[len(inputs[0]) :]
    return tokenizer.decode(response, skip_special_tokens=True)
