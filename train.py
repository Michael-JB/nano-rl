import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "What number between 0 and 9 am I thinking of right now? Answer only with a single integer."
ROLLOUT_COUNT = 10


# A simple reward to favour 7
def seven_reward(response: str) -> int:
    return 1 if "7" in response else 0


def rollout(
    tokenizer: PreTrainedTokenizer, model: PreTrainedModel, count: int
) -> list[tuple[torch.Tensor, list[bool]]]:
    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )

    def complete() -> tuple[torch.Tensor, list[bool]]:
        # We squeeze as we're working with batch size 1
        completion = model.generate(inputs, max_new_tokens=100).squeeze(0)
        prompt_length = len(inputs[0])
        mask = [False] * prompt_length + [True] * (len(completion) - prompt_length)
        assert len(mask) == len(completion)
        return completion, mask

    return [complete() for _ in range(count)]


def compute_loss(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    completions: list[tuple[torch.Tensor, list[bool]]],
) -> torch.Tensor:
    # Currently not batched for simplicity
    losses = torch.zeros(len(completions))
    for i, (completion, mask) in enumerate(completions):
        # Compute reward (we skip special tokens to strip the EOS token)
        response = completion[mask]
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        reward = seven_reward(response_text)

        # A tensor with (log) probability distributions over the next token for
        # each sequence position. We truncate the final one as it doesn't make
        # sense to have a distribution past the end of the sequence. Note also
        # that there is no distribution for the first token, as there is no
        # prior context. It's for this reason that we later remove the first
        # element of the mask.
        logits = model(completion.unsqueeze(0)).logits[:, :-1, :].squeeze(0)  # T, C
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # T, C

        # pluck out the actual generated token from the vocab dimension
        conditional_tokens = completion[1:]  # T
        conditional_log_probs = log_probs[
            torch.arange(conditional_tokens.size(0)), conditional_tokens
        ]  # T

        # we only want to consider the response tokens for the loss, so apply the mask
        response_tokens_mask = torch.tensor(mask[1:])  # T
        response_log_probs = conditional_log_probs[response_tokens_mask]  # T

        response_log_probs_sum = response_log_probs.sum(dim=-1)  # 1

        losses[i] = response_log_probs_sum * reward

    return losses.mean()


def main() -> None:
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    completions = rollout(tokenizer, model, count=ROLLOUT_COUNT)
    loss = compute_loss(tokenizer, model, completions)

    print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    main()
