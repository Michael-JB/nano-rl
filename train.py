import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_scheduler,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "What number between 0 and 9 am I thinking of right now? Answer only with a single integer."
ROLLOUT_COUNT = 10
MAX_ROLLOUT_TOKENS = 20
TRAIN_STEPS = 50


# A simple reward to favour 7
def seven_reward(response: str) -> int:
    return 1 if "7" in response else 0


@torch.no_grad()
def rollout_group(
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    count: int,
) -> list[tuple[torch.Tensor, list[bool]]]:
    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(device)

    def rollout() -> tuple[torch.Tensor, list[bool]]:
        # We squeeze as we're working with batch size 1
        completion = model.generate(inputs, max_new_tokens=MAX_ROLLOUT_TOKENS).squeeze(
            0
        )
        prompt_length = len(inputs[0])
        mask = [False] * prompt_length + [True] * (len(completion) - prompt_length)
        assert len(mask) == len(completion)
        return completion, mask

    return [rollout() for _ in range(count)]


def rollout_group_loss(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    completions: list[tuple[torch.Tensor, list[bool]]],
) -> torch.Tensor:
    def completion_loss(completion: torch.Tensor, mask: list[bool]) -> torch.Tensor:
        # for i, (completion, mask) in enumerate(completions):
        # Compute reward (we skip special tokens to strip the EOS token)
        response = completion[mask]
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        reward = seven_reward(response_text)
        print(f"Reward: {reward}; Response: {response_text}")

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

        return -response_log_probs_sum * reward

    # Currently not batched for simplicity
    losses = [completion_loss(completion, mask) for completion, mask in completions]
    loss = torch.stack(losses).mean()

    return loss


def train(
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=20,
        num_training_steps=TRAIN_STEPS,
    )
    for step in range(TRAIN_STEPS):
        model.eval()
        with torch.no_grad():
            completions = rollout_group(device, tokenizer, model, count=ROLLOUT_COUNT)
        model.train()
        loss = rollout_group_loss(tokenizer, model, completions)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(f"Step {step + 1}/{TRAIN_STEPS}, Loss: {loss.item():.8f}")


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train(device, tokenizer, model)


if __name__ == "__main__":
    main()
