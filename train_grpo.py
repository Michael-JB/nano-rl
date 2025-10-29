import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_scheduler,
)
from reward import anagram_reward
import pyarrow.parquet as pq
from typing import Iterator

MODEL_NAME = "Qwen/Qwen3-0.6B"
EPOCHS = 2
BATCH_SIZE = 10
GROUP_SIZE = 5
MAX_ROLLOUT_TOKENS = 20


class AnagramDataset:
    DICTIONARY_PARQUET_PATH = "data/dictionary.parquet"
    ANAGRAMS_PARQUET_PATH = "data/anagrams.parquet"

    def __init__(self) -> None:
        self.dictionary = self.load_dictionary()
        self._anagrams_parquet = pq.ParquetFile(self.ANAGRAMS_PARQUET_PATH)

    def load_dictionary(self) -> pq.ParquetFile:
        table = pq.read_table(self.DICTIONARY_PARQUET_PATH)
        word_list = table["word"].to_pylist()
        return set(word_list)

    def iter_batches(self, batch_size: int) -> Iterator[list[str]]:
        for batch in self._anagrams_parquet.iter_batches(batch_size=batch_size):
            batch_dicts = batch.to_pylist()
            rows = [row_dict["prompt"] for row_dict in batch_dicts]
            yield rows


def to_message(prompt: str, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    messages = [
        {
            "role": "system",
            "content": "You are an anagram solver. You must ONLY return the original word and NOTHING ELSE.",
        },
        {"role": "user", "content": prompt},
    ]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokens


@torch.no_grad()
def rollout_batch(
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompts: list[str],
) -> list[list[tuple[torch.Tensor, list[bool]]]]:
    """Returns a list of groups of completions for each prompt."""

    def rollout(inputs: torch.Tensor) -> tuple[torch.Tensor, list[bool]]:
        # We squeeze as we're working with batch size 1
        completion = model.generate(inputs, max_new_tokens=MAX_ROLLOUT_TOKENS).squeeze(
            0
        )
        prompt_length = len(inputs[0])
        mask = [False] * prompt_length + [True] * (len(completion) - prompt_length)
        assert len(mask) == len(completion)
        return completion, mask

    batch = [to_message(prompt, tokenizer).to(device) for prompt in prompts]
    return [[rollout(inputs) for _ in range(GROUP_SIZE)] for inputs in batch]


def rollout_group_loss(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    completions: list[tuple[torch.Tensor, list[bool]]],
    reward_dictionary: set[str],
) -> torch.Tensor:
    def completion_loss(completion: torch.Tensor, mask: list[bool]) -> torch.Tensor:
        # Compute reward (we skip special tokens to strip the EOS token)
        prompt = completion[[not m for m in mask]]
        response = completion[mask]
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)
        reward = anagram_reward(prompt_text, response_text, reward_dictionary)
        print(f"Reward: {reward:.4f}; Response: {response_text}")

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
    dataset: AnagramDataset,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    batch_count = len(list(dataset.iter_batches(batch_size=BATCH_SIZE)))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=EPOCHS * batch_count,
    )
    for epoch in range(EPOCHS):
        for b_i, batch in enumerate(dataset.iter_batches(batch_size=BATCH_SIZE)):
            model.eval()
            with torch.no_grad():
                completions = rollout_batch(device, tokenizer, model, batch)
            model.train()
            for g_i, group in enumerate(completions):
                loss = rollout_group_loss(tokenizer, model, group, dataset.dictionary)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                print(
                    f"Epoch {epoch + 1}/{EPOCHS}, "
                    f"Batch {b_i + 1}/{batch_count}, "
                    f"Group {g_i + 1}/{len(completions)}, "
                    f"Loss: {loss.item():.4f}"
                )


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = AnagramDataset()

    train(device, tokenizer, model, dataset)


if __name__ == "__main__":
    main()
