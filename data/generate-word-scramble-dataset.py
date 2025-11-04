#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "pyarrow",
#   "wordfreq",
# ]
# ///

import random
import pyarrow as pa
import pyarrow.parquet as pq
from wordfreq import top_n_list
from dataclasses import dataclass
from typing import Any

TARGET_ROW_COUNT = 5000
SCRAMBLE_FILE_PATH = "scramble.parquet"
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 10


@dataclass
class ScrambleRow:
    prompt: str
    word: str


def generate_dictionary() -> list[str]:
    words = top_n_list("en", n=TARGET_ROW_COUNT)
    filtered_words = [
        w.lower()
        for w in words
        if w.isalpha()
        and MIN_WORD_LENGTH <= len(w) <= MAX_WORD_LENGTH
        and len(set(w)) > 1  # Exclude words with all identical letters
    ]
    random.shuffle(filtered_words)
    return filtered_words


def generate_scrambles(dictionary: list[str]) -> list[ScrambleRow]:
    def generate_scramble_row(word: str) -> ScrambleRow:
        prompt = f"Create an anagram of the following word: `{word}`"
        return ScrambleRow(prompt=prompt, word=word)

    return [generate_scramble_row(word) for word in dictionary]


def save_to_parquet(rows: list[dict[Any, Any]], file_path: str) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, file_path)


if __name__ == "__main__":
    dictionary = generate_dictionary()
    scrambles = generate_scrambles(dictionary)

    save_to_parquet(
        [row.__dict__ for row in scrambles],
        SCRAMBLE_FILE_PATH,
    )
