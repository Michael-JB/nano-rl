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

TARGET_ROW_COUNT = 5000
ANAGRAM_FILE_PATH = "anagrams.parquet"
DICTIONARY_FILE_PATH = "dictionary.parquet"
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 10


@dataclass
class AnagramRow:
    prompt: str
    answer: str


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


def make_anagram(word: str) -> str:
    letters = list(word)
    while True:
        random.shuffle(letters)
        scrambled = "".join(letters)

        if scrambled == word:
            continue

        return scrambled


def generate_anagrams(dictionary: list[str]) -> list[AnagramRow]:
    def generate_anagram_row(word: str) -> AnagramRow:
        anagram = make_anagram(word)
        prompt = (
            f"The following set of characters are an anagram of a word: "
            f"`{anagram}`. What is the original word?"
        )
        return AnagramRow(prompt=prompt, answer=word)

    return [generate_anagram_row(word) for word in dictionary]


def save_to_parquet(rows: list[dict], file_path: str) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, file_path)


if __name__ == "__main__":
    dictionary = generate_dictionary()
    anagrams = generate_anagrams(dictionary)

    save_to_parquet(
        [row.__dict__ for row in anagrams],
        ANAGRAM_FILE_PATH,
    )
    save_to_parquet(
        [{"word": word} for word in generate_dictionary()],
        DICTIONARY_FILE_PATH,
    )
