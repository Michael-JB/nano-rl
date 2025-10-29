import random
from pytest import fixture

from reward import digit_reward, anagram_reward


def test_digit_reward_non_digit() -> None:
    # Given
    response = "Hello, World!"
    target_digit = 5

    # When
    reward = digit_reward(response, target_digit)

    # Then
    assert reward == 0.0


def test_digit_reward_correct_digit() -> None:
    # Given
    target_digit = 5

    # When
    reward = digit_reward(str(target_digit), target_digit)

    # Then
    assert reward == 1.0


def test_digit_reward_incorrect_digit() -> None:
    # Given
    target_digit = 5
    incorrect_digit = random.choice([d for d in range(10) if d != target_digit])

    # When
    reward = digit_reward(str(incorrect_digit), target_digit)

    # Then
    assert reward > 0.0 and reward < 1.0


def test_digit_reward_scales_with_proximity() -> None:
    # Given
    target_digit, close, far = sorted(random.sample(range(10), 3))

    # When
    reward_close = digit_reward(str(close), target_digit)
    reward_far = digit_reward(str(far), target_digit)

    # Then
    assert reward_close > reward_far


#####


@fixture()
def any_anagram() -> tuple[str, str]:
    word = "example"
    anagram = "".join(random.sample(word, len(word)))
    prompt = (
        f"The following set of characters are an anagram of a word: "
        f"`{anagram}`. What is the original word?"
    )
    return (prompt, word)


def test_anagram_reward_incorrect_word(any_anagram: tuple[str, str]) -> None:
    # Given
    prompt, word = any_anagram
    response = "wrongword"

    # When
    reward = anagram_reward(prompt, response, set([word]))

    # Then
    assert reward == 0


def test_anagram_reward_not_anagram(any_anagram: tuple[str, str]) -> None:
    # Given
    prompt, word = any_anagram
    response = "nonanagram"

    # When
    reward = anagram_reward(prompt, response, set([response]))

    # Then
    assert reward == 0


def test_anagram_valid_anagram(any_anagram: tuple[str, str]) -> None:
    # Given
    prompt, word = any_anagram

    # When
    reward = anagram_reward(prompt, word, set([word]))

    # Then
    assert reward == 1
