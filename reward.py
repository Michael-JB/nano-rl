def digit_reward(response: str, digit: int) -> float:
    assert 0 <= digit <= 9, "digit must be between 0 and 9"

    # return 0 if the response is not a single integer
    if not response.isdigit() or len(response) != 1:
        return 0

    # return a reward between 0 and 1 based on the distance from target
    return 1 - abs(int(response) - digit) / 9


def anagram_reward(prompt: str, response: str, dictionary: set[str]) -> int:
    # Assumes prompt format in dataset
    anagram = prompt.split("`")[1]

    # return 0 if response is a valid word
    if response not in dictionary:
        return 0

    # return 0 if response is not an anagram of the prompt
    if sorted(response) != sorted(anagram):
        return 0

    return 1
