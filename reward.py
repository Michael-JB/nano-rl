def digit_reward(response: str, digit: int) -> float:
    assert 0 <= digit <= 9, "digit must be between 0 and 9"

    # return 0 if the response is not a single integer
    if not response.isdigit() or len(response) != 1:
        return 0

    # return a reward between 0 and 1 based on the distance from target
    return 1 - abs(int(response) - digit) / 9
