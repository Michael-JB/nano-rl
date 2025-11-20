import random

from nano_rl.environment import DigitEnvironment


def test_digit_reward_non_digit() -> None:
    # Given
    response = "Hello, World!"
    environment = DigitEnvironment()

    # When
    reward = environment.reward(response)

    # Then
    assert reward == 0.0


def test_digit_reward_correct_digit() -> None:
    # Given
    environment = DigitEnvironment()

    # When
    reward = environment.reward(str(environment.target_digit))

    # Then
    assert reward == 1.0


def test_digit_reward_incorrect_digit() -> None:
    # Given
    environment = DigitEnvironment()
    incorrect_digit = random.choice(
        [d for d in range(10) if d != environment.target_digit]
    )

    # When
    reward = environment.reward(str(incorrect_digit))

    # Then
    assert reward > 0.0 and reward < 1.0


def test_digit_reward_scales_with_proximity() -> None:
    # Given
    target_digit, close, far = sorted(random.sample(range(10), 3))
    environment = DigitEnvironment(target_digit)

    # When
    reward_close = environment.reward(str(close))
    reward_far = environment.reward(str(far))

    # Then
    assert reward_close > reward_far
