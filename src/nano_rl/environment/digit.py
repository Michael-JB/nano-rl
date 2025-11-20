from .environment import Environment


class DigitEnvironment(Environment):
    PROMPT = "What number between 0 and 9 am I thinking of right now? You MUST ONLY answer with a single integer."

    def __init__(self, target_digit: int = 5) -> None:
        self.target_digit: int = target_digit

    def prompt(self) -> list[dict[str, str]]:
        messages = [{"role": "user", "content": self.PROMPT}]
        return messages

    def reward(self, response: str) -> float:
        # return 0 if the response is not a single integer
        if not response.isdigit() or len(response) != 1:
            return 0

        # return a reward between 0 and 1 based on the distance from target
        return 1 - abs(int(response) - self.target_digit) / 9
