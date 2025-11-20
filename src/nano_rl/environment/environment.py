from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def prompt(self) -> list[dict[str, str]]:
        """Return the prompt for the environment in message format."""
        ...

    @abstractmethod
    def reward(self, response: str) -> float:
        """Score the response with a reward between 0 and 1 (inclusive)."""
        ...
