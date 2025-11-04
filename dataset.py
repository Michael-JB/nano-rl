from abc import ABC, abstractmethod
from typing import Iterator


class Dataset(ABC):
    @abstractmethod
    def iter_batches(self, batch_size: int) -> Iterator[list[str]]:
        pass
