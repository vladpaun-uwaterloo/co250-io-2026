from dataclasses import dataclass
from collections.abc import Iterable
from typing import Self


@dataclass(init=False)
class Basis:
    indices: list[int]

    def __init__(self, indices: Iterable[int]):
        self.indices = list(indices)

    def replace(self,leaving: int,entering: int) -> Self:
        self.indices[leaving] = entering
        return self

    def remove(self, index: int) -> Self:
        del self.indices[index]
        return self