"""DropStrategy — remove chunks entirely."""

from __future__ import annotations
from abc import ABC, abstractmethod
from ..models import Chunk


class BaseCompressionStrategy(ABC):
    @abstractmethod
    def compress(self, chunks: list[Chunk]) -> list[Chunk]:
        """Compress the given chunks. Must preserve ordering."""
        ...


class DropStrategy(BaseCompressionStrategy):
    """Removes chunks entirely. Fastest, most lossy."""

    def compress(self, chunks: list[Chunk]) -> list[Chunk]:
        return []
