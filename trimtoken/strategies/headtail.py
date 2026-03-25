"""HeadTailStrategy — keep first N and last M tokens of each chunk."""

from __future__ import annotations
from .drop import BaseCompressionStrategy
from ..models import Chunk


class HeadTailStrategy(BaseCompressionStrategy):
    """
    Keeps the first N and last M tokens of each chunk.
    Good for preserving context edges in long documents.

    Args:
        head_tokens: tokens to keep from start (default: 50)
        tail_tokens: tokens to keep from end (default: 50)
    """

    def __init__(self, head_tokens: int = 50, tail_tokens: int = 50):
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens

    def compress(self, chunks: list[Chunk]) -> list[Chunk]:
        result = []
        for chunk in chunks:
            words = chunk.content.split()
            if len(words) <= self.head_tokens + self.tail_tokens:
                result.append(chunk)
                continue
            head = " ".join(words[:self.head_tokens])
            tail = " ".join(words[-self.tail_tokens:])
            chunk.content = f"{head} [...] {tail}"
            chunk.token_count = self.head_tokens + self.tail_tokens + 3
            result.append(chunk)
        return result
