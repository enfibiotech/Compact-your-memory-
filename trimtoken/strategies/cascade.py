"""CascadeStrategy — try strategies in order until budget is met."""

from __future__ import annotations

from ..models import Chunk
from .drop import BaseCompressionStrategy


class CascadeStrategy(BaseCompressionStrategy):
    """
    Tries strategies in order; moves to the next if the budget is still over.

    Example:
        strategy = CascadeStrategy([
            HeadTailStrategy(),
            SummarizeStrategy(llm_fn),
            DropStrategy(),
        ])
    """

    def __init__(self, strategies: list[BaseCompressionStrategy]):
        self.strategies = strategies

    def compress(self, chunks: list[Chunk]) -> list[Chunk]:
        result = chunks
        for strategy in self.strategies:
            result = strategy.compress(result)
            if not result:
                break
        return result
