"""BudgetResolver — decides which chunks to keep, compress, or drop."""

from __future__ import annotations

from .models import Chunk


class BudgetResolver:
    """
    Given scored chunks and a token budget, decides which chunks to keep,
    which to compress, and which to drop — maximizing total importance score.

    Resolution order:
        1. Pin system messages + explicitly pinned chunks (never touched)
        2. Drop chunks with score <= drop_threshold
        3. Fill budget greedily by score (highest first)
        4. Chunks that didn't fit but are above drop_threshold → strategy queue

    Args:
        token_budget:          hard cap in tokens for the output context
        pinned_roles:          roles always kept (default: ["system"])
        drop_threshold:        score <= this → drop immediately (default: 0.2)
        summarize_threshold:   score between drop and this → compress (default: 0.5)
        keep_threshold:        score > this → always keep unchanged (default: 0.5)
    """

    def __init__(
        self,
        token_budget: int,
        pinned_roles: list[str] | None = None,
        drop_threshold: float = 0.2,
        summarize_threshold: float = 0.5,
        keep_threshold: float = 0.5,
    ):
        self.token_budget = token_budget
        self.pinned_roles = set(pinned_roles or ["system"])
        self.drop_threshold = drop_threshold
        self.summarize_threshold = summarize_threshold
        self.keep_threshold = keep_threshold

    def resolve(
        self, chunks: list[Chunk]
    ) -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
        """
        Returns (keep, compress, drop) chunk lists.
        Chunks in 'keep' are passed through unchanged.
        Chunks in 'compress' are sent to the active strategy.
        Chunks in 'drop' are removed.
        """
        pinned = [c for c in chunks if c.pinned or c.role.value in self.pinned_roles]
        remaining = [c for c in chunks if c not in pinned]

        pinned_tokens = sum(c.token_count for c in pinned)
        remaining_budget = self.token_budget - pinned_tokens

        # Step 1: immediate drops
        candidates = [c for c in remaining if c.score > self.drop_threshold]
        dropped = [c for c in remaining if c.score <= self.drop_threshold]

        # Step 2: greedy fill by score (highest first), preserving message order
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        kept_set: set[str] = set()
        budget_used = 0

        for chunk in sorted_candidates:
            if budget_used + chunk.token_count <= remaining_budget:
                kept_set.add(chunk.id)
                budget_used += chunk.token_count

        # Restore original ordering
        kept = [c for c in chunks if c.id in kept_set or c in pinned]
        to_compress = [c for c in candidates if c.id not in kept_set]

        return kept, to_compress, dropped
