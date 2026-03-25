"""ContextCompressor — main entry point for trimtoken."""

from __future__ import annotations

from typing import Callable

from .budget import BudgetResolver
from .models import Chunk, CompressedContext, CompressionReport
from .scorer import BaseScorer, TFIDFScorer
from .segmenter import BaseSegmenter, SentenceSegmenter
from .strategies.drop import BaseCompressionStrategy, DropStrategy


class ContextCompressor:
    """
    Orchestrates: segment → score → resolve budget → compress.

    Args:
        token_budget:   hard token cap for the output context
        scorer:         a BaseScorer instance (default: TFIDFScorer)
        strategy:       a BaseCompressionStrategy (default: DropStrategy)
        segmenter:      how to split messages (default: SentenceSegmenter)
        tokenizer:      callable(text: str) -> int (default: tiktoken cl100k)
        preserve_roles: roles never touched (default: ["system"])
        **budget_config: kwargs forwarded to BudgetResolver

    Basic usage:
        compressor = ContextCompressor(
            token_budget=4096,
            scorer=TFIDFScorer(),
            strategy=DropStrategy(),
        )
        result = compressor.compress(messages, query="What was the budget?")
        new_messages = result.messages

    Fluent builder:
        compressor = (
            ContextCompressor(token_budget=4096)
            .with_scorer(EnsembleScorer([...]))
            .with_strategy(CascadeStrategy([...]))
            .with_segmenter(SemanticSegmenter(embed_fn))
        )
    """

    def __init__(
        self,
        token_budget: int,
        scorer: BaseScorer | None = None,
        strategy: BaseCompressionStrategy | None = None,
        segmenter: BaseSegmenter | None = None,
        tokenizer: Callable[[str], int] | None = None,
        preserve_roles: list[str] | None = None,
        **budget_config,
    ):
        self.token_budget = token_budget
        self.scorer = scorer or TFIDFScorer()
        self.strategy = strategy or DropStrategy()
        self.segmenter = segmenter or SentenceSegmenter()
        self.tokenizer = tokenizer or self._default_tokenizer
        self.preserve_roles = preserve_roles or ["system"]
        self._budget_config = budget_config

    # ── fluent builder ────────────────────────────────────────────────────

    def with_scorer(self, scorer: BaseScorer) -> "ContextCompressor":
        self.scorer = scorer
        return self

    def with_strategy(self, strategy: BaseCompressionStrategy) -> "ContextCompressor":
        self.strategy = strategy
        return self

    def with_segmenter(self, segmenter: BaseSegmenter) -> "ContextCompressor":
        self.segmenter = segmenter
        return self

    # ── core ──────────────────────────────────────────────────────────────

    def compress(
        self,
        messages: list[dict],
        query: str | None = None,
        pin: list[int] | None = None,
    ) -> CompressedContext:
        """
        Compress messages to fit within token_budget.

        Args:
            messages: standard {role, content} list
            query:    optional query to anchor relevance scoring
            pin:      message indices to pin (never compressed)
        """
        pin_set = set(pin or [])

        # 1. Segment
        chunks = self.segmenter.segment(messages)

        # 2. Mark pinned
        for chunk in chunks:
            if chunk.metadata.get("message_index") in pin_set:
                chunk.pinned = True

        original_tokens = sum(c.token_count for c in chunks)

        # 3. Score
        chunks = self.scorer.score(chunks, query)

        # 4. Resolve budget
        resolver = BudgetResolver(
            token_budget=self.token_budget,
            pinned_roles=self.preserve_roles,
            **self._budget_config,
        )
        keep, to_compress, dropped = resolver.resolve(chunks)

        # 5. Apply strategy to compression queue
        compressed_chunks = self.strategy.compress(to_compress)

        # 6. Reconstruct messages (preserve original ordering)
        final_chunk_ids = {c.id for c in keep} | {c.id for c in compressed_chunks}
        all_final = [c for c in chunks if c.id in final_chunk_ids]

        messages_out = self._chunks_to_messages(all_final)
        compressed_tokens = sum(c.token_count for c in all_final)

        report = CompressionReport(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens else 1.0,
            chunks_dropped=len(dropped),
            chunks_summarized=sum(
                1 for c in compressed_chunks if c.metadata.get("summarized")
            ),
            chunks_kept=len(keep),
            scorer_used=type(self.scorer).__name__,
            strategy_used=type(self.strategy).__name__,
            score_distribution=[c.score for c in chunks],
            budget_utilization=compressed_tokens / self.token_budget if self.token_budget else 1.0,
        )

        return CompressedContext(
            messages=messages_out,
            report=report,
            chunks=all_final,
        )

    def score_only(
        self, messages: list[dict], query: str | None = None
    ) -> list[Chunk]:
        """Score without compressing. Useful for inspection."""
        chunks = self.segmenter.segment(messages)
        return self.scorer.score(chunks, query)

    def estimate_savings(
        self, messages: list[dict], query: str | None = None
    ) -> CompressionReport:
        """Dry-run: report what would be compressed without modifying messages."""
        result = self.compress(messages, query)
        return result.report

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _chunks_to_messages(chunks: list[Chunk]) -> list[dict]:
        """Re-merge chunks back into role-grouped messages."""
        if not chunks:
            return []
        messages = []
        current_role = chunks[0].role.value
        current_parts = [chunks[0].content]

        for chunk in chunks[1:]:
            if chunk.role.value == current_role:
                current_parts.append(chunk.content)
            else:
                messages.append({"role": current_role, "content": " ".join(current_parts)})
                current_role = chunk.role.value
                current_parts = [chunk.content]

        messages.append({"role": current_role, "content": " ".join(current_parts)})
        return messages

    @staticmethod
    def _default_tokenizer(text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text.split())
