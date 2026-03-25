"""SummarizeStrategy — LLM-based chunk summarization."""

from __future__ import annotations

from collections.abc import Callable

from ..models import Chunk
from .drop import BaseCompressionStrategy

DEFAULT_PROMPT = (
    "Summarize the following conversation excerpt in {max_tokens} tokens or fewer. "
    "Preserve key facts, decisions, and named entities. Be concise.\n\n{content}"
)


class SummarizeStrategy(BaseCompressionStrategy):
    """
    Calls an LLM to summarize low-importance chunks.
    Groups nearby low-scored chunks into batches to reduce API calls.

    Args:
        llm_fn:             callable(prompt: str) -> str
        max_summary_tokens: target length per summary (default: 80)
        batch_size:         how many chunks to summarize together (default: 5)
        prompt_template:    override the default summarization prompt
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        max_summary_tokens: int = 80,
        batch_size: int = 5,
        prompt_template: str | None = None,
    ):
        self.llm_fn = llm_fn
        self.max_summary_tokens = max_summary_tokens
        self.batch_size = batch_size
        self.prompt_template = prompt_template or DEFAULT_PROMPT

    def compress(self, chunks: list[Chunk]) -> list[Chunk]:
        result = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            combined = "\n\n".join(c.content for c in batch)
            prompt = self.prompt_template.format(
                max_tokens=self.max_summary_tokens,
                content=combined,
            )
            try:
                summary = self.llm_fn(prompt)
            except Exception:
                summary = combined[:200] + "..."

            representative = batch[0]
            representative.content = summary
            representative.token_count = len(summary.split())
            representative.metadata["summarized"] = True
            representative.metadata["original_chunk_count"] = len(batch)
            result.append(representative)

        return result
