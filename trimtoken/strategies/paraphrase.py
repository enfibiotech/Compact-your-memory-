"""ParaphraseStrategy — concise rewriting via LLM."""

from __future__ import annotations
from typing import Callable
from .drop import BaseCompressionStrategy
from ..models import Chunk

DEFAULT_PROMPT = (
    "Rewrite the following text more concisely, targeting {ratio:.0%} of the original "
    "length. Preserve all key facts and named entities. Output only the rewritten text.\n\n{content}"
)


class ParaphraseStrategy(BaseCompressionStrategy):
    """
    Rewrites chunks more concisely, preserving key facts.
    More expensive than SummarizeStrategy but higher-fidelity output.

    Args:
        llm_fn:        callable(prompt: str) -> str
        target_ratio:  target output/input token ratio (default: 0.5)
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        target_ratio: float = 0.5,
        prompt_template: str | None = None,
    ):
        self.llm_fn = llm_fn
        self.target_ratio = target_ratio
        self.prompt_template = prompt_template or DEFAULT_PROMPT

    def compress(self, chunks: list[Chunk]) -> list[Chunk]:
        result = []
        for chunk in chunks:
            prompt = self.prompt_template.format(
                ratio=self.target_ratio,
                content=chunk.content,
            )
            try:
                paraphrased = self.llm_fn(prompt)
            except Exception:
                paraphrased = chunk.content

            chunk.content = paraphrased
            chunk.token_count = len(paraphrased.split())
            chunk.metadata["paraphrased"] = True
            result.append(chunk)

        return result
