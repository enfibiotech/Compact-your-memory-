"""OpenAI-compatible integration for trimtoken.

Works with LM Studio, llama.cpp server, Jan, AnythingLLM,
and any endpoint that speaks the OpenAI chat completions format.
"""

from __future__ import annotations

from ..compressor import ContextCompressor
from ..models import CompressionReport
from ..scorer import BaseScorer, EnsembleScorer, RecencyScorer, TFIDFScorer


def compress_for_openai_compat(
    messages: list[dict],
    token_budget: int,
    scorer: BaseScorer | None = None,
    query: str | None = None,
    **compressor_kwargs,
) -> tuple[list[dict], CompressionReport]:
    """
    Drop-in for any OpenAI-compatible messages=[].
    Works with LM Studio, llama.cpp, Jan, AnythingLLM, etc.

    Args:
        messages:     standard {role, content} list
        token_budget: hard token cap (check your model's context size)
        scorer:       scorer to use (default: EnsembleScorer)
        query:        optional query to anchor relevance scoring

    Example:
        from openai import OpenAI
        from trimtoken.integrations.openai_compat import compress_for_openai_compat

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        messages, report = compress_for_openai_compat(
            messages=history,
            token_budget=8192,
            query=latest_user_message,
        )
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
        )
    """
    if scorer is None:
        scorer = EnsembleScorer([
            (TFIDFScorer(), 0.6),
            (RecencyScorer(decay=10), 0.4),
        ])

    compressor = ContextCompressor(
        token_budget=token_budget,
        scorer=scorer,
        **compressor_kwargs,
    )
    result = compressor.compress(messages, query=query)
    return result.messages, result.report


# Alias for clarity
compress_for_lm_studio = compress_for_openai_compat
compress_for_llamacpp = compress_for_openai_compat
