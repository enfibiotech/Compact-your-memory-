"""Anthropic integration for trimtoken."""

from __future__ import annotations

from typing import Any

from ..compressor import ContextCompressor
from ..models import CompressionReport
from ..scorer import BaseScorer, EnsembleScorer, RecencyScorer, TFIDFScorer


def compress_for_anthropic(
    messages: list[dict[str, Any]],
    token_budget: int,
    scorer: BaseScorer | None = None,
    query: str | None = None,
    **compressor_kwargs: Any,
) -> tuple[list[dict[str, Any]], CompressionReport]:
    """
    Drop-in for Anthropic's messages=[].
    Handles system prompt extraction and re-injection correctly.

    Example:
        import anthropic
        from trimtoken.integrations.anthropic import compress_for_anthropic

        messages, report = compress_for_anthropic(
            messages=my_messages,
            token_budget=90_000,
            query=user_query,
        )
        client.messages.create(model="claude-opus-4-5", messages=messages)
    """
    if scorer is None:
        scorer = EnsembleScorer([
            (TFIDFScorer(), 0.6),
            (RecencyScorer(decay=10), 0.4),
        ])

    # Extract system messages (Anthropic passes them separately)
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    compressor = ContextCompressor(
        token_budget=token_budget,
        scorer=scorer,
        preserve_roles=["system"],
        **compressor_kwargs,
    )
    result = compressor.compress(non_system, query=query)

    # Re-inject system messages at the front
    final_messages = system_msgs + result.messages
    return final_messages, result.report
