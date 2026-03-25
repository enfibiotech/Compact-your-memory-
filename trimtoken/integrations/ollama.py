"""Ollama-native integration for trimtoken."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..scorer import BaseScorer

from ..compressor import ContextCompressor
from ..models import CompressionReport
from ..scorer import EnsembleScorer, RecencyScorer, TFIDFScorer

_CONTEXT_CACHE: dict[str, int] = {}


async def get_model_context_size(
    model: str,
    ollama_url: str = "http://localhost:11434",
) -> int:
    """
    Fetch the model's num_ctx from Ollama /api/show.
    Falls back to 4096 if not found. Results are cached per model.
    """
    if model in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[model]

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{ollama_url}/api/show",
                json={"name": model},
            )
            resp.raise_for_status()
            data = resp.json()
            params = data.get("parameters", "")
            for line in params.splitlines():
                if "num_ctx" in line:
                    ctx = int(line.split()[-1])
                    _CONTEXT_CACHE[model] = ctx
                    return ctx
    except Exception:
        pass

    _CONTEXT_CACHE[model] = 4096
    return 4096


async def compress_for_ollama(
    messages: list[dict[str, Any]],
    model: str,
    ollama_url: str = "http://localhost:11434",
    scorer: BaseScorer | None = None,
    query: str | None = None,
    budget_ratio: float = 0.85,
    **compressor_kwargs: Any,
) -> tuple[list[dict[str, Any]], CompressionReport]:
    """
    One-liner drop-in for Ollama's messages=[].

    Automatically fetches the model's context window from Ollama and
    computes a token budget — no token_budget argument needed.

    Args:
        messages:     standard {role, content} list
        model:        Ollama model name (e.g. "llama3.2:3b", "mistral", "phi3")
        ollama_url:   Ollama base URL (default: http://localhost:11434)
        scorer:       scorer to use (default: EnsembleScorer with TF-IDF + Recency)
        query:        optional query to anchor relevance scoring
        budget_ratio: fraction of the model's context window to target (default: 0.85)
        **compressor_kwargs: forwarded to ContextCompressor

    Returns:
        (compressed_messages, CompressionReport)

    Example:
        import ollama
        from trimtoken.integrations.ollama import compress_for_ollama

        messages, report = await compress_for_ollama(
            messages=history,
            model="llama3.2:3b",
            query=latest_user_message,
        )
        print(f"{report.original_tokens} → {report.compressed_tokens} tokens")
        response = ollama.chat(model="llama3.2:3b", messages=messages)
    """
    ctx_size = await get_model_context_size(model, ollama_url)
    token_budget = int(ctx_size * budget_ratio)

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
