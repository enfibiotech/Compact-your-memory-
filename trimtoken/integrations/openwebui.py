"""Open WebUI Pipe integration for trimtoken.

Paste this class into Open WebUI → Functions → New Function.
Compression is automatic on every request — no other code changes needed.
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = object  # type: ignore


class TrimTokenPipe:
    """
    Open WebUI Pipe that auto-compresses context on every request.

    Install: paste this entire class into Open WebUI → Functions → New Function.

    Valves (configurable in the Open WebUI UI):
        OLLAMA_URL:     str   = "http://localhost:11434"
        BUDGET_RATIO:   float = 0.85   -- fraction of model's context window
        SCORER:         str   = "ensemble" | "tfidf" | "recency"
        DROP_THRESHOLD: float = 0.20
        SHOW_REPORT:    bool  = False  -- append compression stats as a footnote
    """

    class Valves(BaseModel):
        OLLAMA_URL: str = "http://localhost:11434"
        BUDGET_RATIO: float = 0.85
        SCORER: str = "ensemble"
        DROP_THRESHOLD: float = 0.20
        SHOW_REPORT: bool = False

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Intercept request, compress messages, return updated body."""
        from trimtoken.integrations.ollama import compress_for_ollama
        from trimtoken.scorer import EnsembleScorer, RecencyScorer, TFIDFScorer

        scorer_map = {
            "tfidf": TFIDFScorer(),
            "recency": RecencyScorer(),
            "ensemble": EnsembleScorer([
                (TFIDFScorer(), 0.6),
                (RecencyScorer(decay=10), 0.4),
            ]),
        }
        scorer = scorer_map.get(self.valves.SCORER, scorer_map["ensemble"])

        latest_user = next(
            (m["content"] for m in reversed(body.get("messages", []))
             if m.get("role") == "user"),
            None,
        )

        messages, report = await compress_for_ollama(
            messages=body.get("messages", []),
            model=body.get("model", ""),
            ollama_url=self.valves.OLLAMA_URL,
            scorer=scorer,
            query=latest_user,
            budget_ratio=self.valves.BUDGET_RATIO,
            drop_threshold=self.valves.DROP_THRESHOLD,
        )

        body["messages"] = messages

        if self.valves.SHOW_REPORT:
            note = (
                f"\n\n---\n*trimtoken: {report.original_tokens} → "
                f"{report.compressed_tokens} tokens "
                f"({1 - report.compression_ratio:.0%} reduction)*"
            )
            if messages and messages[-1].get("role") == "assistant":
                messages[-1]["content"] += note

        return body
