"""
examples/openwebui_pipe.py
--------------------------
Shows the full TrimTokenPipe class ready to paste into Open WebUI.

Instructions:
  1. Open your Open WebUI instance
  2. Go to Workspace → Functions → + New Function
  3. Paste the TrimTokenPipe class below
  4. Click Save — compression is now active for every chat

The pipe automatically reads each model's context window size from
Ollama via /api/show, so no manual configuration is needed.
"""

# ── Paste this into Open WebUI → Functions ────────────────────────────────────

from pydantic import BaseModel


class TrimTokenPipe:
    """
    trimtoken — Auto-compress context on every Open WebUI request.

    Valves (configure in the UI after installing):
        OLLAMA_URL:     Your Ollama server URL
        BUDGET_RATIO:   Fraction of model context to target (0.85 = 85%)
        SCORER:         ensemble | tfidf | recency
        DROP_THRESHOLD: Chunks below this score are dropped (0.0–1.0)
        SHOW_REPORT:    Append token savings as a footnote to replies
    """

    class Valves(BaseModel):
        OLLAMA_URL: str = "http://localhost:11434"
        BUDGET_RATIO: float = 0.85
        SCORER: str = "ensemble"
        DROP_THRESHOLD: float = 0.20
        SHOW_REPORT: bool = False

    def __init__(self):
        self.valves = self.Valves()

    async def pipe(self, body: dict, __user__: dict | None = None) -> dict:
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
            reduction = 1 - report.compression_ratio
            note = (
                f"\n\n---\n"
                f"*✂️ trimtoken: {report.original_tokens:,} → "
                f"{report.compressed_tokens:,} tokens "
                f"({reduction:.0%} reduction)*"
            )
            if messages and messages[-1].get("role") == "assistant":
                messages[-1]["content"] += note

        return body

# ── end paste ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Paste the TrimTokenPipe class above into Open WebUI → Functions.")
    print("See the README for full setup instructions.")
