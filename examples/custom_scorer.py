"""
examples/custom_scorer.py
--------------------------
Shows how to build a custom scorer and plug it into EnsembleScorer.
No subclassing required — just implement .score(chunks, query).

Run with: python examples/custom_scorer.py
"""

from trimtoken import ContextCompressor, DropStrategy, EnsembleScorer, MessageSegmenter, TFIDFScorer
from trimtoken.models import Chunk

# ── Custom scorer: boost chunks mentioning specific entities ──────────────────

class EntityScorer:
    """
    Boosts chunks that mention invoice numbers, order IDs, or dollar amounts.
    Works without any external dependencies.
    """

    import re
    PATTERNS = [
        re.compile(r"\b(INV|ORD|AC)-[\d]+\b", re.I),  # invoice/order IDs
        re.compile(r"\$[\d,]+(\.\d{2})?"),              # dollar amounts
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),      # dates
    ]

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        for chunk in chunks:
            matches = sum(
                1 for p in self.PATTERNS if p.search(chunk.content)
            )
            # Scale: 0 matches = 0.3, 3 matches = 0.9
            chunk.score = max(0.0, min(1.0, 0.3 + matches * 0.2))
        return chunks


# ── Plug it into EnsembleScorer ───────────────────────────────────────────────

MESSAGES = [
    {"role": "system", "content": "You are a billing assistant."},
    {"role": "user", "content": "I have a question about invoice INV-2024-998."},
    {"role": "assistant", "content": (
        "Happy to help! Invoice INV-2024-998 is for $1,250.00 due on 15/06/2024."
    )},
    {"role": "user", "content": "Who is the weather today?"},
    {"role": "assistant", "content": "I'm a billing assistant, not a weather service!"},
    {"role": "user", "content": "Can I get an extension on INV-2024-998?"},
]

scorer = EnsembleScorer([
    (TFIDFScorer(),    0.5),
    (EntityScorer(),   0.5),
])

compressor = ContextCompressor(
    token_budget=60,
    scorer=scorer,
    strategy=DropStrategy(),
    segmenter=MessageSegmenter(),
)

result = compressor.compress(MESSAGES, query="invoice extension")

print("Chunk scores:")
for chunk in result.chunks:
    bar = "█" * int(chunk.score * 15)
    print(f"  {chunk.score:.2f} {bar:<15} {chunk.content[:60]}")

print(f"\n{result.report.original_tokens} → {result.report.compressed_tokens} tokens")
