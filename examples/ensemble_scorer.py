"""
examples/ensemble_scorer.py
----------------------------
Advanced usage: EnsembleScorer with TF-IDF, recency decay, and keyword boosting.
Fully offline — no Ollama embedding model needed.

Run with: python examples/ensemble_scorer.py
"""

from trimtoken import (
    CascadeStrategy,
    ContextCompressor,
    DropStrategy,
    EnsembleScorer,
    HeadTailStrategy,
    KeywordScorer,
    MessageSegmenter,
    RecencyScorer,
    TFIDFScorer,
)

MESSAGES = [
    {"role": "system", "content": "You are a legal document assistant."},
    {"role": "user", "content": "I need help reviewing a software licence agreement."},
    {"role": "assistant", "content": (
        "Of course. Please share the key clauses you'd like me to review."
    )},
    {"role": "user", "content": (
        "The indemnification clause states that the vendor is liable for direct damages only,"
        " capped at the contract value."
    )},
    {"role": "assistant", "content": (
        "That's a standard limitation of liability. Direct damages are recoverable, but"
        " consequential and incidental damages are excluded. The cap at contract value is"
        " typical for SaaS agreements."
    )},
    {"role": "user", "content": (
        "What about the termination clause? It says 30 days notice for either party."
    )},
    {"role": "assistant", "content": (
        "Thirty days is reasonable for month-to-month SaaS. For annual contracts you'd"
        " typically want 60-90 days. Check if there's a cure period — most contracts allow"
        " 30 days to remedy a breach before termination."
    )},
    {"role": "user", "content": (
        "There's also a data retention clause — they keep our data for 90 days post-termination."
    )},
    {"role": "assistant", "content": (
        "Ninety days is standard for data export purposes. Ensure it includes a clause"
        " requiring certified deletion after that period, especially if you're handling"
        " personal data under GDPR or CCPA."
    )},
    {"role": "user", "content": "Can you summarise the key risks in this agreement?"},
]

QUERY = "What are the key risks in the licence agreement?"

scorer = EnsembleScorer([
    (TFIDFScorer(query_weight=0.7),              0.5),  # relevance to query
    (RecencyScorer(decay=5),                     0.3),  # prefer recent turns
    (KeywordScorer(["liability", "termination",
                    "data", "GDPR", "damages"],
                   boost=1.8),                   0.2),  # domain keywords
])

compressor = (
    ContextCompressor(token_budget=120)
    .with_scorer(scorer)
    .with_strategy(CascadeStrategy([
        HeadTailStrategy(head_tokens=30, tail_tokens=30),
        DropStrategy(),
    ]))
    .with_segmenter(MessageSegmenter())
)

result = compressor.compress(MESSAGES, query=QUERY)

print("── Chunk scores (before compression) ────────────────")
chunks_sorted = sorted(result.chunks, key=lambda c: c.score, reverse=True)
for c in result.chunks:  # preserve order for display
    bar = "█" * int(c.score * 20)
    role = c.role.value[:4].upper()
    preview = c.content[:55]
    print(f"  {c.score:.2f} {bar:<20} [{role}] {preview}")

print("\n── Report ────────────────────────────────────────────")
r = result.report
print(f"  {r.original_tokens} → {r.compressed_tokens} tokens  "
      f"({1 - r.compression_ratio:.0%} reduction)")
print(f"  Dropped: {r.chunks_dropped}   Kept: {r.chunks_kept}")
