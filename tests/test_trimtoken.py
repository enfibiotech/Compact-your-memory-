"""Tests for trimtoken core components."""

from trimtoken import (
    CascadeStrategy,
    ContextCompressor,
    DropStrategy,
    EnsembleScorer,
    EntropyScorer,
    HeadTailStrategy,
    KeywordScorer,
    MessageSegmenter,
    RecencyScorer,
    SentenceSegmenter,
    TFIDFScorer,
)
from trimtoken.budget import BudgetResolver
from trimtoken.models import Chunk, Role

# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! I need help with my invoice. The total is $500."},
    {"role": "assistant", "content": "Sure, I can help with your invoice."},
    {"role": "user", "content": "Actually, the invoice number is INV-2024-001."},
    {"role": "assistant", "content": "Got it. Invoice INV-2024-001 for $500."},
    {"role": "user", "content": "Can you summarize what we discussed?"},
]


def make_chunk(content: str, role: str = "user", score: float = 0.5) -> Chunk:
    return Chunk(
        id=str(hash(content)),
        content=content,
        role=Role(role),
        token_count=len(content.split()),
        score=score,
    )


# ── segmenters ────────────────────────────────────────────────────────────────

class TestSentenceSegmenter:
    def test_basic_segmentation(self):
        seg = SentenceSegmenter()
        chunks = seg.segment(SAMPLE_MESSAGES)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_roles_preserved(self):
        seg = SentenceSegmenter()
        chunks = seg.segment(SAMPLE_MESSAGES)
        roles = {c.role.value for c in chunks}
        assert "system" in roles
        assert "user" in roles

    def test_min_tokens_filter(self):
        seg = SentenceSegmenter(min_tokens=100)
        chunks = seg.segment(SAMPLE_MESSAGES)
        # Very high min_tokens should produce fewer or zero chunks
        assert len(chunks) < len(SAMPLE_MESSAGES) * 3


class TestMessageSegmenter:
    def test_one_chunk_per_message(self):
        seg = MessageSegmenter()
        chunks = seg.segment(SAMPLE_MESSAGES)
        assert len(chunks) == len(SAMPLE_MESSAGES)

    def test_content_preserved(self):
        seg = MessageSegmenter()
        chunks = seg.segment(SAMPLE_MESSAGES)
        assert chunks[0].content == "You are a helpful assistant."


# ── scorers ───────────────────────────────────────────────────────────────────

class TestTFIDFScorer:
    def test_scores_in_range(self):
        seg = MessageSegmenter()
        chunks = seg.segment(SAMPLE_MESSAGES)
        scorer = TFIDFScorer()
        scored = scorer.score(chunks, query="invoice")
        assert all(0.0 <= c.score <= 1.0 for c in scored)

    def test_relevant_chunk_higher_score(self):
        scorer = TFIDFScorer()
        chunks = [
            make_chunk("The invoice total is five hundred dollars INV-2024"),
            make_chunk("Hello how are you doing today"),
        ]
        scorer.score(chunks, query="invoice")
        assert chunks[0].score > chunks[1].score


class TestRecencyScorer:
    def test_recent_higher(self):
        scorer = RecencyScorer(decay=5)
        chunks = [make_chunk(f"message {i}") for i in range(5)]
        scorer.score(chunks)
        # Last chunk (most recent) should have highest score
        assert chunks[-1].score > chunks[0].score

    def test_scores_in_range(self):
        scorer = RecencyScorer()
        chunks = [make_chunk(f"msg {i}") for i in range(10)]
        scorer.score(chunks)
        assert all(0.0 <= c.score <= 1.0 for c in chunks)


class TestEntropyScorer:
    def test_unique_content_higher(self):
        scorer = EntropyScorer()
        chunks = [
            make_chunk("the the the the the the the"),  # low entropy
            make_chunk("invoice contract payment deadline client meeting"),  # high entropy
        ]
        scorer.score(chunks)
        assert chunks[1].score > chunks[0].score


class TestKeywordScorer:
    def test_keyword_boost(self):
        scorer = KeywordScorer(keywords=["invoice"], boost=2.0)
        chunks = [
            make_chunk("invoice number INV-001", score=0.5),
            make_chunk("hello world today", score=0.5),
        ]
        scorer.score(chunks)
        assert chunks[0].score > chunks[1].score


class TestEnsembleScorer:
    def test_normalized_weights(self):
        scorer = EnsembleScorer([
            (TFIDFScorer(), 2.0),
            (RecencyScorer(), 2.0),
        ], normalize=True)
        seg = MessageSegmenter()
        chunks = seg.segment(SAMPLE_MESSAGES)
        scored = scorer.score(chunks, query="invoice")
        assert all(0.0 <= c.score <= 1.0 for c in scored)


# ── budget resolver ───────────────────────────────────────────────────────────

class TestBudgetResolver:
    def test_system_always_pinned(self):
        seg = MessageSegmenter()
        chunks = seg.segment(SAMPLE_MESSAGES)
        for c in chunks:
            c.score = 0.1  # all low score

        resolver = BudgetResolver(token_budget=1000, pinned_roles=["system"])
        keep, compress, drop = resolver.resolve(chunks)

        system_in_keep = any(c.role.value == "system" for c in keep)
        assert system_in_keep

    def test_low_score_chunks_dropped(self):
        chunks = [make_chunk(f"content {i}", score=0.1) for i in range(5)]
        resolver = BudgetResolver(token_budget=10000, drop_threshold=0.2)
        keep, compress, drop = resolver.resolve(chunks)
        assert len(drop) == 5

    def test_budget_respected(self):
        chunks = [make_chunk("word " * 50, score=0.9) for _ in range(20)]
        resolver = BudgetResolver(token_budget=100)
        keep, compress, drop = resolver.resolve(chunks)
        total = sum(c.token_count for c in keep)
        assert total <= 100


# ── strategies ────────────────────────────────────────────────────────────────

class TestDropStrategy:
    def test_returns_empty(self):
        strategy = DropStrategy()
        chunks = [make_chunk("some content")]
        assert strategy.compress(chunks) == []


class TestHeadTailStrategy:
    def test_long_chunk_truncated(self):
        strategy = HeadTailStrategy(head_tokens=5, tail_tokens=5)
        chunk = make_chunk(" ".join([f"word{i}" for i in range(100)]))
        result = strategy.compress([chunk])
        assert "[...]" in result[0].content
        assert result[0].token_count <= 13  # 5 + 5 + separator

    def test_short_chunk_unchanged(self):
        strategy = HeadTailStrategy(head_tokens=50, tail_tokens=50)
        chunk = make_chunk("short content here")
        result = strategy.compress([chunk])
        assert result[0].content == "short content here"


class TestCascadeStrategy:
    def test_cascade_applies_first_strategy(self):
        strategy = CascadeStrategy([DropStrategy()])
        chunks = [make_chunk("some content")]
        result = strategy.compress(chunks)
        assert result == []


# ── compressor integration ────────────────────────────────────────────────────

class TestContextCompressor:
    def test_basic_compress(self):
        compressor = ContextCompressor(
            token_budget=50,
            scorer=TFIDFScorer(),
            strategy=DropStrategy(),
            segmenter=MessageSegmenter(),
        )
        result = compressor.compress(SAMPLE_MESSAGES, query="invoice")
        assert result.messages is not None
        assert result.report.original_tokens > result.report.compressed_tokens

    def test_system_preserved(self):
        compressor = ContextCompressor(
            token_budget=30,
            scorer=TFIDFScorer(),
            strategy=DropStrategy(),
            segmenter=MessageSegmenter(),
            preserve_roles=["system"],
        )
        result = compressor.compress(SAMPLE_MESSAGES)
        roles = [m["role"] for m in result.messages]
        assert "system" in roles

    def test_fluent_builder(self):
        compressor = (
            ContextCompressor(token_budget=200)
            .with_scorer(TFIDFScorer())
            .with_strategy(DropStrategy())
            .with_segmenter(MessageSegmenter())
        )
        result = compressor.compress(SAMPLE_MESSAGES)
        assert result.report is not None

    def test_score_only(self):
        compressor = ContextCompressor(token_budget=4096)
        chunks = compressor.score_only(SAMPLE_MESSAGES, query="invoice")
        assert all(0.0 <= c.score <= 1.0 for c in chunks)

    def test_estimate_savings(self):
        compressor = ContextCompressor(
            token_budget=30,
            segmenter=MessageSegmenter(),
        )
        report = compressor.estimate_savings(SAMPLE_MESSAGES)
        assert report.original_tokens > 0
        assert 0.0 <= report.compression_ratio <= 1.0

    def test_report_fields(self):
        compressor = ContextCompressor(
            token_budget=50,
            segmenter=MessageSegmenter(),
        )
        result = compressor.compress(SAMPLE_MESSAGES)
        r = result.report
        assert r.chunks_dropped + r.chunks_summarized + r.chunks_kept >= 0
        assert r.scorer_used == "TFIDFScorer"
