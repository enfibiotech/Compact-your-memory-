"""
trimtoken — Surgical context compression for Ollama and local LLMs.

Trim the fat. Keep the signal.
"""

from .compressor import ContextCompressor
from .scorer import (
    BaseScorer,
    TFIDFScorer,
    RecencyScorer,
    EntropyScorer,
    KeywordScorer,
    EmbeddingScorer,
    OllamaEmbeddingScorer,
    EnsembleScorer,
)
from .segmenter import (
    BaseSegmenter,
    SentenceSegmenter,
    ParagraphSegmenter,
    MessageSegmenter,
    SemanticSegmenter,
)
from .models import Chunk, CompressedContext, CompressionReport, Role
from .strategies import (
    DropStrategy,
    HeadTailStrategy,
    SummarizeStrategy,
    ParaphraseStrategy,
    CascadeStrategy,
)

__version__ = "0.1.0"
__all__ = [
    "ContextCompressor",
    "EnsembleScorer",
    "TFIDFScorer",
    "RecencyScorer",
    "EntropyScorer",
    "KeywordScorer",
    "EmbeddingScorer",
    "OllamaEmbeddingScorer",
    "SentenceSegmenter",
    "ParagraphSegmenter",
    "MessageSegmenter",
    "SemanticSegmenter",
    "DropStrategy",
    "HeadTailStrategy",
    "SummarizeStrategy",
    "ParaphraseStrategy",
    "CascadeStrategy",
    "Chunk",
    "CompressedContext",
    "CompressionReport",
    "Role",
]
