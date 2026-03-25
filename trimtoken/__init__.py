"""
trimtoken — Surgical context compression for Ollama and local LLMs.

Trim the fat. Keep the signal.
"""

from .compressor import ContextCompressor
from .models import Chunk, CompressedContext, CompressionReport, Role
from .scorer import (
    BaseScorer,
    EmbeddingScorer,
    EnsembleScorer,
    EntropyScorer,
    KeywordScorer,
    OllamaEmbeddingScorer,
    RecencyScorer,
    TFIDFScorer,
)
from .segmenter import (
    BaseSegmenter,
    MessageSegmenter,
    ParagraphSegmenter,
    SemanticSegmenter,
    SentenceSegmenter,
)
from .strategies import (
    CascadeStrategy,
    DropStrategy,
    HeadTailStrategy,
    ParaphraseStrategy,
    SummarizeStrategy,
)

__version__ = "0.1.0"
__all__ = [
    "ContextCompressor",
    "BaseScorer",
    "EnsembleScorer",
    "TFIDFScorer",
    "RecencyScorer",
    "EntropyScorer",
    "KeywordScorer",
    "EmbeddingScorer",
    "OllamaEmbeddingScorer",
    "BaseSegmenter",
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
