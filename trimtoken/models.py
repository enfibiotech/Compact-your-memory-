"""Core data models for trimtoken."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


@dataclass
class Chunk:
    """
    Atomic unit of context. Produced by a Segmenter,
    scored by a Scorer, acted on by a compression strategy.
    """
    id: str
    content: str
    role: Role
    token_count: int
    score: float = 0.0      # importance: 0.0 (discard) → 1.0 (keep)
    pinned: bool = False    # if True, never compressed/dropped
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionReport:
    """Full stats from a compression pass."""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float          # compressed / original
    chunks_dropped: int
    chunks_summarized: int
    chunks_kept: int
    scorer_used: str
    strategy_used: str
    score_distribution: list[float]   # histogram-ready
    budget_utilization: float         # compressed / budget


@dataclass
class CompressedContext:
    """Return value of ContextCompressor.compress()."""
    messages: list[dict[str, Any]]        # drop-in replacement for messages[]
    report: CompressionReport
    chunks: list[Chunk]         # inspectable scored chunks


@dataclass
class OllamaModelInfo:
    """Cached metadata fetched from Ollama /api/show."""
    name: str
    context_size: int
    family: str                 # llama, mistral, phi, gemma, qwen, etc.
