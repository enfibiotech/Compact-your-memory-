"""Compression strategies for trimtoken."""

from .cascade import CascadeStrategy
from .drop import DropStrategy
from .headtail import HeadTailStrategy
from .paraphrase import ParaphraseStrategy
from .summarize import SummarizeStrategy

__all__ = [
    "DropStrategy",
    "HeadTailStrategy",
    "ParaphraseStrategy",
    "SummarizeStrategy",
    "CascadeStrategy",
]
