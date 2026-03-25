"""Compression strategies for trimtoken."""

from .drop import DropStrategy
from .headtail import HeadTailStrategy
from .paraphrase import ParaphraseStrategy
from .summarize import SummarizeStrategy
from .cascade import CascadeStrategy

__all__ = [
    "DropStrategy",
    "HeadTailStrategy",
    "ParaphraseStrategy",
    "SummarizeStrategy",
    "CascadeStrategy",
]
