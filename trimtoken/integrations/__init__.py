"""Integration helpers for trimtoken."""

from .anthropic import compress_for_anthropic
from .ollama import compress_for_ollama
from .openai_compat import compress_for_lm_studio, compress_for_openai_compat
from .openwebui import TrimTokenPipe

__all__ = [
    "compress_for_ollama",
    "compress_for_openai_compat",
    "compress_for_lm_studio",
    "compress_for_anthropic",
    "TrimTokenPipe",
]
