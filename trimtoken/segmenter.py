"""Segmenter implementations for trimtoken."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Callable

from .models import Chunk, Role


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, messages: list[dict]) -> list[Chunk]:
        """Split messages into scored-ready chunks."""
        ...

    @staticmethod
    def _hash(content: str, position: int) -> str:
        return hashlib.md5(f"{position}:{content}".encode()).hexdigest()[:12]

    @staticmethod
    def _count_tokens(text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text.split())


class SentenceSegmenter(BaseSegmenter):
    """
    Splits each message into individual sentences.
    Uses spaCy if available, falls back to regex.

    Args:
        min_tokens: drop chunks below this size (default: 4)
        backend:    "spacy" | "regex" (default: "regex")
    """

    def __init__(self, min_tokens: int = 4, backend: str = "regex"):
        self.min_tokens = min_tokens
        self.backend = backend
        self._nlp = None

    def _get_sentences(self, text: str) -> list[str]:
        if self.backend == "spacy":
            if self._nlp is None:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            return [s.text.strip() for s in self._nlp(text).sents if s.text.strip()]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def segment(self, messages: list[dict]) -> list[Chunk]:
        chunks = []
        pos = 0
        for msg in messages:
            role = Role(msg.get("role", "user"))
            content = msg.get("content", "")
            for sentence in self._get_sentences(content):
                tc = self._count_tokens(sentence)
                if tc < self.min_tokens:
                    continue
                chunks.append(Chunk(
                    id=self._hash(sentence, pos),
                    content=sentence,
                    role=role,
                    token_count=tc,
                    metadata={"source_role": role.value},
                ))
                pos += 1
        return chunks


class ParagraphSegmenter(BaseSegmenter):
    """
    Splits on double newlines. Fast, good for structured documents.
    """

    def segment(self, messages: list[dict]) -> list[Chunk]:
        chunks = []
        pos = 0
        for msg in messages:
            role = Role(msg.get("role", "user"))
            content = msg.get("content", "")
            for para in re.split(r"\n\n+", content):
                para = para.strip()
                if not para:
                    continue
                tc = self._count_tokens(para)
                chunks.append(Chunk(
                    id=self._hash(para, pos),
                    content=para,
                    role=role,
                    token_count=tc,
                ))
                pos += 1
        return chunks


class MessageSegmenter(BaseSegmenter):
    """
    Treats each full message as one chunk (turn-level granularity).
    Best for aggressive compression of long multi-turn chats.
    """

    def segment(self, messages: list[dict]) -> list[Chunk]:
        chunks = []
        for i, msg in enumerate(messages):
            role = Role(msg.get("role", "user"))
            content = msg.get("content", "")
            tc = self._count_tokens(content)
            chunks.append(Chunk(
                id=self._hash(content, i),
                content=content,
                role=role,
                token_count=tc,
                metadata={"message_index": i},
            ))
        return chunks


class SemanticSegmenter(BaseSegmenter):
    """
    Uses embedding cosine similarity to detect topic-shift boundaries.

    Args:
        embedding_fn:      callable(texts: list[str]) -> list[list[float]]
        similarity_cutoff: boundary when sim drops below this (default: 0.75)
    """

    def __init__(self, embedding_fn: Callable, similarity_cutoff: float = 0.75):
        self.embedding_fn = embedding_fn
        self.similarity_cutoff = similarity_cutoff

    def segment(self, messages: list[dict]) -> list[Chunk]:
        import numpy as np

        # First pass: sentence-level
        base = SentenceSegmenter()
        sentences = base.segment(messages)
        if not sentences:
            return sentences

        texts = [c.content for c in sentences]
        embeddings = np.array(self.embedding_fn(texts))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        # Group by cosine similarity
        groups: list[list[Chunk]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = float(embeddings[i] @ embeddings[i - 1])
            if sim >= self.similarity_cutoff:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        # Merge groups into chunks
        merged = []
        for i, group in enumerate(groups):
            content = " ".join(c.content for c in group)
            tc = sum(c.token_count for c in group)
            merged.append(Chunk(
                id=self._hash(content, i),
                content=content,
                role=group[0].role,
                token_count=tc,
            ))
        return merged
