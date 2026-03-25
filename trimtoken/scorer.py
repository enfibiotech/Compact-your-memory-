"""Scorer implementations for trimtoken."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from typing import Protocol

from .models import Chunk


class ScorerProtocol(Protocol):
    """Duck-typing protocol — no inheritance required."""
    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]: ...


class BaseScorer(ABC):
    @abstractmethod
    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        """
        Accept chunks, return same list with .score populated.
        Must never raise — clamp all scores to [0.0, 1.0].
        """
        ...

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class TFIDFScorer(BaseScorer):
    """
    Scores chunks by TF-IDF relevance to the query (or to the most recent
    user message if no query is given). Pure-python, no GPU needed.

    Args:
        query_weight: how much to weight query match vs corpus rarity (0-1)
    """

    def __init__(self, query_weight: float = 0.6):
        self.query_weight = query_weight

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        if not chunks:
            return chunks

        q = query or next(
            (c.content for c in reversed(chunks) if c.role.value == "user"), ""
        )
        q_tokens = self._tokenize(q)
        docs = [self._tokenize(c.content) for c in chunks]
        N = len(docs)
        df: Counter = Counter()
        for doc in docs:
            df.update(set(doc))

        for chunk, doc in zip(chunks, docs, strict=False):
            if not doc or not q_tokens:
                chunk.score = 0.1
                continue
            score = 0.0
            for term in q_tokens:
                tf = doc.count(term) / len(doc)
                idf = math.log((N + 1) / (df.get(term, 0) + 1)) + 1
                score += tf * idf
            max_possible = sum(
                math.log((N + 1) / (df.get(t, 0) + 1)) + 1 for t in q_tokens
            )
            chunk.score = self._clamp(score / max_possible if max_possible else 0.1)

        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())


class RecencyScorer(BaseScorer):
    """
    Assigns higher scores to more recent chunks using exponential decay.

    Args:
        decay: half-life in number of chunks (default: 10)
    """

    def __init__(self, decay: float = 10.0):
        self.decay = decay

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        n = len(chunks)
        for i, chunk in enumerate(chunks):
            age = n - i - 1  # 0 = most recent
            chunk.score = self._clamp(math.exp(-age * math.log(2) / self.decay))
        return chunks


class EntropyScorer(BaseScorer):
    """
    Scores chunks by information entropy.
    High entropy = more unique, information-dense content.
    Punishes repetitive boilerplate and filler.
    """

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        scores = [self._entropy(c.content) for c in chunks]
        max_s = max(scores) if scores else 1.0
        for chunk, s in zip(chunks, scores, strict=False):
            chunk.score = self._clamp(s / max_s if max_s else 0.0)
        return chunks

    @staticmethod
    def _entropy(text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        counts = Counter(words)
        total = len(words)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())


class KeywordScorer(BaseScorer):
    """
    Boosts chunks containing user-specified high-value terms.

    Args:
        keywords: list of strings to prioritize
        boost:    score multiplier when a keyword matches (default: 1.5)
    """

    def __init__(self, keywords: list[str], boost: float = 1.5):
        self.keywords = [k.lower() for k in keywords]
        self.boost = boost

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        for chunk in chunks:
            lower = chunk.content.lower()
            if any(kw in lower for kw in self.keywords):
                chunk.score = self._clamp(chunk.score * self.boost)
            else:
                chunk.score = self._clamp(chunk.score * 0.5)
        return chunks


class EmbeddingScorer(BaseScorer):
    """
    Scores by cosine similarity between chunk embeddings and a query embedding.

    Args:
        embedding_fn: callable(texts: list[str]) -> list[list[float]]
        batch_size:   passed to embedding_fn in batches (default: 32)
    """

    def __init__(self, embedding_fn: Callable, batch_size: int = 32):
        self.embedding_fn = embedding_fn
        self.batch_size = batch_size

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        import numpy as np

        texts = [c.content for c in chunks]
        all_texts = texts + ([query] if query else [texts[-1]])
        embeddings = []
        for i in range(0, len(all_texts), self.batch_size):
            embeddings.extend(self.embedding_fn(all_texts[i:i + self.batch_size]))

        chunk_embs = np.array(embeddings[:-1])
        query_emb = np.array(embeddings[-1])

        norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        chunk_embs = chunk_embs / norms
        query_emb = query_emb / (np.linalg.norm(query_emb) or 1)

        sims = chunk_embs @ query_emb
        for chunk, sim in zip(chunks, sims, strict=False):
            chunk.score = self._clamp((float(sim) + 1) / 2)

        return chunks


class OllamaEmbeddingScorer(BaseScorer):
    """
    Uses Ollama's local embedding models for semantic scoring.
    Fully offline — no API keys, no cloud.

    Args:
        model:      Ollama embedding model (default: "nomic-embed-text")
        ollama_url: Ollama base URL (default: "http://localhost:11434")
        batch_size: chunks per embedding call (default: 32)
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        batch_size: int = 32,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        delegate = EmbeddingScorer(self._embed, self.batch_size)
        return delegate.score(chunks, query)


class EnsembleScorer(BaseScorer):
    """
    Combines multiple scorers with configurable weights.
    Final score = weighted average, clamped to [0, 1].

    Args:
        scorers:   list of (BaseScorer, weight: float) tuples
        normalize: re-normalize weights to sum to 1 (default: True)

    Example:
        scorer = EnsembleScorer([
            (TFIDFScorer(), 0.4),
            (OllamaEmbeddingScorer(), 0.4),
            (RecencyScorer(decay=8), 0.2),
        ])
    """

    def __init__(
        self,
        scorers: list[tuple[BaseScorer, float]],
        normalize: bool = True,
    ):
        self.scorers = scorers
        if normalize:
            total = sum(w for _, w in scorers)
            self.scorers = [(s, w / total) for s, w in scorers]

    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        import copy

        weighted: list[list[float]] = []
        for scorer, weight in self.scorers:
            cloned = copy.deepcopy(chunks)
            scored = scorer.score(cloned, query)
            weighted.append([(c.score * weight) for c in scored])

        for i, chunk in enumerate(chunks):
            chunk.score = self._clamp(sum(w[i] for w in weighted))

        return chunks
