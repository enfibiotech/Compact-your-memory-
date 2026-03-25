# Changelog

All notable changes to trimtoken will be documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2025-06-01

### Added
- `ContextCompressor` — main orchestration class with fluent builder API
- `TFIDFScorer`, `RecencyScorer`, `EntropyScorer`, `KeywordScorer` — offline scorers
- `EmbeddingScorer`, `OllamaEmbeddingScorer` — semantic scorers
- `EnsembleScorer` — composable weighted scorer combination
- `SentenceSegmenter`, `ParagraphSegmenter`, `MessageSegmenter`, `SemanticSegmenter`
- `DropStrategy`, `HeadTailStrategy`, `SummarizeStrategy`, `ParaphraseStrategy`, `CascadeStrategy`
- `BudgetResolver` — greedy budget allocation with pinning and thresholds
- `compress_for_ollama` — Ollama-native helper with auto-budget from `/api/show`
- `OllamaEmbeddingScorer` — local semantic scoring via Ollama embedding models
- `TrimTokenPipe` — Open WebUI Pipe for zero-config compression
- `compress_for_openai_compat` — LM Studio / llama.cpp / Jan / AnythingLLM support
- `compress_for_anthropic` — Anthropic messages integration
- `trimtoken` CLI with `compress`, `estimate`, and `score` subcommands
- Full `CompressionReport` with token savings, chunk stats, and budget utilization
- `score_only()` and `estimate_savings()` for inspection without compression

### Notes
- Requires Python ≥ 3.10
- TF-IDF + regex works fully offline — no API keys, no cloud dependencies
- Ollama integration requires Ollama ≥ 0.2 running locally
