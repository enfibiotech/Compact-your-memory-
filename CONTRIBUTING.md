# Contributing to trimtoken

Thank you for helping make trimtoken better! This guide covers everything you need to get started.

## Development setup

```bash
git clone https://github.com/your-org/trimtoken
cd trimtoken
pip install -e ".[all,dev]"
```

Run the test suite:

```bash
pytest tests/ -v
pytest tests/ --cov=trimtoken --cov-report=term-missing
```

Lint and type-check:

```bash
ruff check trimtoken/
mypy trimtoken/
```

## Project structure

```
trimtoken/
├── trimtoken/
│   ├── compressor.py       # Main ContextCompressor class
│   ├── scorer.py           # All scorer implementations
│   ├── segmenter.py        # All segmenter implementations
│   ├── budget.py           # BudgetResolver
│   ├── models.py           # Data classes
│   ├── exceptions.py       # Custom exceptions
│   ├── cli.py              # CLI entry point
│   ├── strategies/         # Compression strategies
│   └── integrations/       # Provider integrations
├── tests/
├── examples/
└── docs/
```

## Areas where help is most wanted

- **New scorers** — BM25, cross-encoder reranking, LLM-as-judge scoring
- **New integrations** — LlamaIndex, Haystack, CrewAI, AutoGen
- **Streaming support** — compress as tokens arrive rather than in batch
- **GGUF tokenizer** — eliminate tiktoken dependency for pure-local setups
- **Benchmarks** — more models, datasets, and answer-quality metrics
- **Documentation** — tutorials, how-tos, and worked examples

## Submitting a pull request

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes with tests
3. Run `pytest` and `ruff check` — both must pass
4. Open a PR with a clear description of what changes and why

## Adding a new scorer

Implement `BaseScorer` from `trimtoken.scorer`:

```python
from trimtoken.scorer import BaseScorer
from trimtoken.models import Chunk

class MyScorer(BaseScorer):
    def score(self, chunks: list[Chunk], query: str | None = None) -> list[Chunk]:
        for chunk in chunks:
            # assign chunk.score between 0.0 and 1.0
            chunk.score = self._clamp(your_scoring_logic(chunk))
        return chunks
```

Add it to `trimtoken/scorer.py`, export it from `trimtoken/__init__.py`, and add tests in `tests/test_scorers.py`.

## Adding a new integration

Create `trimtoken/integrations/yourplatform.py` following the pattern of `ollama.py` or `openai_compat.py`. Export it from `trimtoken/integrations/__init__.py`.

## Code style

- Python 3.10+ features are fine (`match`, `X | Y` unions, etc.)
- Type annotations on all public functions
- Docstrings on all public classes and methods
- `ruff` for formatting and linting

## Reporting bugs

Open an issue with:
- trimtoken version (`pip show trimtoken`)
- Python version
- Minimal reproducible example
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
