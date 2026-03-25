"""Custom exceptions for trimtoken."""


class TrimTokenError(Exception):
    """Base exception for all trimtoken errors."""


class BudgetExceededError(TrimTokenError):
    """Raised when pinned content alone exceeds the token budget."""

    def __init__(self, pinned_tokens: int, budget: int):
        super().__init__(
            f"Pinned content ({pinned_tokens} tokens) exceeds token budget ({budget}). "
            "Increase token_budget or reduce preserve_roles."
        )


class OllamaConnectionError(TrimTokenError):
    """Raised when trimtoken cannot connect to the Ollama server."""

    def __init__(self, url: str, original: Exception):
        super().__init__(
            f"Cannot connect to Ollama at {url}: {original}. "
            "Is Ollama running? Try: ollama serve"
        )


class ScorerError(TrimTokenError):
    """Raised when a scorer fails to produce valid scores."""


class StrategyError(TrimTokenError):
    """Raised when a compression strategy fails."""


class TokenizerError(TrimTokenError):
    """Raised when the tokenizer cannot count tokens."""
