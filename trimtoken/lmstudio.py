"""
test_lmstudio.py
----------------
Test trimtoken with LM Studio's local OpenAI-compatible API.

Requirements:
    1. LM Studio installed — https://lmstudio.ai
    2. A model downloaded and loaded in LM Studio
    3. Local Server started in LM Studio (default: http://localhost:1234)
    4. pip install -e ".[openai]"

Run with:
    python test_lmstudio.py
"""

from openai import OpenAI
from trimtoken.integrations.openai_compat import compress_for_openai_compat
from trimtoken import EnsembleScorer
from trimtoken.scorer import TFIDFScorer, RecencyScorer, KeywordScorer

# ── Config ────────────────────────────────────────────────────────────────────

LM_STUDIO_URL = "http://localhost:1234/v1"
API_KEY       = "lm-studio"          # LM Studio accepts any non-empty string
TOKEN_BUDGET  = 2048                 # safe for most models in LM Studio
MAX_RESPONSE  = 300                  # tokens for the model reply

# ── Simulated long conversation ───────────────────────────────────────────────

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a knowledgeable assistant helping a developer understand "
            "Python best practices and software architecture."
        ),
    },
    {
        "role": "user",
        "content": "Can you explain what a context manager is in Python?",
    },
    {
        "role": "assistant",
        "content": (
            "A context manager in Python is an object that defines the runtime "
            "context for a with statement. It handles setup and teardown "
            "automatically, for example opening and closing a file. You implement "
            "one by defining __enter__ and __exit__ methods, or using the "
            "@contextmanager decorator from the contextlib module."
        ),
    },
    {
        "role": "user",
        "content": "What is the difference between a list and a tuple?",
    },
    {
        "role": "assistant",
        "content": (
            "Lists are mutable. You can add, remove, or change elements after "
            "creation. Tuples are immutable. Once created they cannot be changed. "
            "Lists use square brackets, tuples use parentheses. Tuples are "
            "slightly faster and are used for fixed collections, while "
            "lists are for ordered collections that may change."
        ),
    },
    {
        "role": "user",
        "content": "How does Python's GIL affect multithreading?",
    },
    {
        "role": "assistant",
        "content": (
            "The Global Interpreter Lock (GIL) is a mutex in CPython that allows "
            "only one thread to execute Python bytecode at a time. This means "
            "CPU-bound tasks do not benefit from multithreading in CPython. However "
            "IO-bound tasks like network calls and file reads do benefit because "
            "threads release the GIL while waiting. For CPU-bound parallelism "
            "use the multiprocessing module instead."
        ),
    },
    {
        "role": "user",
        "content": "What are decorators and how do they work?",
    },
    {
        "role": "assistant",
        "content": (
            "A decorator is a function that takes another function as input and "
            "returns a modified version of it. They use the @ syntax as shorthand. "
            "Under the hood @my_decorator above def foo() is equivalent to "
            "foo = my_decorator(foo). Decorators are commonly used for logging, "
            "authentication, caching, and timing without modifying the original "
            "function code."
        ),
    },
    {
        "role": "user",
        "content": "Can you explain async and await in Python?",
    },
    {
        "role": "assistant",
        "content": (
            "async and await are Python syntax for writing asynchronous code using "
            "coroutines. An async def function returns a coroutine object. The "
            "await keyword suspends execution until the awaited task completes, "
            "letting other tasks run in the meantime. This is managed by an event "
            "loop via asyncio.run(). It is ideal for IO-bound workloads like "
            "web requests or database queries."
        ),
    },
    {
        "role": "user",
        "content": "What is a generator and when should I use one?",
    },
    {
        "role": "assistant",
        "content": (
            "A generator is a function that yields values one at a time instead of "
            "returning them all at once using the yield keyword. Generators are "
            "lazy and only compute the next value when asked. This makes them "
            "memory-efficient for large datasets. Use them when processing large "
            "files, infinite sequences, or pipelines where you do not need all "
            "values in memory at the same time."
        ),
    },
    {
        "role": "user",
        "content": (
            "Given everything we discussed about context managers, the GIL, "
            "decorators, async and await, and generators — which concept should "
            "I focus on first as an intermediate Python developer?"
        ),
    },
]

QUERY = "Which Python concept should an intermediate developer focus on first?"

DIVIDER = "-" * 52


# ── Scorer ────────────────────────────────────────────────────────────────────

def build_scorer():
    return EnsembleScorer([
        (TFIDFScorer(query_weight=0.7),                           0.5),
        (RecencyScorer(decay=6),                                  0.3),
        (KeywordScorer(
            ["context manager", "GIL", "decorator",
             "async", "generator", "intermediate"],
            boost=1.6,
        ),                                                         0.2),
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_report(report):
    reduction = 1 - report.compression_ratio
    print("")
    print("  trimtoken compression report")
    print("  " + DIVIDER)
    items = [
        ("Original tokens",    f"{report.original_tokens:,}"),
        ("Compressed tokens",  f"{report.compressed_tokens:,}"),
        ("Reduction",          f"{reduction:.0%}"),
        ("Chunks kept",        str(report.chunks_kept)),
        ("Chunks dropped",     str(report.chunks_dropped)),
        ("Budget utilization", f"{report.budget_utilization:.1%}"),
        ("Scorer",             report.scorer_used),
        ("Strategy",           report.strategy_used),
    ]
    for label, value in items:
        print(f"  {label:<24}{value}")
    print("")


def print_chunk_scores(chunks):
    print(f"  {'Score':>5}  {'Role':<11} Content preview")
    print("  " + DIVIDER)
    for chunk in chunks:
        filled = int(chunk.score * 10)
        bar = "#" * filled + "." * (10 - filled)
        role = chunk.role.value[:10]
        preview = chunk.content[:48].replace("\n", " ")
        print(f"  {chunk.score:.2f}  [{bar}] [{role}] {preview}")
    print("")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("")
    print("  trimtoken x LM Studio -- Integration Test")
    print("  " + DIVIDER)
    print(f"  Server : {LM_STUDIO_URL}")
    print(f"  Budget : {TOKEN_BUDGET:,} tokens")
    print(f"  Query  : {QUERY}")
    print("")

    # Step 1: connect
    print("Step 1: Connecting to LM Studio...")
    client = OpenAI(base_url=LM_STUDIO_URL, api_key=API_KEY)

    try:
        models = client.models.list()
        loaded_model = models.data[0].id if models.data else "unknown"
        print(f"  OK -- model loaded: {loaded_model}")
        print("")
    except Exception as e:
        print(f"  FAILED to connect to LM Studio at {LM_STUDIO_URL}")
        print(f"  Error: {e}")
        print("")
        print("  Make sure:")
        print("    - LM Studio is open")
        print("    - A model is loaded (left sidebar)")
        print("    - Local Server is ON (green toggle, default port 1234)")
        return

    # Step 2: compress
    print("Step 2: Compressing context with trimtoken...")
    compressed, report = compress_for_openai_compat(
        messages=MESSAGES,
        token_budget=TOKEN_BUDGET,
        scorer=build_scorer(),
        query=QUERY,
    )
    print_report(report)

    # Step 3: chunk scores
    print("Step 3: Chunk scores (how trimtoken ranked each turn):")
    from trimtoken import ContextCompressor
    from trimtoken.segmenter import MessageSegmenter
    compressor = ContextCompressor(
        token_budget=TOKEN_BUDGET,
        scorer=build_scorer(),
        segmenter=MessageSegmenter(),
    )
    chunks = compressor.score_only(MESSAGES, query=QUERY)
    print_chunk_scores(chunks)

    # Step 4: show compressed messages
    print("Step 4: Compressed messages being sent to LM Studio:")
    print("  " + DIVIDER)
    for msg in compressed:
        role    = msg["role"].upper()
        content = msg["content"][:88].replace("\n", " ")
        suffix  = "..." if len(msg["content"]) > 88 else ""
        print(f"  [{role}] {content}{suffix}")
    print("")

    # Step 5: send to LM Studio
    print("Step 5: Sending to LM Studio...")
    print("  " + DIVIDER)
    try:
        response = client.chat.completions.create(
            model=loaded_model,
            messages=compressed,
            max_tokens=MAX_RESPONSE,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        usage  = response.usage

        print(f"  Model             : {loaded_model}")
        print(f"  Prompt tokens used: {usage.prompt_tokens}")
        print(f"  Response tokens   : {usage.completion_tokens}")
        print("")
        print("  Model Response")
        print("  " + DIVIDER)
        print(answer)
        print("  " + DIVIDER)

        saved = report.original_tokens - report.compressed_tokens
        print(f"\n  Done! Saved {saved:,} tokens "
              f"({1 - report.compression_ratio:.0%} reduction) "
              f"before sending to LM Studio.\n")

    except Exception as e:
        print(f"  LM Studio call failed: {e}")
        print("  Check that a model is loaded and the server is running.")


if __name__ == "__main__":
    main()
