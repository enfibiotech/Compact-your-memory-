"""trimtoken CLI — compress, estimate, and score LLM context windows."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trimtoken",
        description="Surgical context compression for local LLMs.",
    )
    sub = parser.add_subparsers(dest="command")

    # compress
    c = sub.add_parser("compress", help="Compress messages to fit a token budget.")
    c.add_argument("input", nargs="?", help="Path to JSON file (or - for stdin)")
    c.add_argument("--model", help="Ollama model name (auto-reads context size)")
    c.add_argument("--budget", type=int, help="Explicit token budget")
    c.add_argument("--budget-ratio", type=float, default=0.85)
    c.add_argument("--out", default="-", help="Output path (default: stdout)")
    c.add_argument("--report", choices=["none", "inline", "table"], default="inline")
    c.add_argument("--query", help="Query to anchor relevance scoring")

    # estimate
    e = sub.add_parser("estimate", help="Dry-run: show what would be compressed.")
    e.add_argument("input", nargs="?")
    e.add_argument("--model", help="Ollama model name")
    e.add_argument("--budget", type=int)
    e.add_argument("--query")

    # score
    s = sub.add_parser("score", help="Score and inspect chunks without compressing.")
    s.add_argument("input", nargs="?")
    s.add_argument("--model", help="Ollama model name")
    s.add_argument("--budget", type=int)
    s.add_argument("--query")

    return parser


def _read_input(path: str | None) -> list[dict]:
    if path is None or path == "-":
        data = sys.stdin.read()
    else:
        with open(path) as f:
            data = f.read()
    return json.loads(data)


def _print_report_table(report) -> None:
    rows = [
        ("Original tokens", f"{report.original_tokens:,}"),
        ("Compressed tokens", f"{report.compressed_tokens:,}"),
        ("Compression ratio", f"{report.compression_ratio:.1%}"),
        ("Chunks dropped", str(report.chunks_dropped)),
        ("Chunks summarized", str(report.chunks_summarized)),
        ("Chunks kept", str(report.chunks_kept)),
        ("Budget utilization", f"{report.budget_utilization:.1%}"),
        ("Scorer", report.scorer_used),
        ("Strategy", report.strategy_used),
    ]
    width = max(len(k) for k, _ in rows) + 4
    print("\n┌" + "─" * (width + 22) + "┐")
    print(f"│  {'trimtoken compression report':<{width + 20}}│")
    print("├" + "─" * (width) + "┬" + "─" * 22 + "┤")
    for k, v in rows:
        print(f"│  {k:<{width - 2}}│  {v:<20}│")
    print("└" + "─" * width + "┴" + "─" * 22 + "┘\n")


async def _run(args: argparse.Namespace) -> None:
    messages = _read_input(getattr(args, "input", None))

    if args.command in ("compress", "estimate"):
        if args.model:
            from trimtoken.integrations.ollama import compress_for_ollama
            compressed, report = await compress_for_ollama(
                messages=messages,
                model=args.model,
                query=getattr(args, "query", None),
                budget_ratio=getattr(args, "budget_ratio", 0.85),
            )
        elif args.budget:
            from trimtoken import ContextCompressor
            compressor = ContextCompressor(token_budget=args.budget)
            result = compressor.compress(
                messages, query=getattr(args, "query", None)
            )
            compressed, report = result.messages, result.report
        else:
            print("Error: provide --model (Ollama) or --budget.", file=sys.stderr)
            sys.exit(1)

        if args.command == "estimate":
            _print_report_table(report)
            return

        out = json.dumps(compressed, indent=2)
        if args.out == "-":
            print(out)
        else:
            with open(args.out, "w") as f:
                f.write(out)

        report_mode = getattr(args, "report", "inline")
        if report_mode == "table":
            _print_report_table(report)
        elif report_mode == "inline":
            reduction = 1 - report.compression_ratio
            print(
                f"\n✓ {report.original_tokens:,} → {report.compressed_tokens:,} tokens  "
                f"({reduction:.0%} reduction)",
                file=sys.stderr,
            )

    elif args.command == "score":
        from trimtoken import ContextCompressor
        budget = args.budget or 999999
        compressor = ContextCompressor(token_budget=budget)
        chunks = compressor.score_only(messages, query=getattr(args, "query", None))
        print(f"\n{'Score':>6}  {'Tokens':>6}  {'Role':<12}  Content")
        print("─" * 72)
        for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
            preview = chunk.content[:60].replace("\n", " ")
            print(f"{chunk.score:>6.2f}  {chunk.token_count:>6}  {chunk.role.value:<12}  {preview}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
