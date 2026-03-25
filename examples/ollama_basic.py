"""
examples/ollama_basic.py
------------------------
Basic trimtoken usage with Ollama.
Run with: python examples/ollama_basic.py
Requires: pip install "trimtoken[ollama]" and Ollama running locally.
"""

import asyncio

# Simulate a long conversation history
LONG_HISTORY = [
    {"role": "system", "content": "You are a helpful customer support assistant for Acme Corp."},
    {"role": "user", "content": "Hi, I placed an order last week. Order #AC-7823."},
    {"role": "assistant", "content": (
        "Hello! I can see order #AC-7823 placed on May 15th. It contains 3 items totalling"
        " $247.50. What can I help you with?"
    )},
    {"role": "user", "content": "One of the items arrived damaged. The blue ceramic mug."},
    {"role": "assistant", "content": (
        "I'm sorry to hear that. I'll arrange a replacement for the blue ceramic mug at no"
        " charge. Can you confirm your shipping address?"
    )},
    {"role": "user", "content": "Yes, it's 42 Maple Street, Springfield, IL 62701."},
    {"role": "assistant", "content": (
        "Perfect. I've logged the replacement for the blue ceramic mug to 42 Maple Street,"
        " Springfield, IL 62701. It should arrive within 3-5 business days."
    )},
    {"role": "user", "content": "Also, I wanted to ask about your loyalty programme."},
    {"role": "assistant", "content": (
        "Our AcmeRewards programme gives you 1 point per dollar spent. At 500 points you get"
        " a $10 voucher. You currently have 312 points from your purchase history."
    )},
    {"role": "user", "content": "Great. And do you offer bulk discounts?"},
    {"role": "assistant", "content": (
        "Yes! Orders over $200 get 5% off, over $500 get 10% off, and over $1000 get 15%"
        " off. These are applied automatically at checkout."
    )},
    {"role": "user", "content": "What's your return policy?"},
    {"role": "assistant", "content": (
        "We offer a 30-day no-questions-asked return policy. Items must be in original"
        " packaging. Damaged or defective items like yours are covered separately under"
        " our quality guarantee."
    )},
    {"role": "user", "content": (
        "Can I return the other items from order AC-7823 if I change my mind?"
    )},
    {"role": "assistant", "content": (
        "Yes, the other items in order #AC-7823 are eligible for return until June 14th."
        " Just use the pre-paid label in your order confirmation email."
    )},
    {"role": "user", "content": (
        "Perfect. What was the delivery date again for my replacement mug?"
    )},
]

QUERY = "What was the delivery date for the replacement mug?"


async def main():
    try:
        from trimtoken.integrations.ollama import compress_for_ollama
    except ImportError:
        print("Install Ollama integration: pip install 'trimtoken[ollama]'")
        return

    MODEL = "llama3.2:3b"

    print(f"Original messages: {len(LONG_HISTORY)}")
    print(f"Query: {QUERY}\n")

    messages, report = await compress_for_ollama(
        messages=LONG_HISTORY,
        model=MODEL,
        query=QUERY,
        budget_ratio=0.85,
    )

    print("── Compression Report ─────────────────────────")
    print(f"  Original tokens  : {report.original_tokens:,}")
    print(f"  Compressed tokens: {report.compressed_tokens:,}")
    print(f"  Reduction        : {1 - report.compression_ratio:.0%}")
    print(f"  Chunks dropped   : {report.chunks_dropped}")
    print(f"  Chunks kept      : {report.chunks_kept}")
    print(f"  Budget used      : {report.budget_utilization:.1%}")
    print("───────────────────────────────────────────────\n")

    print("Compressed messages:")
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"][:80] + ("..." if len(msg["content"]) > 80 else "")
        print(f"  [{role}] {content}")

    # Uncomment to actually call Ollama:
    # import ollama
    # response = ollama.chat(model=MODEL, messages=messages)
    # print("\nModel response:", response["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
