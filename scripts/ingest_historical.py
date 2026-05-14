"""
scripts/ingest_historical.py
One-time script to embed all historical tickets and store them in ChromaDB.

Usage:
    python scripts/ingest_historical.py

What it does:
    1. Loads historical_tickets.json (30 past resolved tickets)
    2. Embeds each ticket's complaint using all-MiniLM-L6-v2
    3. Stores embeddings + metadata in a persistent ChromaDB collection
    4. Verifies the ingestion with a sample query
"""

import json
import os
import sys
import time

# Add project root to path so we can import our modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag.retriever import add_tickets, search_similar_tickets, get_collection_count, reset_collection


def main():
    print("=" * 60)
    print("  Maintenance Ticket Ingestion Pipeline")
    print("=" * 60)

    # ---- Step 1: Load historical tickets ----
    data_path = os.path.join(PROJECT_ROOT, "data", "historical_tickets.json")
    print(f"\n📂 Loading tickets from: {data_path}")

    with open(data_path, "r") as f:
        tickets = json.load(f)

    print(f"   Found {len(tickets)} historical tickets")

    # Show category/urgency distribution
    from collections import Counter
    categories = Counter(t["category"] for t in tickets)
    urgencies = Counter(t["urgency"] for t in tickets)

    print("\n   Category distribution:")
    for cat, count in sorted(categories.items()):
        print(f"     {cat:<25} {count}")

    print("\n   Urgency distribution:")
    for urg, count in sorted(urgencies.items()):
        print(f"     {urg:<25} {count}")

    # ---- Step 2: Reset and ingest ----
    print("\n🔄 Resetting ChromaDB collection...")
    reset_collection()

    print("🧠 Embedding and storing tickets (this may take a moment on first run)...")
    start = time.time()
    count = add_tickets(tickets)
    elapsed = time.time() - start

    print(f"   ✅ Ingested {count} tickets in {elapsed:.1f}s")
    print(f"   Collection size: {get_collection_count()}")

    # ---- Step 3: Verification queries ----
    print("\n" + "=" * 60)
    print("  Verification — Sample Queries")
    print("=" * 60)

    test_queries = [
        "My sink is leaking water everywhere",
        "The AC isn't working and it's really hot",
        "I see cockroaches in my apartment",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        results = search_similar_tickets(query, top_k=3)

        for i, r in enumerate(results, 1):
            print(f"   {i}. [{r['ticket_id']}] (similarity: {r['similarity_score']:.3f})")
            print(f"      Category: {r['category']} | Urgency: {r['urgency']}")
            print(f"      \"{r['complaint'][:80]}...\"")

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete! RAG pipeline is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
