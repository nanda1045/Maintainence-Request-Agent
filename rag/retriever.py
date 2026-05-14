"""
rag/retriever.py
ChromaDB search and retrieval for similar maintenance tickets.

Manages a persistent ChromaDB collection and provides:
  - Collection initialization
  - Adding tickets (with embeddings + metadata)
  - Querying for the top-k most similar tickets given a complaint
"""

import os
from typing import Optional

import chromadb

from rag.embedder import embed_text, embed_batch

# ---------------------------------------------------------------------------
# ChromaDB configuration
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store"
)
COLLECTION_NAME = "maintenance_tickets"

_client = None
_collection = None


def get_client() -> chromadb.PersistentClient:
    """Get or create the persistent ChromaDB client."""
    global _client
    if _client is None:
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the maintenance_tickets collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
    return _collection


def add_tickets(tickets: list[dict]) -> int:
    """
    Add a batch of historical tickets to the ChromaDB collection.

    Each ticket should have at minimum:
      - ticket_id: unique identifier
      - complaint: the text to embed
      - category, urgency, vendor, resolution: stored as metadata

    Args:
        tickets: List of ticket dictionaries.

    Returns:
        Number of tickets added.
    """
    collection = get_collection()

    # Prepare data for ChromaDB
    ids = [t["ticket_id"] for t in tickets]
    documents = [t["complaint"] for t in tickets]
    metadatas = [
        {
            "ticket_id": t["ticket_id"],
            "unit": t.get("unit", ""),
            "category": t.get("category", ""),
            "urgency": t.get("urgency", ""),
            "vendor": t.get("vendor", ""),
            "resolution": t.get("resolution", ""),
            "sla_hours": t.get("sla_hours", 0),
            "resolved_date": t.get("resolved_date", ""),
        }
        for t in tickets
    ]

    # Embed all complaints in one batch
    embeddings = embed_batch(documents)

    # Upsert into ChromaDB (idempotent — safe to re-run)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)


def search_similar_tickets(query: str, top_k: int = 3) -> list[dict]:
    """
    Search for the most similar historical tickets given a new complaint.

    Args:
        query: The raw complaint text from a resident.
        top_k: Number of similar tickets to return (default 3).

    Returns:
        A list of dictionaries, each containing:
          - ticket_id
          - complaint (original complaint text)
          - category, urgency, vendor, resolution, sla_hours
          - similarity_score (cosine similarity, 0-1 where 1 = identical)
    """
    collection = get_collection()

    # Check if collection has any data
    if collection.count() == 0:
        return []

    # Embed the query
    query_embedding = embed_text(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    similar_tickets = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        # ChromaDB returns cosine distance; convert to similarity
        # cosine_similarity = 1 - cosine_distance
        distance = results["distances"][0][i]
        similarity = 1 - distance

        similar_tickets.append(
            {
                "ticket_id": results["ids"][0][i],
                "complaint": results["documents"][0][i],
                "category": metadata.get("category", ""),
                "urgency": metadata.get("urgency", ""),
                "vendor": metadata.get("vendor", ""),
                "resolution": metadata.get("resolution", ""),
                "sla_hours": metadata.get("sla_hours", 0),
                "resolved_date": metadata.get("resolved_date", ""),
                "similarity_score": round(similarity, 4),
            }
        )

    return similar_tickets


def get_collection_count() -> int:
    """Return the number of tickets currently in the vector store."""
    return get_collection().count()


def reset_collection() -> None:
    """Delete and recreate the collection (useful for re-ingestion)."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    get_collection()  # recreate
