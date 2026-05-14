"""
rag/embedder.py
Embedding logic using sentence-transformers (all-MiniLM-L6-v2).

Provides a singleton embedding model that can:
  - Embed a single text string
  - Embed a batch of text strings
  - Return the embedding dimension for ChromaDB config
"""

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformer model (singleton).
    The model is ~80 MB and loads in ~1-2 seconds on first call.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string into a dense vector.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector (384 dimensions).
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of text strings into dense vectors.

    Args:
        texts: List of input texts to embed.

    Returns:
        A list of embedding vectors, each with 384 dimensions.
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors (384 for MiniLM-L6-v2)."""
    model = get_model()
    return model.get_sentence_embedding_dimension()
