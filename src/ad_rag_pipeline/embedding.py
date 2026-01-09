from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def embed_texts(
    texts: list[str],
    model_id: str,
    batch_size: int,
    device: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using SentenceTransformers.

    Args:
        texts: List of strings to embed.
        model_id: Hugging Face model ID (e.g. 'BAAI/bge-base-en-v1.5').
        batch_size: Batch size for inference.
        device: Device to use ('cpu', 'cuda', 'mps').
        normalize: Whether to normalize embeddings to unit length (default: True).

    Returns:
        np.ndarray: Matrix of shape (N, d) with float32 embeddings.

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("The 'texts' list is empty. Cannot generate embeddings.")

    model = SentenceTransformer(model_id, device=device)

    # SentenceTransformers handles batching internally
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32)
