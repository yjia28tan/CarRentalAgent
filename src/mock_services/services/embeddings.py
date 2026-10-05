"""Local sentence-transformers embeddings for the mock KB (all-MiniLM-L6-v2, 384-dim)."""
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from src.utils.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384


@lru_cache
def _model() -> "SentenceTransformer":
    # Imported lazily: torch is slow to import and not needed by most code paths/tests.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 vectors, shape (len(texts), 384). CPU-bound: call via a thread."""
    vectors = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
    return vectors.astype(np.float32)
