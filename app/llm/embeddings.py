"""Local sentence embeddings (fastembed, BAAI/bge-small-en-v1.5, CPU).

Local so ingestion needs only the one LLM key, vectors are deterministic, and
nothing leaves the container. The Docker image bakes the model in
(scripts/prefetch_models.py, FASTEMBED_CACHE_PATH), so startup never downloads it.
"""

import os
from collections.abc import Sequence

import numpy as np

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
# bge v1.5 retrieval: queries (not passages) get this instruction prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length float32 vectors, one row per text (cosine = dot product)."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        vectors = np.asarray(list(self._load().embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def embed_query(self, query: str) -> np.ndarray:
        """One unit-length vector for a search query."""
        return self.embed([QUERY_PREFIX + query])[0]
