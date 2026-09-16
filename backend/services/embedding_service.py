from __future__ import annotations

import threading
from typing import Any

import numpy as np


class EmbeddingService:
    """Lazily loads a local sentence-transformers model.

    Two things matter for latency here:

    * The load is guarded by a lock, so the startup warmup thread and the first
      concurrent search can never build the model twice (double the ~30 s CPU
      cost and double the RAM).
    * The cached copy is loaded with ``local_files_only=True`` first, so the
      Hugging Face Hub is never contacted on a normal start.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self._load_lock = threading.Lock()

    def _build_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        try:
            # Fast path: load the cached weights without any network round-trip.
            return SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            # Not cached yet on this machine — allow the one-time download.
            return SentenceTransformer(self.model_name)

    def _get_model(self) -> Any:
        if self.model is not None:
            return self.model
        with self._load_lock:
            if self.model is None:
                self.model = self._build_model()
        return self.model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

