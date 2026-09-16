from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from services.embedding_service import EmbeddingService
from services.text_cleaning import clean_text, tokenize

# Weight of the semantic (FAISS cosine) score vs the keyword-overlap score
# in the final hybrid ranking.
SEMANTIC_WEIGHT = 0.72
KEYWORD_WEIGHT = 0.28


class RetrievalService:
    def __init__(self, data_path: str | os.PathLike[str], index_path: str | os.PathLike[str], metadata_path: str | os.PathLike[str]) -> None:
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_service = EmbeddingService()
        self.index: faiss.Index | None = None
        self.metadata: pd.DataFrame | None = None
        # Guards the lazy load so the startup warmup thread and the first
        # concurrent search cannot read the index/metadata halfway through.
        self._load_lock = threading.Lock()

    @staticmethod
    def _document_text(record: Any) -> str:
        """Richer document text used for embedding / keyword scoring.

        Indexing the cause together with the actions and the solution makes
        the vectors capture the full escalation context, not just the
        (often typo-laden) problem sentence.
        """
        parts = [
            str(record.get("cause", "")),
            str(record.get("effort_taken", "")),
            str(record.get("solution", "")),
        ]
        return " ".join(part for part in parts if part and part.lower() != "nan")

    def build_index(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            raise ValueError("No records available to index")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        documents = [self._document_text(row) for _, row in dataframe.iterrows()]
        embeddings = self.embedding_service.embed(documents)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.metadata = dataframe.reset_index(drop=True)
        faiss.write_index(index, str(self.index_path))
        self.metadata.to_pickle(self.metadata_path)

    def load_index(self) -> None:
        with self._load_lock:
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = pd.read_pickle(self.metadata_path)

    def search(self, cause_query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Hybrid semantic + keyword search.

        1. FAISS inner-product search over normalized embeddings fetches a
           candidate pool (semantic recall).
        2. A keyword overlap score re-ranks the pool so exact domain terms
           (material names, KP numbers, departments) boost the right cases.
        3. The reported similarity is the honest combined score — no
           fabricated numbers.
        """
        if not self.index or self.metadata is None:
            self.load_index()
        query_clean = clean_text(cause_query)
        query_embedding = self.embedding_service.embed([query_clean])

        ntotal = int(self.index.ntotal)  # type: ignore[union-attr]
        if ntotal == 0:
            return []
        fetch_n = min(max(top_k * 5, 25), ntotal)
        scores, indices = self.index.search(query_embedding, fetch_n)  # type: ignore[union-attr]

        query_tokens = tokenize(query_clean)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = self.metadata.iloc[int(idx)]
            doc_tokens = tokenize(self._document_text(record))
            if query_tokens and doc_tokens:
                keyword_score = len(query_tokens & doc_tokens) / min(len(query_tokens), len(doc_tokens))
            else:
                keyword_score = 0.0
            combined = SEMANTIC_WEIGHT * float(score) + KEYWORD_WEIGHT * keyword_score
            results.append(
                {
                    "cause": self._field(record, "cause"),
                    "prevention": self._field(record, "prevention"),
                    "effort_taken": self._field(record, "effort_taken"),
                    "status": self._field(record, "status"),
                    "escalation_id": self._field(record, "escalation_id"),
                    "timestamp": self._field(record, "timestamp"),
                    "department": self._field(record, "department"),
                    "solution": self._field(record, "solution"),
                    "kp_no": self._field(record, "kp_no"),
                    "customer_name": self._field(record, "customer_name"),
                    "material_name": self._field(record, "material_name"),
                    "raised_by": self._field(record, "raised_by"),
                    "level": self._field(record, "level"),
                    "similarity": round(combined, 4),
                }
            )
        results.sort(key=lambda item: item["similarity"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _field(record: pd.Series, name: str) -> str:
        value = record.get(name, "")
        if value is None:
            return ""
        text = str(value)
        return "" if text.lower() == "nan" else text
