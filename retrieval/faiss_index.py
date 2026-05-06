from typing import Optional

import numpy as np

from core.contracts import IndexHandle, Retriever, TopKResult
from .modes import ANN_FULL, ANN_SUBSET

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class FaissRetriever(Retriever):
    def __init__(self, supports_subset: bool = False):
        self.supports_subset = supports_subset
        self.item_ids: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.index = None

    def build_index(
        self,
        item_ids: np.ndarray,
        item_embeddings: np.ndarray,
        mode_config: Optional[dict] = None,
    ) -> IndexHandle:
        self.item_ids = item_ids.astype(np.uint64)
        self.item_embeddings = item_embeddings.astype(np.float32)
        if faiss is None:
            self.index = None
            return IndexHandle(index_type="ann_fallback_numpy", metadata={"faiss_available": False})

        dim = self.item_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        normalized = self._normalize(self.item_embeddings)
        index.add(normalized)
        self.index = index
        return IndexHandle(index_type=ANN_SUBSET if self.supports_subset else ANN_FULL)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def top_k(
        self,
        user_embedding: np.ndarray,
        k: int,
        candidate_subset: Optional[np.ndarray] = None,
    ) -> TopKResult:
        if self.item_ids is None or self.item_embeddings is None:
            raise RuntimeError("Retriever index is not built")
        if k <= 0:
            return TopKResult(np.array([], dtype=np.uint64), np.array([], dtype=np.float32))

        user = self._normalize(user_embedding.reshape(1, -1).astype(np.float32))
        if candidate_subset is not None:
            subset = candidate_subset.astype(np.int64)
            subset_emb = self._normalize(self.item_embeddings[subset])
            scores = subset_emb @ user[0]
            k_eff = min(k, scores.shape[0])
            idx_unsorted = np.argpartition(scores, -k_eff)[-k_eff:]
            order = np.argsort(scores[idx_unsorted])[::-1]
            idx = idx_unsorted[order]
            return TopKResult(self.item_ids[subset[idx]], scores[idx].astype(np.float32))

        if self.index is None:
            scores = self._normalize(self.item_embeddings) @ user[0]
            k_eff = min(k, scores.shape[0])
            idx_unsorted = np.argpartition(scores, -k_eff)[-k_eff:]
            order = np.argsort(scores[idx_unsorted])[::-1]
            idx = idx_unsorted[order]
            return TopKResult(self.item_ids[idx], scores[idx].astype(np.float32))

        distances, indices = self.index.search(user, k)
        idx = indices[0]
        dist = distances[0]
        return TopKResult(self.item_ids[idx], dist.astype(np.float32))

    def supports_mode(self, mode: str) -> bool:
        if self.supports_subset:
            return mode in {ANN_FULL, ANN_SUBSET}
        return mode == ANN_FULL
