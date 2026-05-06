from typing import Optional

import numpy as np

from core.contracts import IndexHandle, Retriever, TopKResult
from .modes import BRUTEFORCE_FULL, BRUTEFORCE_SUBSET


class BruteForceRetriever(Retriever):
    def __init__(self, supports_subset: bool = False):
        self.supports_subset = supports_subset
        self.item_ids: Optional[np.ndarray] = None
        self.normed_item_embeddings: Optional[np.ndarray] = None

    def build_index(
        self,
        item_ids: np.ndarray,
        item_embeddings: np.ndarray,
        mode_config: Optional[dict] = None,
    ) -> IndexHandle:
        self.item_ids = item_ids.astype(np.uint64)
        emb = item_embeddings.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self.normed_item_embeddings = emb / norms
        return IndexHandle(index_type=BRUTEFORCE_SUBSET if self.supports_subset else BRUTEFORCE_FULL)

    def top_k(
        self,
        user_embedding: np.ndarray,
        k: int,
        candidate_subset: Optional[np.ndarray] = None,
    ) -> TopKResult:
        if self.item_ids is None or self.normed_item_embeddings is None:
            raise RuntimeError("Retriever index is not built")
        if k <= 0:
            return TopKResult(np.array([], dtype=np.uint64), np.array([], dtype=np.float32))

        user = user_embedding.astype(np.float32)
        user_norm = np.linalg.norm(user)
        if user_norm == 0.0:
            user_norm = 1.0
        user = user / user_norm

        if candidate_subset is not None:
            subset = candidate_subset.astype(np.int64)
            scores = self.normed_item_embeddings[subset] @ user
            ids = self.item_ids[subset]
        else:
            scores = self.normed_item_embeddings @ user
            ids = self.item_ids

        k_eff = min(k, scores.shape[0])
        top_idx_unsorted = np.argpartition(scores, -k_eff)[-k_eff:]
        order = np.argsort(scores[top_idx_unsorted])[::-1]
        top_idx = top_idx_unsorted[order]
        return TopKResult(ids[top_idx], scores[top_idx].astype(np.float32))

    def supports_mode(self, mode: str) -> bool:
        if self.supports_subset:
            return mode in {BRUTEFORCE_FULL, BRUTEFORCE_SUBSET}
        return mode == BRUTEFORCE_FULL
