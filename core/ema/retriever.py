import numpy as np
from typing import Optional

from core.contracts import IndexHandle, Retriever, TopKResult


class SimpleTopKRetriever(Retriever):
    """Brute-force retriever kept behind interface for ANN swap later."""

    def __init__(self) -> None:
        self.item_ids: Optional[np.ndarray] = None
        self.normed_item_embeddings: Optional[np.ndarray] = None

    def build_index(
        self,
        item_ids: np.ndarray,
        item_embeddings: np.ndarray,
        mode_config: Optional[dict] = None,
    ) -> IndexHandle:
        self.item_ids = item_ids.astype(np.uint64)
        emb = np.asarray(item_embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self.normed_item_embeddings = emb / norms
        return IndexHandle(index_type="bruteforce_full")

    def top_k(
        self,
        user_embedding: np.ndarray,
        k: int,
        candidate_subset: Optional[np.ndarray] = None,
    ) -> TopKResult:
        if self.item_ids is None or self.normed_item_embeddings is None:
            raise RuntimeError("Retriever index is not built")
        if k <= 0:
            return TopKResult(
                item_ids=np.array([], dtype=np.uint64),
                scores=np.array([], dtype=np.float32),
            )

        user_embedding = user_embedding.astype(np.float32)
        user_norm = np.linalg.norm(user_embedding)
        if user_norm == 0.0:
            user_norm = 1.0
        user_vec = user_embedding / user_norm

        if candidate_subset is not None and candidate_subset.size > 0:
            subset = candidate_subset.astype(np.int64)
            subset_scores = self.normed_item_embeddings[subset] @ user_vec
            k_eff = min(k, subset_scores.shape[0])
            top_idx_unsorted = np.argpartition(subset_scores, -k_eff)[-k_eff:]
            order = np.argsort(subset_scores[top_idx_unsorted])[::-1]
            top_local_idx = top_idx_unsorted[order]
            top_global_idx = subset[top_local_idx]
            return TopKResult(
                item_ids=self.item_ids[top_global_idx],
                scores=subset_scores[top_local_idx].astype(np.float32),
            )

        scores = self.normed_item_embeddings @ user_vec
        k_eff = min(k, scores.shape[0])
        top_idx_unsorted = np.argpartition(scores, -k_eff)[-k_eff:]
        order = np.argsort(scores[top_idx_unsorted])[::-1]
        top_idx = top_idx_unsorted[order]
        return TopKResult(item_ids=self.item_ids[top_idx], scores=scores[top_idx].astype(np.float32))

    def supports_mode(self, mode: str) -> bool:
        return mode in {"bruteforce_full", "bruteforce_subset"}
