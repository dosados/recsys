from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.contracts import (
    BatchRecommendationResult,
    DatasetPaths,
    EmbeddingArtifactRef,
    FitSummary,
    RecommendationResult,
    RecSysCore,
    RetrievalRequest,
)
from retrieval.factory import build_retriever
from retrieval.subset_sampler import RandomSubsetSampler

from .item_tower import build_item_vectors
from .trainer import train_bpr_stub
from .user_tower import build_user_vectors


class TwoTowerCore(RecSysCore):
    def __init__(self, embed_dim: int = 64, epochs: int = 2, scoring: str = "cosine"):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.scoring = scoring
        self.user_ids = np.array([], dtype=np.uint64)
        self.user_embeddings = np.zeros((0, embed_dim), dtype=np.float32)
        self.user_id_to_idx: dict[int, int] = {}
        self.item_ids = np.array([], dtype=np.uint64)
        self.item_embeddings = np.zeros((0, embed_dim), dtype=np.float32)
        self.retriever = build_retriever("bruteforce_full")

    def _fit(self, dataset: DatasetPaths) -> FitSummary:
        if not dataset.item_embeddings_npz:
            raise ValueError("two_tower requires item_embeddings_npz")
        self.user_ids, user_vec = build_user_vectors(dataset.users_path, self.embed_dim)
        self.item_ids, item_vec = build_item_vectors(dataset.items_path, dataset.item_embeddings_npz, self.embed_dim)
        self.user_embeddings, self.item_embeddings = train_bpr_stub(user_vec, item_vec, epochs=self.epochs)
        self.user_id_to_idx = {int(uid): i for i, uid in enumerate(self.user_ids)}
        self.retriever.build_index(self.item_ids, self.item_embeddings, mode_config=None)
        return FitSummary(n_users=len(self.user_ids), n_items=len(self.item_ids))

    def fit_offline(self, dataset: DatasetPaths, config: Optional[dict] = None) -> FitSummary:
        return self._fit(dataset)

    def update_incremental(self, dataset: DatasetPaths, config: Optional[dict] = None) -> FitSummary:
        return self._fit(dataset)

    def recommend(
        self,
        user_ids: Sequence[int],
        k: int,
        retrieval_request: Optional[RetrievalRequest] = None,
    ) -> BatchRecommendationResult:
        subset = None
        if (
            retrieval_request is not None
            and retrieval_request.mode in {"bruteforce_subset", "ann_subset"}
            and self.item_ids.size > 0
        ):
            subset = RandomSubsetSampler(retrieval_request.random_seed).sample(
                self.item_ids.size, retrieval_request.subset_size
            )

        results: list[RecommendationResult] = []
        for uid in user_ids:
            idx = self.user_id_to_idx.get(int(uid))
            if idx is None:
                results.append(RecommendationResult(int(uid), np.array([], dtype=np.uint64), np.array([], dtype=np.float32)))
                continue
            topk = self.retriever.top_k(self.user_embeddings[idx], k=k, candidate_subset=subset)
            results.append(RecommendationResult(int(uid), topk.item_ids, topk.scores))
        return BatchRecommendationResult(results=results)

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "two_tower_model.npz",
            user_id=self.user_ids,
            user_embedding=self.user_embeddings,
            item_id=self.item_ids,
            item_embedding=self.item_embeddings,
        )

    def load(self, output_dir: str) -> None:
        data = np.load(Path(output_dir) / "two_tower_model.npz")
        self.user_ids = data["user_id"].astype(np.uint64)
        self.user_embeddings = data["user_embedding"].astype(np.float32)
        self.item_ids = data["item_id"].astype(np.uint64)
        self.item_embeddings = data["item_embedding"].astype(np.float32)
        self.user_id_to_idx = {int(uid): i for i, uid in enumerate(self.user_ids)}
        self.retriever.build_index(self.item_ids, self.item_embeddings, mode_config=None)

    def export_user_embeddings(self, output_path: str) -> EmbeddingArtifactRef:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, user_id=self.user_ids, embedding=self.user_embeddings)
        return EmbeddingArtifactRef(
            artifact_path=str(out),
            n_vectors=int(self.user_embeddings.shape[0]),
            dim=int(self.user_embeddings.shape[1]) if self.user_embeddings.ndim == 2 else 0,
            dtype=str(self.user_embeddings.dtype),
            metadata={"core": "two_tower", "scoring": self.scoring},
        )
