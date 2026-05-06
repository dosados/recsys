from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

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

from .centers_updater import update_centers
from .drift_handler import apply_interest_drift


class OnlineKMeansCore(RecSysCore):
    def __init__(self, embed_dim: int = 64, k_centers: int = 4, drift_alpha: float = 0.05, seed: int = 42):
        self.embed_dim = embed_dim
        self.k_centers = k_centers
        self.drift_alpha = drift_alpha
        self.rng = np.random.default_rng(seed)
        self.item_ids = np.array([], dtype=np.uint64)
        self.item_embeddings = np.array([], dtype=np.float32).reshape(0, embed_dim)
        self.item_id_to_idx: dict[int, int] = {}
        self.user_centers: dict[int, np.ndarray] = {}
        self.retriever = build_retriever("bruteforce_full")

    @staticmethod
    def _positive_mask(batch: dict[str, np.ndarray]) -> np.ndarray:
        mask = (
            batch["like"].astype(bool)
            | batch["share"].astype(bool)
            | batch["bookmark"].astype(bool)
            | batch["click_on_author"].astype(bool)
            | batch["open_comments"].astype(bool)
        )
        mask |= batch["timespent"].astype(np.int64) > 30
        return mask

    def _fit(self, dataset: DatasetPaths) -> FitSummary:
        if not dataset.item_embeddings_npz:
            raise ValueError("online_kmeans requires item_embeddings_npz")
        item_data = np.load(dataset.item_embeddings_npz)
        self.item_ids = item_data["item_id"].astype(np.uint64)
        self.item_embeddings = item_data["embedding"].astype(np.float32)
        self.item_id_to_idx = {int(i): idx for idx, i in enumerate(self.item_ids)}
        self.retriever.build_index(self.item_ids, self.item_embeddings, mode_config=None)

        interaction_paths = dataset.interaction_paths or [dataset.interactions_path]
        cols = ["user_id", "item_id", "timespent", "like", "share", "bookmark", "click_on_author", "open_comments"]
        for path in interaction_paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=262_144, columns=cols):
                payload = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in cols}
                pos = self._positive_mask(payload)
                if not np.any(pos):
                    continue
                users = payload["user_id"][pos]
                items = payload["item_id"][pos]
                for i in range(len(users)):
                    uid = int(users[i])
                    item_idx = self.item_id_to_idx.get(int(items[i]))
                    if item_idx is None:
                        continue
                    vec = self.item_embeddings[item_idx]
                    if uid not in self.user_centers:
                        init = self.rng.normal(0.0, 0.1, size=(self.k_centers, self.embed_dim)).astype(np.float32)
                        self.user_centers[uid] = init
                    self.user_centers[uid] = update_centers(self.user_centers[uid], vec, lr=0.15)
                    self.user_centers[uid] = apply_interest_drift(self.user_centers[uid], alpha=self.drift_alpha)
        return FitSummary(n_users=len(self.user_centers), n_items=len(self.item_ids))

    def fit_offline(self, dataset: DatasetPaths, config: Optional[dict] = None) -> FitSummary:
        self.user_centers = {}
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
            centers = self.user_centers.get(int(uid))
            if centers is None:
                results.append(RecommendationResult(int(uid), np.array([], dtype=np.uint64), np.array([], dtype=np.float32)))
                continue
            # Aggregate recommendations from all centers.
            agg_scores: dict[int, float] = {}
            for c in centers:
                topk = self.retriever.top_k(c, k=k, candidate_subset=subset)
                for item_id, score in zip(topk.item_ids.tolist(), topk.scores.tolist()):
                    agg_scores[item_id] = max(agg_scores.get(int(item_id), -1e9), float(score))
            sorted_items = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            rec_items = np.array([x[0] for x in sorted_items], dtype=np.uint64)
            rec_scores = np.array([x[1] for x in sorted_items], dtype=np.float32)
            results.append(RecommendationResult(int(uid), rec_items, rec_scores))
        return BatchRecommendationResult(results=results)

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "online_kmeans_model.npz",
            item_id=self.item_ids,
            item_embedding=self.item_embeddings,
            user_ids=np.array(list(self.user_centers.keys()), dtype=np.uint64),
            centers=np.stack(list(self.user_centers.values()), axis=0)
            if self.user_centers
            else np.zeros((0, self.k_centers, self.embed_dim), dtype=np.float32),
        )

    def load(self, output_dir: str) -> None:
        data = np.load(Path(output_dir) / "online_kmeans_model.npz")
        self.item_ids = data["item_id"].astype(np.uint64)
        self.item_embeddings = data["item_embedding"].astype(np.float32)
        self.item_id_to_idx = {int(i): idx for idx, i in enumerate(self.item_ids)}
        user_ids = data["user_ids"].astype(np.uint64)
        centers = data["centers"].astype(np.float32)
        self.user_centers = {int(uid): centers[i] for i, uid in enumerate(user_ids)}
        self.retriever.build_index(self.item_ids, self.item_embeddings, mode_config=None)

    def export_user_embeddings(self, output_path: str) -> EmbeddingArtifactRef:
        user_ids = np.array(list(self.user_centers.keys()), dtype=np.uint64)
        if user_ids.size > 0:
            emb = np.stack([self.user_centers[int(uid)].mean(axis=0) for uid in user_ids], axis=0).astype(np.float32)
        else:
            emb = np.zeros((0, self.embed_dim), dtype=np.float32)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, user_id=user_ids, embedding=emb)
        return EmbeddingArtifactRef(
            artifact_path=str(out),
            n_vectors=int(emb.shape[0]),
            dim=int(emb.shape[1]) if emb.ndim == 2 else 0,
            dtype=str(emb.dtype),
            metadata={"core": "online_kmeans", "k_centers": self.k_centers},
        )
