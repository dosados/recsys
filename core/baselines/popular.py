import logging
from collections import Counter
from typing import List, Sequence, Set

import numpy as np
import pyarrow.parquet as pq

from core.contracts import BatchRecommendationResult, DatasetPaths, FitSummary, RecommendationResult, RecSysCore
from core.contracts import EmbeddingArtifactRef, RetrievalRequest

logger = logging.getLogger(__name__)


def _parquet_column_names(path: str) -> Set[str]:
    return {f.name for f in pq.ParquetFile(path).schema_arrow}


class PopularCore(RecSysCore):
    def __init__(self, min_positive_score: int = 2):
        self.min_positive_score = min_positive_score
        self.popular_items = np.array([], dtype=np.uint64)

    def _fit_from_item_rank(self, dataset: DatasetPaths) -> None:
        table = pq.read_table(dataset.items_path, columns=["item_id", "train_interactions_rank"])
        ids = table.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        ranks = table.column("train_interactions_rank").to_numpy(zero_copy_only=False).astype(np.int64)
        order = np.lexsort((ids, ranks))
        self.popular_items = ids[order]

    def _fit_from_scored_interactions(self, paths: List[str]) -> bool:
        counts: Counter = Counter()
        for path in paths:
            inter = pq.read_table(path, columns=["item_id", "score"])
            item_ids = inter.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)
            scores = inter.column("score").to_numpy(zero_copy_only=False).astype(np.int64)
            item_ids = item_ids[scores >= self.min_positive_score]
            if item_ids.size == 0:
                continue
            ui, cnt = np.unique(item_ids, return_counts=True)
            counts.update({int(u): int(n) for u, n in zip(ui, cnt)})
        if not counts:
            return False
        self.popular_items = np.array([k for k, _ in counts.most_common()], dtype=np.uint64)
        return True

    def _fit_from_impression_counts(self, paths: List[str]) -> bool:
        counts: Counter = Counter()
        for path in paths:
            inter = pq.read_table(path, columns=["item_id"])
            item_ids = inter.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)
            if item_ids.size == 0:
                continue
            ui, cnt = np.unique(item_ids, return_counts=True)
            counts.update({int(u): int(n) for u, n in zip(ui, cnt)})
        if not counts:
            return False
        self.popular_items = np.array([k for k, _ in counts.most_common()], dtype=np.uint64)
        return True

    def fit_offline(self, dataset: DatasetPaths, config: dict | None = None) -> FitSummary:
        paths = list(dataset.interaction_paths or [dataset.interactions_path])
        item_cols = _parquet_column_names(dataset.items_path)

        if "train_interactions_rank" in item_cols:
            logger.info("Popular: ordering items by train_interactions_rank")
            self._fit_from_item_rank(dataset)
        else:
            inter_cols = _parquet_column_names(paths[0])
            if "score" in inter_cols:
                if not self._fit_from_scored_interactions(paths):
                    items = pq.read_table(dataset.items_path, columns=["item_id"])
                    self.popular_items = items.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)
            else:
                if not self._fit_from_impression_counts(paths):
                    items = pq.read_table(dataset.items_path, columns=["item_id"])
                    self.popular_items = items.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)

        users = pq.read_table(dataset.users_path, columns=["user_id"])
        logger.info("Popular fit: n_items=%d", self.popular_items.size)
        return FitSummary(n_users=users.num_rows, n_items=self.popular_items.size)

    def update_incremental(self, dataset: DatasetPaths, config: dict | None = None) -> FitSummary:
        return self.fit_offline(dataset)

    def recommend(
        self,
        user_ids: Sequence[int],
        k: int,
        retrieval_request: RetrievalRequest | None = None,
    ) -> BatchRecommendationResult:
        if self.popular_items.size == 0 or k <= 0:
            return BatchRecommendationResult(results=[])
        k_eff = min(k, self.popular_items.size)
        top_items = self.popular_items[:k_eff]
        scores = np.linspace(1.0, 0.1, num=k_eff, dtype=np.float32)

        results = []
        for user_id in user_ids:
            results.append(
                RecommendationResult(
                    user_id=int(user_id),
                    item_ids=top_items.copy(),
                    scores=scores.copy(),
                )
            )
        return BatchRecommendationResult(results=results)

    def save(self, output_dir: str) -> None:
        return

    def load(self, output_dir: str) -> None:
        return

    def export_user_embeddings(self, output_path: str) -> EmbeddingArtifactRef:
        raise NotImplementedError("Popular baseline does not produce user embeddings")

