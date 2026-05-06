import logging
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq

from core.contracts import BatchRecommendationResult, DatasetPaths, FitSummary, RecommendationResult, RecSysCore
from core.contracts import EmbeddingArtifactRef, RetrievalRequest

logger = logging.getLogger(__name__)


class RandomCore(RecSysCore):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.item_ids = np.array([], dtype=np.uint64)

    def fit_offline(self, dataset: DatasetPaths, config: dict | None = None) -> FitSummary:
        items = pq.read_table(dataset.items_path, columns=["item_id"])
        self.item_ids = items.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        users = pq.read_table(dataset.users_path, columns=["user_id"])
        logger.info("Random baseline: n_items=%d", self.item_ids.size)
        return FitSummary(n_users=users.num_rows, n_items=self.item_ids.size)

    def update_incremental(self, dataset: DatasetPaths, config: dict | None = None) -> FitSummary:
        return self.fit_offline(dataset)

    def recommend(
        self,
        user_ids: Sequence[int],
        k: int,
        retrieval_request: RetrievalRequest | None = None,
    ) -> BatchRecommendationResult:
        if self.item_ids.size == 0 or k <= 0:
            return BatchRecommendationResult(results=[])
        k_eff = min(k, self.item_ids.size)

        results = []
        for user_id in user_ids:
            choice_idx = self.rng.choice(self.item_ids.size, size=k_eff, replace=False)
            rec_items = self.item_ids[choice_idx]
            rec_scores = np.ones(k_eff, dtype=np.float32) / float(k_eff)
            results.append(
                RecommendationResult(
                    user_id=int(user_id),
                    item_ids=rec_items.astype(np.uint64),
                    scores=rec_scores,
                )
            )
        return BatchRecommendationResult(results=results)

    def save(self, output_dir: str) -> None:
        # Stateless baseline for now.
        return

    def load(self, output_dir: str) -> None:
        # Stateless baseline for now.
        return

    def export_user_embeddings(self, output_path: str) -> EmbeddingArtifactRef:
        raise NotImplementedError("Random baseline does not produce user embeddings")

