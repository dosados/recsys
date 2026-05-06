import logging
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from core.contracts import DatasetPaths, RecSysCore, RetrievalRequest

from .holdout import build_positives_holdout, build_positives_holdout_from_paths
from .metrics import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .reporting import write_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    ndcg_at_k: float
    coverage_at_k: float
    n_eval_users: int


def retrieval_request_from_eval_dict(m: dict | None) -> RetrievalRequest:
    m = m or {}
    return RetrievalRequest(
        mode=m.get("mode", "bruteforce_full"),
        ann_enabled=m.get("ann_enabled", False),
        subset_size=m.get("subset_size"),
        random_seed=int(m.get("random_seed", 42)),
        index_path=m.get("index_path"),
        extra=m.get("extra") or {},
    )


class EvaluationRunner:
    def __init__(
        self,
        k: int = 20,
        positive_policy: str = "positives_only",
        metrics_output_dir: str | None = None,
        retrieval_request: RetrievalRequest | None = None,
        recommend_chunk_size: int = 10_000,
        max_eval_users: int | None = None,
    ):
        self.k = k
        self.positive_policy = positive_policy
        self.metrics_output_dir = metrics_output_dir
        self.retrieval_request = retrieval_request
        self.recommend_chunk_size = max(1, int(recommend_chunk_size))
        self.max_eval_users = max_eval_users

    def _build_holdout(self, interactions_path: str, interaction_paths: list[str] | None = None) -> dict[int, np.ndarray]:
        if self.positive_policy != "positives_only":
            raise ValueError(f"Unsupported positive policy: {self.positive_policy}")
        if interaction_paths:
            return build_positives_holdout_from_paths(interaction_paths)
        return build_positives_holdout(interactions_path)

    def _holdout_source_description(self, dataset: DatasetPaths) -> str:
        paths = dataset.interaction_paths if dataset.interaction_paths else [dataset.interactions_path]
        if len(paths) == 1:
            return paths[0]
        return f"{len(paths)} files: " + ", ".join(paths)

    def evaluate(self, core: RecSysCore, dataset: DatasetPaths, core_name: str = "unknown") -> EvalResult:
        logger.info(
            "Building holdout from %s (policy=%s)",
            self._holdout_source_description(dataset),
            self.positive_policy,
        )
        holdout = self._build_holdout(dataset.interactions_path, dataset.interaction_paths)
        if not holdout:
            logger.warning("Empty holdout; metrics are zero")
            return EvalResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        all_user_ids = sorted(holdout.keys())
        n_holdout = len(all_user_ids)
        eval_user_ids = all_user_ids
        if self.max_eval_users is not None:
            eval_user_ids = all_user_ids[: self.max_eval_users]
            if n_holdout > len(eval_user_ids):
                logger.info(
                    "max_eval_users=%d: using first %d users in sorted user_id order "
                    "(%d users in holdout; metrics are on this subset only)",
                    self.max_eval_users,
                    len(eval_user_ids),
                    n_holdout,
                )
        pos_lens = [len(holdout[u]) for u in eval_user_ids]
        logger.info(
            "Evaluating %d users @k=%d (mean positives per user=%.1f, max=%d)",
            len(eval_user_ids),
            self.k,
            float(np.mean(pos_lens)),
            int(np.max(pos_lens)),
        )

        req = self.retrieval_request or RetrievalRequest()
        n_users = len(eval_user_ids)
        precision_sum = 0.0
        recall_sum = 0.0
        map_sum = 0.0
        ndcg_sum = 0.0
        seen_items_for_coverage: set[int] = set()

        for start in range(0, n_users, self.recommend_chunk_size):
            chunk_uids = eval_user_ids[start : start + self.recommend_chunk_size]
            batch = core.recommend(chunk_uids, k=self.k, retrieval_request=req)
            for result in batch.results:
                true_items = holdout.get(result.user_id, np.array([], dtype=np.uint64))
                precision_sum += precision_at_k(result.item_ids, true_items, self.k)
                recall_sum += recall_at_k(result.item_ids, true_items, self.k)
                map_sum += map_at_k(result.item_ids, true_items, self.k)
                ndcg_sum += ndcg_at_k(result.item_ids, true_items, self.k)
                pred = result.item_ids
                if pred.size > 0:
                    take = pred[: self.k]
                    for x in take:
                        seen_items_for_coverage.add(int(x))

        items_table = pq.read_table(dataset.items_path, columns=["item_id"])
        total_items = items_table.num_rows
        if total_items <= 0 or not seen_items_for_coverage:
            coverage = 0.0
        else:
            coverage = float(len(seen_items_for_coverage) / total_items)

        p_at_k = float(precision_sum / n_users)
        r_at_k = float(recall_sum / n_users)
        map_k = float(map_sum / n_users)
        ndcg_k = float(ndcg_sum / n_users)
        logger.info(
            "Metrics: P@k=%.4f R@k=%.4f MAP@k=%.4f nDCG@k=%.4f coverage@k=%.4f",
            p_at_k,
            r_at_k,
            map_k,
            ndcg_k,
            coverage,
        )
        return EvalResult(
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            map_at_k=map_k,
            ndcg_at_k=ndcg_k,
            coverage_at_k=coverage,
            n_eval_users=n_users,
        )

    def save_result(self, core_name: str, result: EvalResult, config: dict, data_slice: str) -> str | None:
        if not self.metrics_output_dir:
            return None
        metrics = {
            "precision_at_k": result.precision_at_k,
            "recall_at_k": result.recall_at_k,
            "map_at_k": result.map_at_k,
            "ndcg_at_k": result.ndcg_at_k,
            "coverage_at_k": result.coverage_at_k,
            "n_eval_users": float(result.n_eval_users),
        }
        artifacts = write_metrics(
            core_name=core_name,
            metrics=metrics,
            output_dir=self.metrics_output_dir,
            config=config,
            data_slice=data_slice,
            k=self.k,
        )
        return artifacts.metrics_path
