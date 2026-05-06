import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from core.contracts import (
    BatchRecommendationResult,
    DatasetPaths,
    EmbeddingArtifactRef,
    FitSummary,
    RecommendationResult,
    RecSysCore,
    RetrievalRequest,
    Retriever,
)
from data.contracts import load_dataset_tables

from .batch_processor import BatchProcessor
from .checkpoint import load_checkpoint, save_checkpoint
from .config import EMAConfig
from .item_embeddings import ItemEmbeddingLoader
from .memmap_manager import (
    capacity_from_memmap_file,
    create_user_memmap,
    grow_user_memmap,
    open_user_memmap,
)
from .retriever import SimpleTopKRetriever
from retrieval.subset_sampler import RandomSubsetSampler

logger = logging.getLogger(__name__)


def _estimate_interaction_batches(paths: Sequence[str], batch_size: int) -> int:
    rows = 0
    for p in paths:
        rows += pq.ParquetFile(p).metadata.num_rows
    return max(1, (int(rows) + batch_size - 1) // batch_size)


class EMACore(RecSysCore):
    def __init__(self, config: EMAConfig, retriever: Optional[Retriever] = None):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.user_emb_memmap_file = self.work_dir / "user_embeddings.npy"
        self.user_ids_file = self.work_dir / "user_ids.npy"
        self.checkpoint_file = self.work_dir / "checkpoint.json"
        self.model_export_file = self.work_dir / "ema_model.npz"

        self.item_loader: Optional[ItemEmbeddingLoader] = None
        self.batch_processor: Optional[BatchProcessor] = None
        self.user_emb_mm: Optional[np.memmap] = None
        self.user_id_to_idx: Dict[int, int] = {}
        self.n_users: int = 0
        self.current_capacity: int = 0
        self.weight_arr: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self._ema_model_npz: Any = None
        self.retriever = retriever or SimpleTopKRetriever()
        self.subset_sampler = RandomSubsetSampler(random_seed=42)

    def _close_ema_model_npz(self) -> None:
        if self._ema_model_npz is not None:
            self._ema_model_npz.close()
            self._ema_model_npz = None

    def _prepare_weight_array(self) -> None:
        max_score = max(self.config.score_weights.keys()) if self.config.score_weights else 0
        self.weight_arr = np.zeros(max_score + 1, dtype=np.float32)
        for k, v in self.config.score_weights.items():
            if 0 <= k <= max_score:
                self.weight_arr[k] = float(v)

    def _restore_state(self) -> tuple[dict, np.ndarray]:
        checkpoint = load_checkpoint(self.checkpoint_file)
        detected_capacity = capacity_from_memmap_file(self.user_emb_memmap_file, 4, self.config.embed_dim)
        detected_capacity = max(detected_capacity, int(checkpoint.get("capacity", 0)))

        if detected_capacity > 0:
            self.user_emb_mm = open_user_memmap(
                self.user_emb_memmap_file, detected_capacity, self.config.embed_dim
            )
            self.current_capacity = detected_capacity
        else:
            est_capacity = max(self.config.initial_capacity, int(checkpoint.get("capacity", 0)) or 0)
            capacity = min(est_capacity, self.config.max_capacity)
            self.user_emb_mm = create_user_memmap(self.user_emb_memmap_file, capacity, self.config.embed_dim)
            self.current_capacity = capacity

        if self.user_ids_file.exists() and checkpoint.get("n_users", 0) > 0:
            user_ids_arr = np.load(self.user_ids_file)
            self.n_users = int(min(len(user_ids_arr), int(checkpoint.get("n_users", 0))))
            self.user_id_to_idx = {int(user_ids_arr[i]): int(i) for i in range(self.n_users)}
        else:
            self.n_users = 0
            self.user_id_to_idx = {}
            user_ids_arr = np.zeros(self.current_capacity, dtype=np.uint64)
        return checkpoint, user_ids_arr

    def _grow_memmap_if_needed(self, required_total: int, user_ids_arr: np.ndarray) -> np.ndarray:
        if required_total <= self.current_capacity:
            return user_ids_arr

        new_cap = self.current_capacity
        while new_cap < required_total:
            new_cap = int(new_cap * 2)
            if new_cap > self.config.max_capacity:
                new_cap = self.config.max_capacity
                break
        if new_cap <= self.current_capacity:
            raise MemoryError("Cannot grow memmap further")

        self.user_emb_mm = grow_user_memmap(
            self.user_emb_memmap_file,
            self.user_emb_mm,
            self.current_capacity,
            new_cap,
            self.config.embed_dim,
        )
        if self.batch_processor is not None:
            self.batch_processor.user_emb_mm = self.user_emb_mm

        new_user_ids_arr = np.zeros(new_cap, dtype=np.uint64)
        new_user_ids_arr[: len(user_ids_arr)] = user_ids_arr
        self.current_capacity = new_cap
        return new_user_ids_arr

    def _process_interaction_files(self, interaction_files: Sequence[str]) -> None:
        user_ids_arr = np.load(self.user_ids_file) if self.user_ids_file.exists() else np.zeros(
            self.current_capacity, dtype=np.uint64
        )
        if len(user_ids_arr) < self.current_capacity:
            expanded = np.zeros(self.current_capacity, dtype=np.uint64)
            expanded[: len(user_ids_arr)] = user_ids_arr
            user_ids_arr = expanded

        total_batches = _estimate_interaction_batches(interaction_files, self.config.batch_size)
        logger.info(
            "Streaming %d parquet file(s), ~%d batch(es) of size %d",
            len(interaction_files),
            total_batches,
            self.config.batch_size,
        )

        global_batch_idx = 0
        pbar = tqdm(
            total=total_batches,
            desc="EMA batches",
            unit="batch",
            disable=not self.config.show_progress,
        )
        try:
            for interactions_file in interaction_files:
                logger.debug("Reading interactions: %s", interactions_file)
                parquet = pq.ParquetFile(interactions_file)
                for batch in parquet.iter_batches(batch_size=self.config.batch_size):
                    user_ids, item_ids, scores = self.batch_processor.extract_batch_data(batch)
                    unique_uids, batch_emb, has_valid = self.batch_processor.process_batch(
                        user_ids, item_ids, scores
                    )
                    if not has_valid:
                        global_batch_idx += 1
                        pbar.update(1)
                        continue

                    existing_indices = np.array(
                        [self.user_id_to_idx.get(int(uid), -1) for uid in unique_uids], dtype=np.int64
                    )
                    new_mask = existing_indices == -1
                    n_new = int(new_mask.sum())
                    if n_new > 0:
                        required_total = self.n_users + n_new
                        user_ids_arr = self._grow_memmap_if_needed(required_total, user_ids_arr)
                        new_indices = np.arange(self.n_users, self.n_users + n_new, dtype=np.int64)
                        existing_indices[new_mask] = new_indices
                        new_uids = unique_uids[new_mask]
                        user_ids_arr[new_indices] = new_uids
                        self.user_id_to_idx.update(
                            {int(uid): int(idx) for uid, idx in zip(new_uids, new_indices)}
                        )
                        self.n_users += n_new

                    self.batch_processor.update_user_embeddings_ema(unique_uids, batch_emb, existing_indices)
                    if global_batch_idx % self.config.gc_every == 0:
                        gc.collect()
                    global_batch_idx += 1
                    pbar.update(1)
                    if global_batch_idx % 32 == 0:
                        pbar.set_postfix(users=self.n_users, refresh=False)
        finally:
            pbar.close()

        logger.info("Finished interaction pass: %d users in index", self.n_users)
        if self.n_users > 0:
            np.save(self.user_ids_file, user_ids_arr[: self.n_users])

    def _fit_or_update(self, dataset: DatasetPaths) -> FitSummary:
        load_dataset_tables(dataset)
        inter_files = dataset.interaction_paths or [dataset.interactions_path]
        logger.info(
            "EMA fit: items=%s embeddings=%s interaction_files=%d",
            dataset.items_path,
            dataset.item_embeddings_npz or "(in items parquet)",
            len(inter_files),
        )

        item_emb_path = dataset.item_embeddings_npz or dataset.items_path
        self.item_loader = ItemEmbeddingLoader(item_emb_path, self.config.embed_dim)
        self.item_loader.load()
        self.item_ids = self.item_loader.snapshot.item_ids
        self.item_embeddings = self.item_loader.snapshot.item_embeddings

        self._prepare_weight_array()
        checkpoint, user_ids_arr = self._restore_state()

        self.batch_processor = BatchProcessor(
            embed_dim=self.config.embed_dim,
            weight_arr=self.weight_arr,
            ema_alpha=self.config.ema_alpha,
            item_loader=self.item_loader,
            user_emb_mm=self.user_emb_mm,
            user_id_to_idx=self.user_id_to_idx,
        )

        # keep persisted user id storage for incremental update use-cases
        if self.n_users > 0 and not self.user_ids_file.exists():
            np.save(self.user_ids_file, user_ids_arr[: self.n_users])

        self._process_interaction_files(inter_files)

        checkpoint["n_users"] = self.n_users
        checkpoint["capacity"] = self.current_capacity
        if self.user_ids_file.exists():
            full_user_ids = np.load(self.user_ids_file)
        else:
            full_user_ids = np.array([], dtype=np.uint64)
        save_checkpoint(checkpoint, self.checkpoint_file, self.user_ids_file, full_user_ids, self.n_users)
        self.user_emb_mm.flush()
        self._rebuild_retriever()
        logger.info("EMA fit done: n_users=%d n_items=%d", self.n_users, len(self.item_ids))
        return FitSummary(n_users=self.n_users, n_items=len(self.item_ids))

    def _rebuild_retriever(self) -> None:
        if self.item_ids is None or self.item_embeddings is None:
            return
        self.retriever.build_index(self.item_ids, self.item_embeddings, mode_config=None)

    def fit_offline(self, dataset: DatasetPaths, config: Optional[dict] = None) -> FitSummary:
        logger.info("EMA offline fit: resetting work_dir state under %s", self.work_dir)
        self._close_ema_model_npz()
        for path in [self.user_emb_memmap_file, self.user_ids_file, self.checkpoint_file]:
            if path.exists():
                path.unlink()
        self.user_id_to_idx = {}
        self.n_users = 0
        self.current_capacity = 0
        return self._fit_or_update(dataset)

    def update_incremental(self, dataset: DatasetPaths, config: Optional[dict] = None) -> FitSummary:
        return self._fit_or_update(dataset)

    def recommend(
        self,
        user_ids: Sequence[int],
        k: int,
        retrieval_request: Optional[RetrievalRequest] = None,
    ) -> BatchRecommendationResult:
        if self.user_emb_mm is None:
            raise RuntimeError("Core is not fitted yet")
        self._rebuild_retriever()

        results: list[RecommendationResult] = []
        n = len(user_ids)
        use_pbar = self.config.show_progress and n > 100
        uid_iter = tqdm(user_ids, desc="Recommend", unit="user", disable=not use_pbar)
        subset = None
        if (
            retrieval_request is not None
            and retrieval_request.mode in {"bruteforce_subset", "ann_subset"}
            and self.item_ids is not None
        ):
            self.subset_sampler = RandomSubsetSampler(random_seed=retrieval_request.random_seed)
            subset = self.subset_sampler.sample(len(self.item_ids), retrieval_request.subset_size)
        for user_id in uid_iter:
            idx = self.user_id_to_idx.get(int(user_id))
            if idx is None:
                results.append(
                    RecommendationResult(
                        user_id=int(user_id),
                        item_ids=np.array([], dtype=np.uint64),
                        scores=np.array([], dtype=np.float32),
                    )
                )
                continue
            emb = np.asarray(self.user_emb_mm[idx], dtype=np.float32)
            topk = self.retriever.top_k(emb, k=k, candidate_subset=subset)
            results.append(
                RecommendationResult(
                    user_id=int(user_id),
                    item_ids=topk.item_ids,
                    scores=topk.scores,
                )
            )
        return BatchRecommendationResult(results=results)

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.user_emb_mm is None:
            raise RuntimeError("Nothing to save; fit core first")
        user_ids = np.array(sorted(self.user_id_to_idx.keys()), dtype=np.uint64)
        idx = np.array([self.user_id_to_idx[int(uid)] for uid in user_ids], dtype=np.int64)
        emb = np.asarray(self.user_emb_mm[idx], dtype=np.float32)
        if self.item_ids is None or self.item_embeddings is None:
            raise RuntimeError("Item embeddings are not available; fit or update before save")
        np.savez_compressed(
            out / "ema_model.npz",
            user_id=user_ids,
            embedding=emb,
            item_id=self.item_ids.astype(np.uint64),
            item_embedding=self.item_embeddings.astype(np.float32),
        )

    def load(self, output_dir: str) -> None:
        model_path = Path(output_dir) / "ema_model.npz"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self._close_ema_model_npz()
        data = np.load(model_path, mmap_mode="r", allow_pickle=False)
        self._ema_model_npz = data
        user_ids = np.asarray(data["user_id"], dtype=np.uint64)
        embeddings = data["embedding"]
        self.item_ids = np.asarray(data["item_id"], dtype=np.uint64)
        item_emb = data["item_embedding"]
        if item_emb.dtype == np.float32:
            self.item_embeddings = item_emb
        else:
            self.item_embeddings = np.asarray(item_emb, dtype=np.float32)
        capacity = max(len(user_ids), self.config.initial_capacity)
        self.user_emb_mm = create_user_memmap(self.user_emb_memmap_file, capacity, self.config.embed_dim)
        self.user_emb_mm[: len(user_ids)] = np.asarray(embeddings, dtype=np.float32)
        self.user_emb_mm.flush()
        self.n_users = len(user_ids)
        self.current_capacity = capacity
        self.user_id_to_idx = {int(uid): i for i, uid in enumerate(user_ids)}
        np.save(self.user_ids_file, user_ids)
        gc.collect()
        self._rebuild_retriever()

    def export_user_embeddings(self, output_path: str) -> EmbeddingArtifactRef:
        if self.user_emb_mm is None:
            raise RuntimeError("Core is not fitted yet")
        user_ids = np.array(sorted(self.user_id_to_idx.keys()), dtype=np.uint64)
        idx = np.array([self.user_id_to_idx[int(uid)] for uid in user_ids], dtype=np.int64)
        emb = np.asarray(self.user_emb_mm[idx], dtype=np.float32)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, user_id=user_ids, embedding=emb)
        return EmbeddingArtifactRef(
            artifact_path=str(out),
            n_vectors=int(emb.shape[0]),
            dim=int(emb.shape[1]) if emb.ndim == 2 else 0,
            dtype=str(emb.dtype),
            metadata={"core": "ema"},
        )

