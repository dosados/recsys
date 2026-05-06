from typing import Dict, Tuple

import numpy as np


class BatchProcessor:
    def __init__(
        self,
        embed_dim: int,
        weight_arr: np.ndarray,
        ema_alpha: float,
        item_loader,
        user_emb_mm: np.memmap,
        user_id_to_idx: Dict[int, int],
    ):
        self.embed_dim = embed_dim
        self.weight_arr = weight_arr
        self.ema_alpha = ema_alpha
        self.item_loader = item_loader
        self.user_emb_mm = user_emb_mm
        self.user_id_to_idx = user_id_to_idx

    @staticmethod
    def _vk_feedback_to_score_int(batch) -> np.ndarray:
        """Map VK-LSVD feedback columns to integer scores for score_weights (0..3)."""
        ts = batch.column("timespent").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        s = np.clip(ts // 40, 0, 2).astype(np.int64)
        share = np.asarray(batch.column("share").to_numpy(zero_copy_only=False), dtype=np.bool_)
        bookmark = np.asarray(batch.column("bookmark").to_numpy(zero_copy_only=False), dtype=np.bool_)
        click_author = np.asarray(
            batch.column("click_on_author").to_numpy(zero_copy_only=False), dtype=np.bool_
        )
        open_comments = np.asarray(
            batch.column("open_comments").to_numpy(zero_copy_only=False), dtype=np.bool_
        )
        engage = share | bookmark | click_author | open_comments
        s = np.where(engage, np.maximum(s, np.int64(1)), s)
        like = np.asarray(batch.column("like").to_numpy(zero_copy_only=False), dtype=np.bool_)
        s = np.where(like, np.maximum(s, np.int64(2)), s)
        dislike = np.asarray(batch.column("dislike").to_numpy(zero_copy_only=False), dtype=np.bool_)
        s = np.where(dislike, np.int64(0), s)
        return np.clip(s, 0, 3).astype(np.int64)

    def extract_batch_data(self, batch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        names = batch.schema.names
        try:
            user_ids = batch.column("user_id").to_numpy(zero_copy_only=False).astype(np.uint64)
            item_ids = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)
        except Exception:
            d = batch.to_pydict()
            user_ids = np.asarray(d["user_id"], dtype=np.uint64)
            item_ids = np.asarray(d["item_id"], dtype=np.uint64)

        if "score" in names:
            try:
                scores = batch.column("score").to_numpy(zero_copy_only=False).astype(np.int64)
            except Exception:
                scores = np.asarray(batch.to_pydict()["score"], dtype=np.int64)
        else:
            scores = self._vk_feedback_to_score_int(batch)

        return user_ids, item_ids, scores

    def process_batch(
        self, user_ids: np.ndarray, item_ids: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        item_indices = self.item_loader.vectorized_lookup(item_ids)
        valid_mask = item_indices != -1

        if not valid_mask.any():
            return np.array([], dtype=np.uint64), np.zeros((0, self.embed_dim), dtype=np.float32), False

        item_indices_valid = item_indices[valid_mask]
        user_ids_valid = user_ids[valid_mask]
        scores_valid = scores[valid_mask]

        item_embs = self.item_loader.get_embeddings(item_indices_valid)
        scores_clipped = np.clip(scores_valid, 0, len(self.weight_arr) - 1)
        weights = self.weight_arr[scores_clipped]
        weighted_embs = item_embs * weights[:, None]

        unique_uids, inverse = np.unique(user_ids_valid, return_inverse=True)
        batch_emb = np.zeros((unique_uids.size, self.embed_dim), dtype=np.float32)
        weight_sums_abs = np.zeros(unique_uids.size, dtype=np.float32)

        np.add.at(batch_emb, inverse, weighted_embs)
        np.add.at(weight_sums_abs, inverse, np.abs(weights))

        weight_sums_safe = np.where(weight_sums_abs == 0, 1.0, weight_sums_abs)
        batch_emb = batch_emb / weight_sums_safe[:, None]

        return unique_uids, batch_emb, True

    def update_user_embeddings_ema(
        self, unique_uids: np.ndarray, batch_emb: np.ndarray, all_indices: np.ndarray
    ) -> None:
        old_embs = self.user_emb_mm[all_indices]
        new_embs = self.ema_alpha * old_embs + (1.0 - self.ema_alpha) * batch_emb
        self.user_emb_mm[all_indices] = new_embs

