from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq


POSITIVE_COLUMNS = ["like", "share", "bookmark", "click_on_author", "open_comments"]


def is_positive_batch(batch: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros_like(batch["timespent"], dtype=bool)
    for col in POSITIVE_COLUMNS:
        if col in batch:
            mask |= batch[col].astype(bool)
    mask |= batch["timespent"].astype(np.int64) > 30
    return mask


def _finalize_user_sets_to_holdout(user_items: dict[int, set[int]]) -> dict[int, np.ndarray]:
    holdout: dict[int, np.ndarray] = {}
    for uid, items in user_items.items():
        holdout[uid] = np.array(sorted(items), dtype=np.uint64)
    return holdout


def _accumulate_positive_edges(
    interactions_path: str,
    user_items: dict[int, set[int]],
    batch_size: int = 262_144,
) -> None:
    columns = ["user_id", "item_id", "timespent"] + POSITIVE_COLUMNS
    pf = pq.ParquetFile(interactions_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        payload = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in columns}
        pos_mask = is_positive_batch(payload)
        if not np.any(pos_mask):
            continue
        user_ids = payload["user_id"][pos_mask]
        item_ids = payload["item_id"][pos_mask]
        for i in range(len(user_ids)):
            user_items[int(user_ids[i])].add(int(item_ids[i]))


def build_positives_holdout(interactions_path: str, batch_size: int = 262_144) -> dict[int, np.ndarray]:
    user_items: dict[int, set[int]] = defaultdict(set)
    _accumulate_positive_edges(interactions_path, user_items, batch_size=batch_size)
    return _finalize_user_sets_to_holdout(user_items)


def build_positives_holdout_from_paths(
    interactions_paths: list[str], batch_size: int = 262_144
) -> dict[int, np.ndarray]:
    merged: dict[int, set[int]] = defaultdict(set)
    for path in interactions_paths:
        _accumulate_positive_edges(path, merged, batch_size=batch_size)
    return _finalize_user_sets_to_holdout(merged)
