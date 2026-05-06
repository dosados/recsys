from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _normalized_random_embeddings(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    emb = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return emb / norms


def generate_synthetic_dataset(
    output_dir: str,
    n_users: int = 5_000,
    n_items: int = 2_000,
    embedding_dim: int = 32,
    interactions_per_user: int = 25,
    seed: int = 42,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, n_users + 1, dtype=np.uint64)
    item_ids = np.arange(1, n_items + 1, dtype=np.uint64)

    item_embeddings = _normalized_random_embeddings(rng, n_items, embedding_dim)
    user_pref = _normalized_random_embeddings(rng, n_users, embedding_dim)

    users_path = out / "users.parquet"
    items_path = out / "items.parquet"
    interactions_path = out / "interactions.parquet"

    users_table = pa.Table.from_arrays([pa.array(user_ids)], names=["user_id"])
    pq.write_table(users_table, users_path, compression="zstd")

    item_emb_list = [row.tolist() for row in item_embeddings]
    items_table = pa.Table.from_arrays(
        [pa.array(item_ids), pa.array(item_emb_list)], names=["item_id", "embedding"]
    )
    pq.write_table(items_table, items_path, compression="zstd")

    sampled_users = np.repeat(user_ids, interactions_per_user)
    sampled_idx = np.repeat(np.arange(n_users, dtype=np.int64), interactions_per_user)

    raw_scores = user_pref[sampled_idx] @ item_embeddings.T
    noise = rng.normal(0.0, 0.15, size=raw_scores.shape).astype(np.float32)
    raw_scores = raw_scores + noise

    top_pool = np.argpartition(raw_scores, -50, axis=1)[:, -50:]
    pick = rng.integers(0, top_pool.shape[1], size=top_pool.shape[0])
    sampled_items_idx = top_pool[np.arange(top_pool.shape[0]), pick]
    sampled_items = item_ids[sampled_items_idx]

    sampled_raw = raw_scores[np.arange(raw_scores.shape[0]), sampled_items_idx]
    sampled_raw = np.clip(sampled_raw, -1.0, 1.0)
    score = np.digitize(sampled_raw, bins=[-0.2, 0.2, 0.55]).astype(np.int8)

    base_ts = 1_700_000_000
    timestamps = base_ts + rng.integers(0, 3600 * 24 * 14, size=len(sampled_users))
    event_type = np.where(score >= 2, "positive", np.where(score == 1, "neutral", "negative"))
    event_value = score.astype(np.float32) + rng.normal(0.0, 0.05, size=len(score)).astype(np.float32)

    interactions_table = pa.Table.from_arrays(
        [
            pa.array(sampled_users),
            pa.array(sampled_items),
            pa.array(score),
            pa.array(timestamps, type=pa.int64()),
            pa.array(event_type),
            pa.array(event_value),
        ],
        names=["user_id", "item_id", "score", "timestamp", "event_type", "event_value"],
    )
    pq.write_table(interactions_table, interactions_path, compression="zstd")

    return {
        "users_path": str(users_path),
        "items_path": str(items_path),
        "interactions_path": str(interactions_path),
    }

