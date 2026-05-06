import numpy as np
import pyarrow.parquet as pq


def build_item_vectors(items_path: str, item_embeddings_npz: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    items = pq.read_table(items_path, columns=["item_id", "author_id", "duration", "train_interactions_rank"])
    meta_ids = items.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)
    author = items.column("author_id").to_numpy(zero_copy_only=False).astype(np.float32)
    duration = items.column("duration").to_numpy(zero_copy_only=False).astype(np.float32)
    rank = items.column("train_interactions_rank").to_numpy(zero_copy_only=False).astype(np.float32)
    meta = np.stack([author, duration, rank], axis=1)
    reps = int(np.ceil(dim / meta.shape[1]))
    meta_vec = np.tile(meta, (1, reps))[:, :dim].astype(np.float32)

    emb = np.load(item_embeddings_npz)
    emb_ids = emb["item_id"].astype(np.uint64)
    emb_vec = emb["embedding"].astype(np.float32)
    emb_map = {int(i): idx for idx, i in enumerate(emb_ids)}

    out = np.zeros((len(meta_ids), dim), dtype=np.float32)
    for i, item_id in enumerate(meta_ids):
        j = emb_map.get(int(item_id))
        if j is not None:
            out[i] = 0.5 * meta_vec[i] + 0.5 * emb_vec[j, :dim]
        else:
            out[i] = meta_vec[i]
    return meta_ids, out
