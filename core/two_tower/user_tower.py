import numpy as np
import pyarrow.parquet as pq


def build_user_vectors(users_path: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(users_path, columns=["user_id", "age", "gender", "geo", "train_interactions_rank"])
    user_ids = table.column("user_id").to_numpy(zero_copy_only=False).astype(np.uint64)
    age = table.column("age").to_numpy(zero_copy_only=False).astype(np.float32)
    gender = table.column("gender").to_numpy(zero_copy_only=False).astype(np.float32)
    geo = table.column("geo").to_numpy(zero_copy_only=False).astype(np.float32)
    rank = table.column("train_interactions_rank").to_numpy(zero_copy_only=False).astype(np.float32)
    base = np.stack([age, gender, geo, rank], axis=1)
    # Tile compact metadata features into target embedding dimensionality.
    reps = int(np.ceil(dim / base.shape[1]))
    vectors = np.tile(base, (1, reps))[:, :dim].astype(np.float32)
    return user_ids, vectors
