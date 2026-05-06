from typing import Sequence

import numpy as np
import pyarrow.parquet as pq


def collect_unique_users(interaction_paths: Sequence[str], batch_size: int = 262_144) -> np.ndarray:
    users: set[int] = set()
    for path in interaction_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["user_id"]):
            u = batch.column("user_id").to_numpy(zero_copy_only=False)
            users.update(int(x) for x in u.tolist())
    return np.array(sorted(users), dtype=np.uint64)
