import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass
class ItemEmbeddingSnapshot:
    item_ids: np.ndarray
    item_embeddings: np.ndarray


class ItemEmbeddingLoader:
    def __init__(self, items_path: str, embed_dim: int):
        self.items_path = Path(items_path)
        self.embed_dim = embed_dim
        self.snapshot: Optional[ItemEmbeddingSnapshot] = None
        self.keys_sorted: Optional[np.ndarray] = None
        self.vals_sorted: Optional[np.ndarray] = None
        self._npz: Any = None

    def load(self) -> None:
        if not self.items_path.exists():
            raise FileNotFoundError(f"Items file not found: {self.items_path}")

        logger.info("Loading item embeddings from %s", self.items_path)
        suffix = self.items_path.suffix.lower()
        if suffix == ".npz":
            self._load_npz()
        else:
            self._load_parquet()
        assert self.snapshot is not None
        logger.info("Loaded %d items, dim=%d", len(self.snapshot.item_ids), self.embed_dim)

    def _load_npz(self) -> None:
        if self._npz is not None:
            self._npz.close()
            self._npz = None
        self._npz = np.load(self.items_path, mmap_mode="r", allow_pickle=False)
        z = self._npz
        if "item_id" not in z.files or "embedding" not in z.files:
            raise KeyError("expected arrays 'item_id' and 'embedding'")
        emb = z["embedding"]
        if emb.ndim != 2:
            raise ValueError("array 'embedding' must be 2-D (n_items, dim)")
        if emb.shape[1] < self.embed_dim:
            raise ValueError("Configured embed_dim is larger than item embedding width")
        item_ids = np.asarray(z["item_id"], dtype=np.uint64)
        sl = emb[:, : self.embed_dim]
        if sl.dtype == np.float32:
            embeddings = sl
        else:
            embeddings = np.asarray(sl, dtype=np.float32)

        order = np.argsort(item_ids)
        self.keys_sorted = item_ids[order]
        self.vals_sorted = np.arange(item_ids.size, dtype=np.int64)[order]
        self.snapshot = ItemEmbeddingSnapshot(item_ids=item_ids, item_embeddings=embeddings)

    def _load_parquet(self) -> None:
        table = pq.read_table(self.items_path, columns=["item_id", "embedding"])
        item_ids = table.column("item_id").to_numpy(zero_copy_only=False).astype(np.uint64)

        embedding_py = table.column("embedding").to_pylist()
        embeddings = np.asarray(embedding_py, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Column 'embedding' must contain fixed-size vectors")
        if embeddings.shape[1] < self.embed_dim:
            raise ValueError("Configured embed_dim is larger than item embedding size")
        embeddings = embeddings[:, : self.embed_dim]

        order = np.argsort(item_ids)
        self.keys_sorted = item_ids[order]
        self.vals_sorted = np.arange(item_ids.size, dtype=np.int64)[order]
        self.snapshot = ItemEmbeddingSnapshot(item_ids=item_ids, item_embeddings=embeddings)

    def vectorized_lookup(self, item_ids: np.ndarray) -> np.ndarray:
        if self.keys_sorted is None or self.keys_sorted.size == 0:
            return np.full(item_ids.shape, -1, dtype=np.int64)

        pos = np.searchsorted(self.keys_sorted, item_ids)
        pos_safe = np.clip(pos, 0, self.keys_sorted.size - 1)
        matches = (pos < self.keys_sorted.size) & (self.keys_sorted[pos_safe] == item_ids)
        return np.where(matches, self.vals_sorted[pos_safe], -1).astype(np.int64)

    def get_embeddings(self, indices: np.ndarray) -> np.ndarray:
        if self.snapshot is None:
            raise RuntimeError("Item embeddings are not loaded")
        out = self.snapshot.item_embeddings[indices]
        return np.ascontiguousarray(out, dtype=np.float32)
