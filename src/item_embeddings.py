import numpy as np
import gc
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ItemEmbeddingLoader:
    """Handles loading and lookup of item embeddings."""
    
    def __init__(self, embeddings_file: str, memmap_file: Path, embed_dim: int):
        """
        Initialize item embedding loader.
        
        Args:
            embeddings_file: Path to .npz file with item embeddings
            memmap_file: Path where memmap file will be stored
            embed_dim: Embedding dimension to use
        """
        self.embeddings_file = embeddings_file
        self.memmap_file = memmap_file
        self.embed_dim = embed_dim
        self.memmap: Optional[np.memmap] = None
        self.keys_sorted: Optional[np.ndarray] = None
        self.vals_sorted: Optional[np.ndarray] = None
    
    def load(self):
        """Load item embeddings into memmap and prepare lookup structures."""
        logger.info("Loading item embeddings into memmap...")
        
        if not (Path(self.embeddings_file)).exists():
            raise FileNotFoundError(f"Embeddings file {self.embeddings_file} not found")

        item_data = np.load(self.embeddings_file)
        item_ids = item_data["item_id"].astype(np.uint64)
        item_embeddings = item_data["embedding"].astype(np.float32)
        
        if self.memmap_file.exists():
            self.memmap_file.unlink()
        
        self.memmap = np.memmap(
            self.memmap_file,
            dtype=np.float32,
            mode='w+',
            shape=item_embeddings.shape
        )
        self.memmap[:] = item_embeddings[:]
        self.memmap.flush()
        
        del item_embeddings
        gc.collect()
        
        if item_ids.size > 0:
            vals = np.arange(item_ids.size, dtype=np.int64)
            order = np.argsort(item_ids)
            self.keys_sorted = item_ids[order]
            self.vals_sorted = vals[order]
            del item_ids
            gc.collect()
        else:
            self.keys_sorted = np.array([], dtype=np.uint64)
            self.vals_sorted = np.array([], dtype=np.int64)
        
        logger.info(f"Loaded {len(self.keys_sorted)} item embeddings")
    
    def vectorized_lookup(self, item_ids: np.ndarray) -> np.ndarray:
        """
        Fast vectorized lookup of item indices using searchsorted.
        
        Args:
            item_ids: Array of item IDs to look up
            
        Returns:
            Array of indices (-1 for not found)
        """
        if self.keys_sorted is None or self.keys_sorted.size == 0:
            return np.full(item_ids.shape, -1, dtype=np.int64)
        
        pos = np.searchsorted(self.keys_sorted, item_ids)
        pos_safe = np.clip(pos, 0, self.keys_sorted.size - 1)
        matches = (pos < self.keys_sorted.size) & (self.keys_sorted[pos_safe] == item_ids)
        return np.where(matches, self.vals_sorted[pos_safe], -1).astype(np.int64)
    
    def get_embeddings(self, indices: np.ndarray) -> np.ndarray:
        if self.memmap is None:
            raise RuntimeError("Item embeddings not loaded. Call load() first.")
        return self.memmap[indices][:, :self.embed_dim]
