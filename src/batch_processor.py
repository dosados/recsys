import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes batches of interactions to update user embeddings using EMA."""
    
    def __init__(
        self,
        embed_dim: int,
        weight_arr: np.ndarray,
        ema_alpha: float,
        item_loader,
        user_emb_mm: np.memmap,
        user_id_to_idx: Dict[int, int]
    ):
        """
        Initialize batch processor.
        
        Args:
            embed_dim: Embedding dimension
            weight_arr: Array mapping scores to weights
            ema_alpha: EMA parameter (p in p*old + (1-p)*new)
            item_loader: ItemEmbeddingLoader instance
            user_emb_mm: User embedding memmap
            user_id_to_idx: Mapping from user_id to index
        """
        self.embed_dim = embed_dim
        self.weight_arr = weight_arr
        self.ema_alpha = ema_alpha
        self.item_loader = item_loader
        self.user_emb_mm = user_emb_mm
        self.user_id_to_idx = user_id_to_idx
    
    def extract_batch_data(self, batch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #batch: PyArrow RecordBatch -> Tuple of (user_ids, item_ids, scores)
        try:
            user_ids = batch.column('user_id').to_numpy(zero_copy_only=False).astype(np.uint64)
            item_ids = batch.column('item_id').to_numpy(zero_copy_only=False).astype(np.uint64)
            scores = batch.column('score').to_numpy(zero_copy_only=False).astype(np.int64)
        except Exception:
            logger.warning('Exception occurred')
            d = batch.to_pydict()
            user_ids = np.asarray(d['user_id'], dtype=np.uint64)
            item_ids = np.asarray(d['item_id'], dtype=np.uint64)
            scores = np.asarray(d['score'], dtype=np.int64)
        
        return user_ids, item_ids, scores
    
    def process_batch(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            scores: Array of interaction scores
            
        Returns:
            Tuple of (unique_user_ids, batch_embeddings, has_valid_items)
            Returns (None, None, False) if no valid items found
        """
        item_indices = self.item_loader.vectorized_lookup(item_ids)
        valid_mask = item_indices != -1
        
        if not valid_mask.any():
            return None, None, False
        
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
        self,
        unique_uids: np.ndarray,
        batch_emb: np.ndarray,
        all_indices: np.ndarray
    ) -> None:
        """        
        Args:
            unique_uids: Array of unique user IDs in batch
            batch_emb: Batch embeddings (weighted average, shape: [len(unique_uids), embed_dim])
            all_indices: Array of memmap indices for all users (including newly added ones)
        """

        old_embs = self.user_emb_mm[all_indices]
        
        new_embs = self.ema_alpha * old_embs + (1.0 - self.ema_alpha) * batch_emb
        self.user_emb_mm[all_indices] = new_embs
