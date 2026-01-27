import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Dict, Optional
import pyarrow.parquet as pq
from tqdm import tqdm
import logging

from .checkpoint import load_checkpoint, save_checkpoint
from .memmap_manager import capacity_from_memmap_file, create_user_memmap, open_user_memmap, grow_user_memmap

from .item_embeddings import ItemEmbeddingLoader
from .batch_processor import BatchProcessor
from .constants import (
    EMBED_DIM,
    BATCH_SIZE,
    SCORE_WEIGHTS,
    INITIAL_CAPACITY,
    MAX_CAPACITY,
    SAVE_EVERY_BATCHES,
    GC_EVERY,
    EMA_ALPHA
)

logger = logging.getLogger(__name__)


class UserEmbeddingAggregator:
    """
    Aggregates user embeddings from item embeddings and interactions using EMA.
    
    Features:
    - Streams datasets from parquet files
    - Uses external memmap files for embeddings
    - Memory-efficient processing
    - Exponential Moving Average for gradual adaptation
    """
    
    def __init__(
        self,
        item_embeddings_file: str,
        interactions_files: List[str],
        output_file: str = "user_embeddings.npz",
        work_dir: str = ".",
        embed_dim: int = EMBED_DIM,
        batch_size: int = BATCH_SIZE,
        score_weights: Dict[int, float] = SCORE_WEIGHTS,
        ema_alpha: float = EMA_ALPHA,
        initial_capacity: int = INITIAL_CAPACITY,
        max_capacity: int = MAX_CAPACITY,
        save_every_batches: int = SAVE_EVERY_BATCHES,
        gc_every: int = GC_EVERY
    ):
        """
        Initialize the aggregator.
        
        Args:
            item_embeddings_file: Path to item embeddings .npz file
            interactions_files: List of paths to interaction parquet files
            output_file: Output file path for final user embeddings
            work_dir: Working directory for memmap files and checkpoints
            embed_dim: Embedding dimension
            batch_size: Batch size for processing
            score_weights: Dictionary mapping scores to weights
            ema_alpha: EMA parameter (p in p*old + (1-p)*new)
            initial_capacity: Initial capacity for user embeddings
            max_capacity: Maximum capacity for user embeddings
            save_every_batches: Save checkpoint every N batches
            gc_every: Run garbage collection every N batches
        """
        self.item_embeddings_file = item_embeddings_file
        self.interactions_files = interactions_files
        self.output_file = output_file
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.score_weights = score_weights 
        self.ema_alpha = ema_alpha
        self.initial_capacity = initial_capacity
        self.max_capacity = max_capacity
        self.save_every_batches = save_every_batches
        self.gc_every = gc_every
        
        self.item_emb_memmap_file = self.work_dir / "item_embeddings_memmap.dat"
        self.user_emb_memmap_file = self.work_dir / "user_embeddings.npy"
        self.user_ids_file = self.work_dir / "user_ids.npy"
        self.checkpoint_file = self.work_dir / "checkpoint.json"
        
        self.item_loader: Optional[ItemEmbeddingLoader] = None
        self.user_emb_mm: Optional[np.memmap] = None
        self.weight_arr: Optional[np.ndarray] = None
        self.user_id_to_idx: Dict[int, int] = {}
        self.n_users: int = 0
        self.current_capacity: int = 0
        self.batch_processor: Optional[BatchProcessor] = None
    
    def _prepare_weight_array(self):
        max_score = max(self.score_weights.keys()) if self.score_weights else 0
        self.weight_arr = np.zeros(max_score + 1, dtype=np.float32)
        for k, v in self.score_weights.items():
            if 0 <= k <= max_score:
                self.weight_arr[k] = float(v)
    
    def _restore_state(self):

        checkpoint = load_checkpoint(self.checkpoint_file)
        
        detected_capacity = capacity_from_memmap_file(self.user_emb_memmap_file, 4, self.embed_dim)
        detected_capacity = max(detected_capacity, checkpoint.get("capacity", 0))
        
        if detected_capacity > 0:
            # Existing embeddings found - preserve them (long-term memory)
            logger.info(f"Restoring embeddings from existing file (capacity={detected_capacity})")
            self.user_emb_mm = open_user_memmap(self.user_emb_memmap_file, detected_capacity, self.embed_dim)
            self.current_capacity = detected_capacity
        else:
            # No existing embeddings - create new
            est_capacity = max(self.initial_capacity, checkpoint.get("capacity", 0) or self.initial_capacity)
            capacity = min(est_capacity, self.max_capacity)
            logger.info(f"Creating new embeddings memmap (capacity={capacity})")
            self.user_emb_mm = create_user_memmap(self.user_emb_memmap_file, capacity, self.embed_dim)
            self.current_capacity = capacity
        
        # Restore user_id -> index mapping
        if self.user_ids_file.exists() and checkpoint.get("n_users", 0) > 0:
            user_ids_arr = np.load(self.user_ids_file)
            self.n_users = int(min(len(user_ids_arr), checkpoint.get("n_users", 0)))
            self.user_id_to_idx = {
                int(user_ids_arr[i]): int(i)
                for i in range(self.n_users)
            }
            if self.n_users > self.current_capacity:
                raise RuntimeError("n_users exceeds memmap capacity")
            logger.info(f"Restored {self.n_users} users from checkpoint")
        else:
            self.user_id_to_idx = {}
            self.n_users = int(checkpoint.get("n_users", 0)) if checkpoint.get("n_users", 0) else 0
            user_ids_arr = np.zeros(self.current_capacity, dtype=np.uint64)
        
        # Grow if needed
        if checkpoint.get("capacity", 0) > self.current_capacity:
            new_cap = checkpoint["capacity"]
            if new_cap <= self.max_capacity and new_cap > self.current_capacity:
                self.user_emb_mm = grow_user_memmap(
                    self.user_emb_memmap_file,
                    self.user_emb_mm,
                    self.current_capacity,
                    new_cap,
                    self.embed_dim
                )
                new_user_ids_arr = np.zeros(new_cap, dtype=np.uint64)
                new_user_ids_arr[:len(user_ids_arr)] = user_ids_arr
                user_ids_arr = new_user_ids_arr
                self.current_capacity = new_cap
        
        return checkpoint, user_ids_arr
    
    def _grow_memmap_if_needed(self, required_total: int, user_ids_arr: np.ndarray) -> np.ndarray:
        if required_total > self.current_capacity:
            new_cap = self.current_capacity
            while new_cap < required_total:
                new_cap = int(new_cap * 2)
                if new_cap > self.max_capacity:
                    new_cap = self.max_capacity
                    break
            if new_cap <= self.current_capacity:
                raise MemoryError("Cannot grow memmap further")
            
            self.user_emb_mm = grow_user_memmap(
                self.user_emb_memmap_file,
                self.user_emb_mm,
                self.current_capacity,
                new_cap,
                self.embed_dim
            )
            self.batch_processor.user_emb_mm = self.user_emb_mm
            new_user_ids_arr = np.zeros(new_cap, dtype=np.uint64)
            new_user_ids_arr[:len(user_ids_arr)] = user_ids_arr
            user_ids_arr = new_user_ids_arr
            self.current_capacity = new_cap
        elif required_total > len(user_ids_arr):
            tmp_arr = np.zeros(required_total, dtype=np.uint64)
            tmp_arr[:len(user_ids_arr)] = user_ids_arr
            user_ids_arr = tmp_arr
        
        return user_ids_arr
    
    def _add_new_users(
        self,
        unique_uids: np.ndarray,
        existing_indices: np.ndarray,
        user_ids_arr: np.ndarray
    ) -> np.ndarray:
        """Add new users and return updated user_ids_arr."""
        new_mask = existing_indices == -1
        n_new = int(new_mask.sum())
        
        if n_new > 0:
            required_total = self.n_users + n_new
            user_ids_arr = self._grow_memmap_if_needed(required_total, user_ids_arr)
            
            new_indices = np.arange(self.n_users, self.n_users + n_new, dtype=np.int64)
            existing_indices[new_mask] = new_indices
            new_uids = unique_uids[new_mask]
            user_ids_arr[new_indices] = new_uids
            
            self.user_id_to_idx.update({
                int(uid): int(idx) for uid, idx in zip(new_uids, new_indices)
            })
            self.n_users += n_new
        
        return user_ids_arr
    
    def process(self):
        logger.info("Starting user embedding aggregation with EMA...")
        
        self.item_loader = ItemEmbeddingLoader(
            self.item_embeddings_file,
            self.item_emb_memmap_file,
            self.embed_dim
        )
        self.item_loader.load()
        
        self._prepare_weight_array()
        
        checkpoint, user_ids_arr = self._restore_state()
        
        self.batch_processor = BatchProcessor(
            self.embed_dim,
            self.weight_arr,
            self.ema_alpha,
            self.item_loader,
            self.user_emb_mm,
            self.user_id_to_idx
        )
        

        for file_path in self.interactions_files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}, skipping")
                continue
            
            logger.info(f"Processing file: {file_path}")
            
            parquet = pq.ParquetFile(file_path)
            n_rows = parquet.metadata.num_rows
            n_batches = (n_rows + self.batch_size - 1) // self.batch_size
            
            it = parquet.iter_batches(batch_size=self.batch_size)
            for batch_idx, batch in enumerate(tqdm(it, total=n_batches, desc=f"Processing {os.path.basename(file_path)}")):
                user_ids, item_ids, scores = self.batch_processor.extract_batch_data(batch)
                
                
                unique_uids, batch_emb, has_valid = self.batch_processor.process_batch(
                    user_ids, item_ids, scores
                )
                
                if not has_valid:
                    if batch_idx % self.gc_every == 0:
                        gc.collect()
                    continue
                
                existing_indices = np.array(
                    [self.batch_processor.user_id_to_idx.get(int(uid), -1) for uid in unique_uids],
                    dtype=np.int64
                )
                new_mask = existing_indices == -1
                n_new = int(new_mask.sum())
                
                if n_new > 0:
                    user_ids_arr = self._add_new_users(
                        unique_uids, existing_indices, user_ids_arr
                    )
                    self.batch_processor.user_id_to_idx = self.user_id_to_idx
                    existing_indices = np.array(
                        [self.batch_processor.user_id_to_idx.get(int(uid), -1) for uid in unique_uids],
                        dtype=np.int64
                    )
                

                self.batch_processor.update_user_embeddings_ema(
                    unique_uids, batch_emb, existing_indices
                )
                
                # Periodic garbage collection
                if batch_idx % self.gc_every == 0:
                    gc.collect()
        
        if self.n_users > 0:
            sorted_order = np.argsort(user_ids_arr[:self.n_users])
            sorted_user_ids = (user_ids_arr[:self.n_users])[sorted_order].copy()
            sorted_embeddings = np.asarray(self.user_emb_mm[sorted_order]).copy()
            self.user_emb_mm[:self.n_users] = sorted_embeddings
            user_ids_arr[:self.n_users] = sorted_user_ids
            self.user_id_to_idx = {int(uid): i for i, uid in enumerate(sorted_user_ids)}
        
        checkpoint["n_users"] = self.n_users
        checkpoint["capacity"] = self.current_capacity
        self._save_checkpoint(checkpoint, user_ids_arr)
        
        self.user_emb_mm.flush()
        gc.collect()
        
        logger.info(f"Processing complete. Total users: {self.n_users}")
    
    def _save_checkpoint(self, checkpoint: Dict, user_ids_arr: np.ndarray):
        self.user_emb_mm.flush()
        save_checkpoint(
            checkpoint,
            self.checkpoint_file,
            self.user_ids_file,
            user_ids_arr,
            self.n_users
        )
    
    def finalize(self):
        logger.info("Finalizing user embeddings...")
        
        if self.n_users == 0:
            np.savez_compressed(
                self.output_file,
                user_id=np.array([], dtype=np.uint32),
                embedding=np.zeros((0, self.embed_dim), dtype=np.float16)
            )
            logger.info("No users found, saved empty embeddings")
            return
        
        user_ids_final = np.empty(self.n_users, dtype=np.uint32)
        for uid, idx in self.user_id_to_idx.items():
            user_ids_final[idx] = np.uint32(uid)
        
        embeddings = self.user_emb_mm[:self.n_users].astype(np.float16)
        
        np.savez_compressed(self.output_file, user_id=user_ids_final, embedding=embeddings)
        logger.info(f"Saved final user embeddings to {self.output_file}")
        logger.info(f"Shape: {embeddings.shape}, Users: {len(user_ids_final)}")
