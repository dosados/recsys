import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def capacity_from_memmap_file(fname: Path, per_elem_bytes: int, second_dim: int = 1) -> int:
    """Calculate capacity from existing memmap file."""
    if not fname.exists():
        return 0
    size = fname.stat().st_size
    return size // (per_elem_bytes * second_dim)


def create_user_memmap(emb_file: Path, capacity: int, embed_dim: int) -> np.memmap:
    """Create user embedding memmap file."""
    emb_mm = np.memmap(
        emb_file,
        dtype=np.float32,
        mode='w+',
        shape=(capacity, embed_dim)
    )
    emb_mm[:] = 0.0
    emb_mm.flush()
    
    emb_mm = np.memmap(
        emb_file,
        dtype=np.float32,
        mode='r+',
        shape=(capacity, embed_dim)
    )
    
    return emb_mm


def open_user_memmap(emb_file: Path, capacity: int, embed_dim: int) -> np.memmap:
    """Open existing user memmap file."""
    emb_mm = np.memmap(
        emb_file,
        dtype=np.float32,
        mode='r+',
        shape=(capacity, embed_dim)
    )
    return emb_mm


def grow_user_memmap(
    emb_file: Path,
    old_emb_mm: np.memmap,
    old_capacity: int,
    new_capacity: int,
    embed_dim: int
) -> np.memmap:
    """Grow user memmap file to new capacity."""
    logger.info(f"Growing memmap: {old_capacity} -> {new_capacity}")

    new_emb_tmp = emb_file.with_suffix('.tmp')
    new_emb = np.memmap(
        new_emb_tmp,
        dtype=np.float32,
        mode='w+',
        shape=(new_capacity, embed_dim)
    )
    new_emb[:] = 0.0
    new_emb[:old_capacity] = old_emb_mm[:old_capacity]
    new_emb.flush()

    try:
        old_emb_mm._mmap.close()
    except Exception:
        pass

    emb_file.unlink()
    new_emb_tmp.replace(emb_file)

    try:
        new_emb._mmap.close()
    except Exception:
        pass
    del new_emb

    return open_user_memmap(emb_file, new_capacity, embed_dim)


def create_item_memmap(
    memmap_file: Path,
    embeddings: np.ndarray
) -> np.memmap:
    """Create memmap file for item embeddings."""
    if memmap_file.exists():
        memmap_file.unlink()
    
    memmap = np.memmap(
        memmap_file,
        dtype=np.float32,
        mode='w+',
        shape=embeddings.shape
    )
    memmap[:] = embeddings[:]
    memmap.flush()
    
    return memmap


def open_item_memmap(memmap_file: Path, shape: Tuple[int, ...]) -> np.memmap:
    """Open existing item memmap file."""
    return np.memmap(
        memmap_file,
        dtype=np.float32,
        mode='r',
        shape=shape
    )
