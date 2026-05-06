from pathlib import Path

import numpy as np


def capacity_from_memmap_file(fname: Path, per_elem_bytes: int, second_dim: int = 1) -> int:
    if not fname.exists():
        return 0
    size = fname.stat().st_size
    return size // (per_elem_bytes * second_dim)


def create_user_memmap(emb_file: Path, capacity: int, embed_dim: int) -> np.memmap:
    emb_mm = np.memmap(emb_file, dtype=np.float32, mode="w+", shape=(capacity, embed_dim))
    emb_mm[:] = 0.0
    emb_mm.flush()
    return np.memmap(emb_file, dtype=np.float32, mode="r+", shape=(capacity, embed_dim))


def open_user_memmap(emb_file: Path, capacity: int, embed_dim: int) -> np.memmap:
    return np.memmap(emb_file, dtype=np.float32, mode="r+", shape=(capacity, embed_dim))


def grow_user_memmap(
    emb_file: Path,
    old_emb_mm: np.memmap,
    old_capacity: int,
    new_capacity: int,
    embed_dim: int,
) -> np.memmap:
    new_emb_tmp = emb_file.with_suffix(".tmp")
    new_emb = np.memmap(new_emb_tmp, dtype=np.float32, mode="w+", shape=(new_capacity, embed_dim))
    new_emb[:] = 0.0
    new_emb[:old_capacity] = old_emb_mm[:old_capacity]
    new_emb.flush()

    try:
        old_emb_mm._mmap.close()
    except Exception:
        pass

    if emb_file.exists():
        emb_file.unlink()
    new_emb_tmp.replace(emb_file)

    try:
        new_emb._mmap.close()
    except Exception:
        pass
    del new_emb

    return open_user_memmap(emb_file, new_capacity, embed_dim)

