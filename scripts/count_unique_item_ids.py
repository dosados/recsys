#!/usr/bin/env python3
"""Count unique item_id in week files and in embeddings file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

DEFAULT_DATASETS_ROOT = Path("/home/mants/Documents/own/recsys/datasets")
DEFAULT_EMBEDDINGS_PATH = Path("/home/mants/Documents/own/recsys/datasets/VK-LSVD/metadata/item_embeddings.npz")
DEFAULT_ITEM_COLUMN = "item_id"
DEFAULT_WEEK_GLOB = "week_*.parquet"


def collect_week_files(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(pattern)
        if path.is_file() and path.suffix == ".parquet" and path.name.startswith("week_")
    )


def unique_item_ids_in_week_files(files: list[Path], item_column: str) -> int:
    item_ids: set[int] = set()
    dataset = ds.dataset(files, format="parquet")
    for batch in dataset.to_batches(columns=[item_column]):
        unique_in_batch = pc.unique(batch[item_column]).to_pylist()
        item_ids.update(unique_in_batch)
    return len(item_ids)


def extract_item_ids_from_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        # Common key names for item ids.
        for key in ("item_id", "item_ids", "ids", "id"):
            if key in data.files:
                return np.asarray(data[key]).reshape(-1)

        # Fallback: if there is exactly one 1D array, treat it as ids.
        one_dim_arrays = [np.asarray(data[key]).reshape(-1) for key in data.files if np.asarray(data[key]).ndim == 1]
        if len(one_dim_arrays) == 1:
            return one_dim_arrays[0]

        available = ", ".join(data.files)
        raise ValueError(
            "Could not infer item id array in embeddings file. "
            f"Available keys: [{available}]"
        )


def unique_item_ids_in_embeddings(npz_path: Path) -> int:
    item_ids = extract_item_ids_from_npz(npz_path)
    return int(np.unique(item_ids).shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--item-column", default=DEFAULT_ITEM_COLUMN)
    parser.add_argument("--week-glob", default=DEFAULT_WEEK_GLOB)
    args = parser.parse_args()

    week_files = collect_week_files(args.datasets_root, args.week_glob)
    if not week_files:
        raise SystemExit(f"No week parquet files found in {args.datasets_root} with pattern {args.week_glob}")
    if not args.embeddings_path.exists():
        raise SystemExit(f"Embeddings file not found: {args.embeddings_path}")

    week_unique = unique_item_ids_in_week_files(week_files, args.item_column)
    emb_unique = unique_item_ids_in_embeddings(args.embeddings_path)

    print("=== Unique item_id statistics ===")
    print(f"week_files_count={len(week_files)}")
    print(f"unique_item_id_in_week_files={week_unique}")
    print(f"unique_item_id_in_item_embeddings={emb_unique}")


if __name__ == "__main__":
    main()
