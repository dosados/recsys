import logging
import os
from pathlib import Path

import pyarrow.parquet as pq

from core.contracts import DatasetPaths

from .schemas import (
    validate_interactions_schema,
    validate_items_schema,
    validate_users_schema,
)

logger = logging.getLogger(__name__)


def load_dataset_tables(dataset: DatasetPaths) -> None:
    """Check paths exist and parquet schemas match the expected dataset layout."""
    for label, path in [
        ("users", dataset.users_path),
        ("items", dataset.items_path),
    ]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"{label} path missing or not found: {path}")

    inter_paths = dataset.interaction_paths or [dataset.interactions_path]
    for i, path in enumerate(inter_paths):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"interactions[{i}] missing or not found: {path}")
    if dataset.interactions_path and not os.path.exists(dataset.interactions_path):
        raise FileNotFoundError(f"interactions_path missing or not found: {dataset.interactions_path}")
    if dataset.item_embeddings_npz:
        p = Path(dataset.item_embeddings_npz)
        if not p.exists():
            raise FileNotFoundError(f"item embeddings npz not found: {p}")

    validate_users_schema(list(pq.ParquetFile(dataset.users_path).schema_arrow.names))
    embeddings_in_parquet = not bool(dataset.item_embeddings_npz)
    validate_items_schema(
        list(pq.ParquetFile(dataset.items_path).schema_arrow.names),
        embeddings_in_parquet=embeddings_in_parquet,
    )
    validate_interactions_schema(list(pq.ParquetFile(inter_paths[0]).schema_arrow.names))
    logger.debug("Dataset paths validated (%d interaction file(s))", len(inter_paths))
