"""Hardcoded VK-LSVD layout under the repo; override with --data-root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBSAMPLE_NAME = "ur0.1"


def vk_layout(data_root: str | Path | None = None) -> dict[str, Any]:
    if data_root is None:
        root = REPO_ROOT / "datasets" / "VK-LSVD"
    else:
        root = Path(data_root)
        if not root.is_absolute():
            root = REPO_ROOT / root
    sub = root / "subsamples" / SUBSAMPLE_NAME
    train_weeks = [sub / "train" / f"week_{i:02d}.parquet" for i in range(25)]
    return {
        "root": root,
        "users": root / "metadata" / "users_metadata.parquet",
        "items": root / "metadata" / "items_metadata.parquet",
        "item_embeddings": root / "metadata" / "item_embeddings.npz",
        "train_interactions": [str(p) for p in train_weeks],
        "validation_interactions": str(sub / "validation" / "week_25.parquet"),
    }


def apply_dataset_cli_defaults(args: Any) -> None:
    cmd = args.command
    layout = vk_layout(getattr(args, "data_root", None))
    small_sample = bool(getattr(args, "small_sample", False))

    if getattr(args, "users_path", None) is None:
        args.users_path = str(layout["users"])
    if getattr(args, "items_path", None) is None:
        args.items_path = str(layout["items"])
    if getattr(args, "item_embeddings_npz", None) is None:
        args.item_embeddings_npz = str(layout["item_embeddings"])

    ip = getattr(args, "interactions_path", None)
    if small_sample:
        sample_paths = list(layout["train_interactions"][:6])
    else:
        sample_paths = list(layout["train_interactions"])

    if cmd in ("train", "update"):
        args._interaction_paths = sample_paths
        if ip is None:
            args.interactions_path = sample_paths[0]
        else:
            args.interactions_path = ip
            args._interaction_paths = [ip]
    elif cmd == "evaluate":
        # Holdout/metrics use a single parquet only; never reuse the multi-week train list here.
        args._interaction_paths = None
        if ip is None:
            # In small-sample mode we evaluate on the same compact train window.
            args.interactions_path = sample_paths[-1] if small_sample else layout["validation_interactions"]
        else:
            args.interactions_path = ip
    else:
        args._interaction_paths = None
