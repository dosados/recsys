import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_checkpoint(checkpoint_file: Path) -> dict[str, Any]:
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"n_users": 0, "capacity": 0}


def save_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_file: Path,
    user_ids_file: Path,
    user_ids_arr: np.ndarray,
    n_users: int,
) -> None:
    tmp_user_ids = user_ids_file.with_suffix(".tmp.npy")
    np.save(tmp_user_ids, user_ids_arr[:n_users])
    tmp_user_ids.replace(user_ids_file)

    tmp_checkpoint = checkpoint_file.with_suffix(".tmp")
    with open(tmp_checkpoint, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_checkpoint.replace(checkpoint_file)

