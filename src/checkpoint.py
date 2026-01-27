import json
import os
from pathlib import Path
from typing import Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file: Path) -> Dict:
    """Load checkpoint from file. Only stores n_users and capacity for memmap restoration."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {"n_users": 0, "capacity": 0}


def save_checkpoint(
    checkpoint: Dict,
    checkpoint_file: Path,
    user_ids_file: Path,
    user_ids_arr: np.ndarray,
    n_users: int
):
    """Save checkpoint atomically. Only saves n_users and capacity."""
    tmp_user_ids = user_ids_file.with_suffix('.tmp.npy')
    np.save(tmp_user_ids, user_ids_arr[:n_users])
    tmp_user_ids.replace(user_ids_file)
    
    tmp_checkpoint = checkpoint_file.with_suffix('.tmp')
    with open(tmp_checkpoint, 'w') as f:
        json.dump(checkpoint, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_checkpoint.replace(checkpoint_file)
    
    logger.debug(f"Checkpoint saved: {n_users} users")
