import numpy as np


def train_bpr_stub(user_vec: np.ndarray, item_vec: np.ndarray, epochs: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # Lightweight MVP placeholder for BPR optimization.
    for _ in range(max(1, epochs)):
        user_vec = user_vec * 0.999
        item_vec = item_vec * 0.999
    return user_vec.astype(np.float32), item_vec.astype(np.float32)
