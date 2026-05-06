import numpy as np


def update_centers(centers: np.ndarray, vector: np.ndarray, lr: float) -> np.ndarray:
    dists = np.linalg.norm(centers - vector.reshape(1, -1), axis=1)
    idx = int(np.argmin(dists))
    centers[idx] = (1.0 - lr) * centers[idx] + lr * vector
    return centers
