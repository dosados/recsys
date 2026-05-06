import numpy as np


def apply_interest_drift(centers: np.ndarray, alpha: float) -> np.ndarray:
    return centers * (1.0 - alpha)
