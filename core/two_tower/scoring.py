import numpy as np


def cosine_scores(query: np.ndarray, items: np.ndarray) -> np.ndarray:
    q = query.astype(np.float32)
    qn = np.linalg.norm(q)
    if qn == 0.0:
        qn = 1.0
    q = q / qn
    norms = np.linalg.norm(items, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (items / norms) @ q
