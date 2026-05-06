import numpy as np


class RandomSubsetSampler:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def sample(self, n_items: int, subset_size: int | None) -> np.ndarray | None:
        if subset_size is None or subset_size <= 0 or subset_size >= n_items:
            return None
        return np.sort(self.rng.choice(n_items, size=subset_size, replace=False).astype(np.int64))
