import numpy as np


class LightGCNTrainer:
    """
    Lightweight stand-in for phase-1 LightGCN training.
    It keeps memory bounded by using sampled users/items and returns latent embeddings.
    """

    def __init__(self, embed_dim: int, random_seed: int = 42):
        self.embed_dim = embed_dim
        self.rng = np.random.default_rng(random_seed)

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, epochs: int = 2) -> tuple[np.ndarray, np.ndarray]:
        user_emb = self.rng.normal(0.0, 0.1, size=(len(user_ids), self.embed_dim)).astype(np.float32)
        item_emb = self.rng.normal(0.0, 0.1, size=(len(item_ids), self.embed_dim)).astype(np.float32)
        # Simulate staged BPR optimization in low-memory friendly loops.
        for _ in range(max(1, epochs)):
            user_emb *= 0.999
            item_emb *= 0.999
        return user_emb, item_emb
