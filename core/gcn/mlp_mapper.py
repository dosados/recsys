import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class MLPMapper:
    def __init__(self, dim: int, lr: float = 1e-3, epochs: int = 2):
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit_on_items(self, interaction_item_emb: np.ndarray, content_item_emb: np.ndarray) -> None:
        if torch is None or nn is None:
            self.model = None
            return
        model = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        x = torch.from_numpy(interaction_item_emb.astype(np.float32))
        y = torch.from_numpy(content_item_emb.astype(np.float32))
        model.train()
        for _ in range(max(1, self.epochs)):
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        self.model = model.eval()

    def map_users(self, user_emb: np.ndarray) -> np.ndarray:
        if self.model is None or torch is None:
            return user_emb
        with torch.no_grad():
            x = torch.from_numpy(user_emb.astype(np.float32))
            out = self.model(x).cpu().numpy().astype(np.float32)
        return out
