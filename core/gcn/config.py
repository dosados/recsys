from dataclasses import dataclass


@dataclass
class GCNConfig:
    embed_dim: int = 64
    epochs: int = 2
    batch_size: int = 262_144
    lr: float = 1e-3
    work_dir: str = "./datasets/work"
    show_progress: bool = True
