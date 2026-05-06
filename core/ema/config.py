from dataclasses import dataclass, field


@dataclass
class EMAConfig:
    embed_dim: int = 32
    batch_size: int = 250_000
    score_weights: dict[int, float] = field(
        default_factory=lambda: {
            0: -0.05,
            1: 1.0,
            2: 2.0,
            3: 3.0,
        }
    )
    ema_alpha: float = 0.95
    initial_capacity: int = 500_000
    max_capacity: int = 5_000_000
    gc_every: int = 4
    work_dir: str = "./datasets/work"
    show_progress: bool = True

