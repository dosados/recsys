from typing import Any

from core.contracts import RecSysCore
from core.baselines import PopularCore, RandomCore
from core.ema.core import EMACore
from core.gcn import GCNConfig, GCNCore
from core.online_kmeans import OnlineKMeansCore
from core.two_tower import TwoTowerCore


def build_core(core_name: str, **kwargs: Any) -> RecSysCore:
    if core_name == "ema":
        return EMACore(**kwargs)
    if core_name == "random":
        return RandomCore(**kwargs)
    if core_name == "popular":
        return PopularCore(**kwargs)
    if core_name == "gcn":
        config = kwargs.pop("config", None)
        if isinstance(config, dict):
            config = GCNConfig(**config)
        return GCNCore(config=config)
    if core_name == "online_kmeans":
        return OnlineKMeansCore(**kwargs)
    if core_name == "two_tower":
        return TwoTowerCore(**kwargs)
    raise ValueError(f"Unknown core '{core_name}'")

