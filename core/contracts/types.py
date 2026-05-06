from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class DatasetPaths:
    users_path: str
    items_path: str
    interactions_path: str
    item_embeddings_npz: Optional[str] = None
    interaction_paths: Optional[List[str]] = None


@dataclass
class RetrievalRequest:
    mode: str = "bruteforce_full"
    ann_enabled: bool = False
    subset_size: Optional[int] = None
    random_seed: int = 42
    index_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingArtifactRef:
    artifact_path: str
    n_vectors: int
    dim: int
    dtype: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationBatch:
    user_ids: np.ndarray
    item_ids: np.ndarray
    scores: np.ndarray


@dataclass
class RecommendationResult:
    user_id: int
    item_ids: np.ndarray
    scores: np.ndarray


@dataclass
class BatchRecommendationResult:
    results: List[RecommendationResult]


@dataclass
class FitSummary:
    n_users: int
    n_items: int
    extras: Optional[Dict[str, float]] = None


@dataclass
class EvalArtifacts:
    metrics_path: str
    run_id: str
    core_name: str
    metrics: Mapping[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreConfig:
    core_name: str
    params: Dict[str, Any]


JSONDict = Dict[str, Any]
UserIdSequence = Sequence[int]

