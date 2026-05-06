from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IndexHandle:
    index_type: str
    index_path: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class TopKResult:
    item_ids: np.ndarray
    scores: np.ndarray


class Retriever(ABC):
    """Unified retrieval API for brute-force and ANN implementations."""

    @abstractmethod
    def build_index(
        self,
        item_ids: np.ndarray,
        item_embeddings: np.ndarray,
        mode_config: Optional[dict] = None,
    ) -> IndexHandle:
        pass

    @abstractmethod
    def top_k(
        self,
        user_embedding: np.ndarray,
        k: int,
        candidate_subset: Optional[np.ndarray] = None,
    ) -> TopKResult:
        pass

    @abstractmethod
    def supports_mode(self, mode: str) -> bool:
        pass

