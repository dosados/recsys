from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from .types import BatchRecommendationResult, DatasetPaths, EmbeddingArtifactRef, FitSummary, RetrievalRequest


class RecSysCore(ABC):
    """Draft contract for interchangeable recommendation cores."""

    @abstractmethod
    def fit_offline(self, dataset: DatasetPaths, config: Optional[dict[str, Any]] = None) -> FitSummary:
        pass

    @abstractmethod
    def update_incremental(
        self, dataset: DatasetPaths, config: Optional[dict[str, Any]] = None
    ) -> FitSummary:
        pass

    @abstractmethod
    def recommend(
        self,
        user_ids: Sequence[int],
        k: int,
        retrieval_request: Optional[RetrievalRequest] = None,
    ) -> BatchRecommendationResult:
        pass

    @abstractmethod
    def save(self, output_dir: str) -> None:
        pass

    @abstractmethod
    def load(self, output_dir: str) -> None:
        pass

    @abstractmethod
    def export_user_embeddings(self, output_path: str) -> EmbeddingArtifactRef:
        pass

