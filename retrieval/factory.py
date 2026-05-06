from core.contracts import Retriever

from .bruteforce import BruteForceRetriever
from .faiss_index import FaissRetriever
from .modes import ANN_FULL, ANN_SUBSET, BRUTEFORCE_FULL, BRUTEFORCE_SUBSET


def build_retriever(mode: str) -> Retriever:
    if mode == BRUTEFORCE_FULL:
        return BruteForceRetriever(supports_subset=False)
    if mode == BRUTEFORCE_SUBSET:
        return BruteForceRetriever(supports_subset=True)
    if mode == ANN_FULL:
        return FaissRetriever(supports_subset=False)
    if mode == ANN_SUBSET:
        return FaissRetriever(supports_subset=True)
    raise ValueError(f"Unknown retrieval mode: {mode}")
