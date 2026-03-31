import math
from src.retrieval.indexer import InvertedIndex

class TFIDFScorer:
    """Scores documents against a query using TF-IDF"""

    def __init__(self, index: InvertedIndex):
        self.index = index

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        score = 0.0
        for term in query_tokens:
            tf  = self.index.index.get(term, {}).get(doc_id, 0)
            if tf == 0:
                continue
            tf_w = 1 + math.log(tf)
            idf  = self.index.idf(term)
            score += tf_w * idf
        return score

    def rank(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Return top_k (doc_id, score) pairs (descending)"""
        candidates: dict[int, float] = {}
        for term in query_tokens:
            for doc_id in self.index.index.get(term, {}):
                candidates[doc_id] = candidates.get(doc_id, 0.0) + self.score([term], doc_id)
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
