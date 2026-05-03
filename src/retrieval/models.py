import math
from src.retrieval.indexer import InvertedIndex
from src.config import BM25_K1, BM25_B, QL_MU, QL_JM_LAMBDA

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

class BM25Scorer:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.k1 = BM25_K1
        self.b = BM25_B

    def rank(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        candidates: dict[int, float] = {}
        avg_dl = self.index.avg_dl
        
        for term in query_tokens:
            idf = self.index.idf(term)
            term_postings = self.index.index.get(term, {})
            
            for doc_id, tf in term_postings.items():
                doc_len = self.index.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_dl))
                candidates[doc_id] = candidates.get(doc_id, 0.0) + idf * (numerator / denominator)
                
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]


class QLDPScorer:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.mu = QL_MU

    def rank(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        candidates: dict[int, float] = {}
        
        # Optimization: Only calculate exact scores for documents that contain at least one query term
        candidate_docs = set()
        for term in query_tokens:
            candidate_docs.update(self.index.index.get(term, {}).keys())
            
        for doc_id in candidate_docs:
            score = 0.0
            doc_len = self.index.doc_lengths[doc_id]
            
            for term in query_tokens:
                tf = self.index.index.get(term, {}).get(doc_id, 0)
                cf = self.index.term_cf.get(term, 0)
                
                # If term is entirely OOV (Out of Vocabulary for the corpus), ignore it
                if cf == 0: continue 
                    
                p_wc = cf / self.index.collection_length
                # Dirichlet smoothing formula
                p_wd = (tf + self.mu * p_wc) / (doc_len + self.mu)
                score += math.log(p_wd)
                
            candidates[doc_id] = score
            
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]


class QLJMScorer:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.lmbda = QL_JM_LAMBDA

    def rank(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        candidates: dict[int, float] = {}

        candidate_docs = set()
        for term in query_tokens:
            candidate_docs.update(self.index.index.get(term, {}).keys())

        for doc_id in candidate_docs:
            score = 0.0
            doc_len = self.index.doc_lengths[doc_id]

            for term in query_tokens:
                tf = self.index.index.get(term, {}).get(doc_id, 0)
                cf = self.index.term_cf.get(term, 0)

                if cf == 0 or doc_len == 0:
                    continue

                p_ml = tf / doc_len
                p_wc = cf / self.index.collection_length
                p_wd = (1 - self.lmbda) * p_ml + self.lmbda * p_wc

                if p_wd > 0:
                    score += math.log(p_wd)

            candidates[doc_id] = score

        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
