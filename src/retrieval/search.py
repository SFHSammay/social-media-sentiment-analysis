from src.config import TOP_K
from src.preprocess import clean_text
from src.retrieval.indexer import InvertedIndex
from src.retrieval.models import BM25Scorer, QLDPScorer, QLJMScorer, TFIDFScorer


class SearchEngine:
    # Search engine that uses specific scoring models
    def __init__(self, index: InvertedIndex, model: str = "tfidf"):
        self.index = index
        self.model = model.lower()

        if self.model == "bm25":
            self.scorer = BM25Scorer(index)
        elif self.model == "ql_dp":
            self.scorer = QLDPScorer(index)
        elif self.model == "ql_jm":
            self.scorer = QLJMScorer(index)
        else:
            self.scorer = TFIDFScorer(index)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        # Returns a list of result dicts with doc_id, score, text, label
        tokens = clean_text(query).split()
        if not tokens:
            return []
        ranked = self.scorer.rank(tokens, top_k=top_k)
        results = []
        for doc_id, score in ranked:
            results.append(
                {
                    "doc_id": doc_id,
                    "score": round(score, 4),
                    "text": self.index.doc_texts.get(doc_id, ""),
                    "label": self.index.doc_labels.get(doc_id, ""),
                }
            )
        return results
