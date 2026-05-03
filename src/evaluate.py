from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import LABELS, TOP_K
from src.preprocess import clean_text
from src.retrieval.indexer import InvertedIndex


def precision_at_k(
    retrieved_doc_ids: list[int], relevant_doc_ids: set[int], k: int = TOP_K
) -> float:
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_doc_ids)
    return hits / k if k > 0 else 0.0


def average_precision(
    retrieved_doc_ids: list[int], relevant_doc_ids: set[int]
) -> float:
    hits, score = 0, 0.0
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            score += hits / i
    return score / len(relevant_doc_ids) if relevant_doc_ids else 0.0


def mean_average_precision(
    results_per_query: dict[str, list[int]], relevant_per_query: dict[str, set[int]]
) -> float:
    aps = []
    for query, retrieved in results_per_query.items():
        relevant = relevant_per_query.get(query, set())
        aps.append(average_precision(retrieved, relevant))
    return sum(aps) / len(aps) if aps else 0.0


def evaluate_ir(index: InvertedIndex, search_fn, queries: list[str]) -> dict:
    results_per_query = {}
    relevant_per_query = {}
    retrieved_per_query = {}

    for query in queries:
        cleaned_query = clean_text(query)
        query_terms = cleaned_query.split()
        relevant_docs = set()
        for doc_id, text in index.doc_clean_texts.items():
            doc_terms = text.split()
            if all(term in doc_terms for term in query_terms):
                relevant_docs.add(doc_id)

        if not relevant_docs:
            continue

        retrieved_results = search_fn(query, top_k=len(index.doc_lengths))
        retrieved_ids = [result["doc_id"] for result in retrieved_results]
        results_per_query[query] = retrieved_ids
        relevant_per_query[query] = relevant_docs
        retrieved_per_query[query] = retrieved_results[:TOP_K]

    map_score = mean_average_precision(results_per_query, relevant_per_query)

    per_query = []
    for query in results_per_query:
        retrieved_ids = results_per_query[query]
        relevant_docs = relevant_per_query[query]

        per_query.append(
            {
                "query": query,
                "P@K": round(precision_at_k(retrieved_ids, relevant_docs, TOP_K), 4),
                "AP": round(average_precision(retrieved_ids, relevant_docs), 4),
                "relevant_docs": len(relevant_docs),
            }
        )

    return {
        "MAP": round(map_score, 4),
        "per_query": per_query,
        "retrieved": retrieved_per_query,
    }


def evaluate_sentiment(y_true: list[int], y_pred: list[int]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "report": report}

def summarize_ir(results: dict) -> dict:
    per_query = results["per_query"]
    if not per_query:
        return {"avg_p": 0.0, "avg_ap": 0.0, "count": 0}
    avg_p = sum(q["P@K"] for q in per_query) / len(per_query)
    avg_ap = sum(q["AP"] for q in per_query) / len(per_query)
    return {
        "avg_p": round(avg_p, 4),
        "avg_ap": round(avg_ap, 4),
        "count": len(per_query),
    }
