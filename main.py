import csv

import src.retrieval.indexer as indexer
from src.config import QUERIES, REPORT_FILE, TOP_K
from src.evaluate import evaluate_ir, evaluate_sentiment, summarize_ir
from src.preprocess import get_train_df, get_val_df
from src.retrieval.background import label_all
from src.retrieval.search import SearchEngine
from src.sentiment.classifier import SentimentClassifier
from src.sentiment.vectorizer import TFIDFFeaturizer, Word2VecFeaturizer


def main():
    # Load Data
    print("\nLoading and preprocessing data...")
    train_df = get_train_df()
    val_df = get_val_df()
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    # Build Inverted Index on Training Set
    print("\nBuilding inverted index...")
    index = indexer.InvertedIndex()
    index.build(train_df)
    engine = SearchEngine(index, model="bm25")

    # User Query
    user_query = input("\nEnter a search query (or 'exit' to quit): ")
    if user_query.lower() != "exit":
        retrieved_results = engine.search(user_query, top_k=TOP_K)

        print("\nRetrieved Results:")
        for i, result in enumerate(retrieved_results, start=1):
            print(f"{i}. doc_id={result['doc_id']} score={result['score']}")
            print(result["text"])
            print()

        query_results_list = [
            {
                "query": user_query,
                "results": retrieved_results,
            }
        ]
        print("Preparing ground truth labels for retrieved results using LLMs")
        ground_truth = label_all(query_results_list)

        print("Ground Truth Labels:")
        print(ground_truth)

    # IR Evaluation on Validation Set
    print(f"\nRunning IR evaluation (P@{TOP_K}, MAP)...")
    # Use a fixed list of  queries for evaluation (for milestone only)
    val_index = indexer.InvertedIndex()
    val_index.build(val_df)
    val_engine = SearchEngine(val_index, model="bm25")
    queries = QUERIES
    ir_results = evaluate_ir(val_index, val_engine.search, queries)
    print(f"MAP = {ir_results['MAP']}")
    for row in ir_results["per_query"][:]:
        print(f"{row['query']:35s} P@{TOP_K}={row['P@K']:.4f}  AP={row['AP']:.4f}")

    ir_models = ["tfidf", "bm25", "ql"]
    all_ir_results = {}

    print(f"\nRunning IR evaluation (P@{TOP_K}, MAP)...")
    for model_name in ir_models:
        # We pass the dynamic model name into the engine
        engine = SearchEngine(val_index, model=model_name)
        results = evaluate_ir(val_index, engine.search, QUERIES)
        all_ir_results[model_name] = results
        print(f"Model: {model_name.upper():6s} | MAP: {results['MAP']:.4f}")

    # Sentiment: Train on training set, evaluate on validation set
    print("\nTraining sentiment classifier (TF-IDF + Logistic Regression)...")
    featurizer = TFIDFFeaturizer()
    X_train = featurizer.fit_transform(train_df["clean_text"].tolist())
    X_val = featurizer.transform(val_df["clean_text"].tolist())

    clf = SentimentClassifier()
    clf.fit(X_train, train_df["label"].tolist())

    y_pred = clf.predict(X_val)
    y_true = val_df["label"].tolist()
    sent_results = evaluate_sentiment(y_true, y_pred)

    print(f"Accuracy  = {sent_results['accuracy']}")
    print(f"F1  = {sent_results['f1']}")
    print("\nClassification Report:")
    print(sent_results["report"])

    featurizers = {"TF-IDF": TFIDFFeaturizer(), "Word2Vec": Word2VecFeaturizer()}

    all_sent_results = {}
    train_texts = train_df["clean_text"].tolist()
    val_texts = val_df["clean_text"].tolist()
    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()

    print("\nTraining sentiment classifiers...")
    for feat_name, featurizer in featurizers.items():
        print(f"  -> Extracting features using {feat_name}...")
        X_train = featurizer.fit_transform(train_texts)
        X_val = featurizer.transform(val_texts)

        clf = SentimentClassifier()
        clf.fit(X_train, train_labels)

        y_pred = clf.predict(X_val)
        sent_results = evaluate_sentiment(val_labels, y_pred)
        all_sent_results[feat_name] = sent_results

    # Write Report
    print(f"\nWriting report → {REPORT_FILE}")
    rows = []

    # IR section
    rows.append(["IR EVALUATION (TF-IDF)", "", "", ""])
    rows.append(["Queries", "P@K", "AP", "Relevant Docs"])
    for r in ir_results["per_query"]:
        rows.append([r["query"], r["P@K"], r["AP"], r["relevant_docs"]])
    rows.append(["OVERALL MAP", ir_results["MAP"], "", ""])
    rows.append(["", "", "", ""])
    rows.append(["=== TOP RETRIEVED RESULTS PER QUERY ===", "", "", ""])
    rows.append(["query", "doc_id", "score", "clean_text"])
    for query, results in ir_results["retrieved"].items():
        for result in results:
            rows.append(
                [
                    query,
                    result["doc_id"],
                    result["score"],
                    val_index.doc_clean_texts.get(result["doc_id"], ""),
                ]
            )
        rows.append(["", "", "", ""])

    # Model comparsion section
    rows.append(["MODEL COMPARISON", "", "", ""])
    rows.append(["Model", "MAP", "Avg P@K", "Avg AP", "Queries Used"])
    for model_name, results in all_ir_results.items():
        summary = summarize_ir(results)
        rows.append(
            [
                model_name.upper(),
                results["MAP"],
                summary["avg_p"],
                summary["avg_ap"],
                summary["count"],
            ]
        )
    rows.append(["", "", "", ""])

    # Sentiment section
    rows.append(["SENTIMENT EVALUATION (TF-IDF + Logistic Regression)", "", "", ""])
    rows.append(["Metric", "Value", "", ""])
    rows.append(["Accuracy", sent_results["accuracy"], "", ""])
    rows.append(["F1", sent_results["f1"], "", ""])
    rows.append(["", "", "", ""])

    # Per sample predictions on validation
    rows.append(["PER SAMPLE PREDICTIONS (Validation)", "", "", ""])
    rows.append(["tweet_id", "true_label", "predicted_label", "clean_text"])
    for i, (_, vrow) in enumerate(val_df.iterrows()):
        if i >= 100:
            break
        rows.append([vrow["id"], y_true[i], y_pred[i], vrow["clean_text"]])

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("Done.")
    print("-------------------------------")
    print(f"MAP : {ir_results['MAP']}")
    print(f"Accuracy : {sent_results['accuracy']}")
    print(f"F1 : {sent_results['f1']}")
    print("-------------------------------")


if __name__ == "__main__":
    main()
