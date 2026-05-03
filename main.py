import csv
from src.config import REPORT_FILE, TOP_K, QUERIES
from src.preprocess import get_train_df, get_val_df, clean_text
import src.retrieval.indexer as indexer
from src.retrieval.search import SearchEngine
from src.sentiment.vectorizer import TFIDFFeaturizer, Word2VecFeaturizer
from src.sentiment.classifier import SentimentClassifier
from src.evaluate import evaluate_ir, evaluate_sentiment, summarize_ir
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

    ir_models = ["tfidf", "bm25", "ql_dp", "ql_jm"]

    # User Query
    user_query = input("\nEnter a search query (or 'exit' to quit): ")
    if user_query.lower() == "exit":
        return
    if user_query.lower() != "exit":
        pooled_results = []
        seen_doc_ids = set()

        for model in ir_models:
            engine = SearchEngine(index, model=model)
            retrieved_results = engine.search(user_query, top_k=TOP_K)

            model_results = []
            newly_added_docs = []

            # print(f"\nRetrieved Results: ({model.upper()})")
            for i, result in enumerate(retrieved_results, start=1):
                """print(f"{i}. doc_id={result['doc_id']} score={result['score']}")
                print(result["text"])
                print()"""
                model_results.append((result["doc_id"], result["score"]))

                if result["doc_id"] not in seen_doc_ids:
                    pooled_results.append(result)
                    seen_doc_ids.add(result["doc_id"])
                    newly_added_docs.append((result["doc_id"], result["score"]))

            print(f"\nRetrieved Results: ({model.upper()})")
            print(model_results)
            print(f"New pooled from {model.upper()}:")
            print(newly_added_docs)

        pooled_texts = [clean_text(result["text"]) for result in pooled_results]
        pooled_true_labels = [result["label"] for result in pooled_results]


        query_results_list = [
            {
                "query": user_query,
                "results": pooled_results,
            }
        ]
        print("Preparing ground truth labels for retrieved results using LLMs")
        ground_truth = label_all(query_results_list)

        print("Ground Truth Labels:")
        print(ground_truth)
      
    # Use a fixed list of  queries for evaluation (for milestone only)
    val_index = indexer.InvertedIndex()
    val_index.build(val_df)

    # IR Evaluation on Validation Set with multiple models
    ir_models = ["tfidf", "bm25", "ql_dp", "ql_jm"]
    all_ir_results = {}

    print(f"\nRunning IR evaluation (P@{TOP_K}, MAP)...")
    for model in ir_models:
        engine = SearchEngine(val_index, model=model)
        results = evaluate_ir(val_index, engine.search, QUERIES)
        all_ir_results[model] = results

    # Sentiment: Train on training set, evaluate on validation set 
    featurizers = {
        "TF-IDF": TFIDFFeaturizer(),
        "Word2Vec": Word2VecFeaturizer()
    }
    
    all_sent_results = {}
    all_sent_predictions = {}

    train_texts = train_df["clean_text"].tolist()
    # val_texts   = val_df["clean_text"].tolist()
    train_labels = train_df["label"].tolist()
    # val_labels  = val_df["label"].tolist()


    print("\nTraining sentiment classifiers...")
    for feat_name, featurizer in featurizers.items():
        print(f"  -> Extracting features using {feat_name}...")
        X_train = featurizer.fit_transform(train_texts)
        # X_val   = featurizer.transform(val_texts)

        clf = SentimentClassifier()
        clf.fit(X_train, train_labels)

        x_pool = featurizer.transform(pooled_texts)
        y_pool_pred = clf.predict(x_pool)
        all_sent_predictions[feat_name] = y_pool_pred

        sent_results = evaluate_sentiment(pooled_true_labels, y_pool_pred)
        all_sent_results[feat_name] = sent_results
        # y_pred = clf.predict(X_val)
        # sent_results = evaluate_sentiment(val_labels, y_pred)
        # all_sent_results[feat_name] = sent_results
        #all_sent_predictions[feat_name] = y_pred

    # Write Report
    print(f"\nWriting report → {REPORT_FILE}")
    rows = []


    
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


    for feat_name, y_pred in all_sent_predictions.items():
        rows.append([f"POOLED SENTIMENT PREDICTIONS ({feat_name})", "", "", ""])
        rows.append(["doc_id", "true_label", "predicted_label", "text", ""])
        for result, pred in zip(pooled_results, y_pred):
            rows.append([result["doc_id"], result["label"], pred, result["text"], ""])
        rows.append(["", "", "", ""])

    rows.append(["POOLED SENTIMENT METRICS", "", "", ""])
    rows.append(["Model", "Accuracy", "F1", ""])

    for feat_name, results in all_sent_results.items():
        rows.append([feat_name, results["accuracy"], results["f1"], ""])

    rows.append(["", "", "", ""])

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


    print("Done.")
    print("-------------------------------")
    for model_name, results in all_ir_results.items():
        print(f"{model_name.upper()} MAP : {results['MAP']}")
    for feat_name, results in all_sent_results.items():
        print(f"{feat_name} Accuracy : {results['accuracy']}")
        print(f"{feat_name} F1 : {results['f1']}")

    print("-------------------------------")


if __name__ == "__main__":
    main()
