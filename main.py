import csv

import src.retrieval.indexer as indexer
from src.config import REPORT_FILE, TOP_K
from src.evaluate import average_precision, evaluate_sentiment, precision_at_k
from src.preprocess import clean_text, get_train_df, get_val_df
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

    run = 0
    model_ap = {"tfidf": [], "bm25": [], "ql_dp": [], "ql_jm": []}

    rows = []

    # Run search engine for each query (3 times)
    while run < 3:
        run += 1
        # User Query
        user_query = input("\nEnter a search query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            return
        if user_query.lower() != "exit":
            pooled_results = []
            seen_doc_ids = set()
            search_results_by_model = {}

            # Run search for each IR model
            for model in ir_models:
                engine = SearchEngine(index, model=model)
                retrieved_results = engine.search(user_query, top_k=TOP_K)
                search_results_by_model[model] = retrieved_results

                model_results = []
                newly_added_docs = []

                # print(f"\nRetrieved Results: ({model.upper()})")
                for i, result in enumerate(retrieved_results, start=1):
                    model_results.append((result["doc_id"], result["score"]))

                    # Add result to pooled results
                    if result["doc_id"] not in seen_doc_ids:
                        pooled_results.append(result)
                        seen_doc_ids.add(result["doc_id"])
                        newly_added_docs.append((result["doc_id"], result["score"]))

                # Print results
                print(f"\nRetrieved Results: ({model.upper()})")
                print(model_results)
                print(f"New pooled from {model.upper()}:")
                print(newly_added_docs)

            # Process pooled results
            pooled_texts = [clean_text(result["text"]) for result in pooled_results]
            pooled_true_labels = [result["label"] for result in pooled_results]

            # Prepare data for sentiment analysis
            query_results_list = [
                {
                    "query": user_query,
                    "results": pooled_results,
                }
            ]

            # Label the pooled results with ground truth labels using LLMs
            print("Preparing ground truth labels for retrieved results using LLMs")
            ground_truth = label_all(query_results_list)
            relevant_doc_ids = {doc_id for doc_id, label in ground_truth if label == 1}

            print("Ground Truth Labels:")
            print(ground_truth)

        all_ir_results = {}

        print(f"\nRunning IR evaluation (P@{TOP_K}, AP)...")
        # Evaluate each IR model
        for model_name, retrieved_results in search_results_by_model.items():
            retrieved_doc_ids = [result["doc_id"] for result in retrieved_results]

            p_at_k = precision_at_k(retrieved_doc_ids, relevant_doc_ids, TOP_K)
            ap = average_precision(retrieved_doc_ids, relevant_doc_ids)

            model_ap[model_name].append(ap)

            all_ir_results[model_name] = {
                "P@K": round(p_at_k, 4),
                "AP": round(ap, 4),
                "MAP": round(sum(model_ap[model_name]) / len(model_ap[model_name]), 4),
                "retrieved": retrieved_results,
            }

        # Sentiment trainining on training set and evaluation on validation set
        featurizers = {"TFIDF": TFIDFFeaturizer(), "Word2Vec": Word2VecFeaturizer()}

        all_sent_results = {}
        all_sent_predictions = {}

        train_texts = train_df["clean_text"].tolist()
        train_labels = train_df["label"].tolist()

        print("\nTraining sentiment classifiers...")
        # Trains sentiment classifiers on training set
        for feat_name, featurizer in featurizers.items():
            print(f"  -> Extracting features using {feat_name}...")
            X_train = featurizer.fit_transform(train_texts)

            clf = SentimentClassifier()
            clf.fit(X_train, train_labels)

            x_pool = featurizer.transform(pooled_texts)
            y_pool_pred = clf.predict(x_pool)
            all_sent_predictions[feat_name] = y_pool_pred

            sent_results = evaluate_sentiment(pooled_true_labels, y_pool_pred)
            all_sent_results[feat_name] = sent_results

        # Write Report
        print(f"\nWriting report → {REPORT_FILE}")

        # Model comparsion section
        rows.append(["MODEL COMPARISON", "", "", ""])
        rows.append(["Model", "MAP", f"P@{TOP_K}", "AP", "Queries Used"])
        for model_name, results in all_ir_results.items():
            rows.append(
                [
                    model_name.upper(),
                    results["MAP"],
                    results["P@K"],
                    results["AP"],
                    len(model_ap[model_name]),
                ]
            )
        rows.append(["", "", "", ""])

        # Sentiment prediction by featurizer
        for feat_name, y_pred in all_sent_predictions.items():
            rows.append([f"POOLED SENTIMENT PREDICTIONS ({feat_name})", "", "", ""])
            rows.append(["doc_id", "true_label", "predicted_label", "text", ""])
            for result, pred in zip(pooled_results, y_pred):
                rows.append(
                    [result["doc_id"], result["label"], pred, result["text"], ""]
                )
            rows.append(["", "", "", ""])

        rows.append(["POOLED SENTIMENT METRICS", "", "", ""])
        rows.append(["Model", "Accuracy", "F1", ""])

        # Display sentiment metrics for each featurizer
        for feat_name, results in all_sent_results.items():
            rows.append([feat_name, results["accuracy"], results["f1"], ""])

        rows.append(["", "", "", ""])

        # Write report to our CSV file
        with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        # Print final results
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
