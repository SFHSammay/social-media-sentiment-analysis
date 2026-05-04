import math
from collections import defaultdict

import pandas as pd


class InvertedIndex:
    # An inverted index for a collection of documents
    def __init__(self):
        self.index: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.doc_lengths: dict[int, int] = {}  # doc_id -> total token count
        self.doc_texts: dict[int, str] = {}  # doc_id -> original text
        self.doc_clean_texts: dict[int, str] = {}  # doc_id -> cleaned text
        self.doc_labels: dict[int, int] = {}  # doc_id -> sentiment label
        self.N = 0  # total documents
        self.collection_length = 0  # total tokens in corpus
        self.term_cf: dict[str, int] = defaultdict(
            int
        )  # term -> total corpus frequency

    # Build the inverted index from DataFrame of documents
    def build(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            doc_id = int(row["id"])
            tokens = str(row["clean_text"]).split()
            self.doc_texts[doc_id] = str(row["text"])
            self.doc_clean_texts[doc_id] = str(row["clean_text"])
            self.doc_labels[doc_id] = int(row["label"])
            self.doc_lengths[doc_id] = len(tokens)
            self.collection_length += len(tokens)
            for token in tokens:
                self.index[token][doc_id] += 1
                self.term_cf[token] += 1
        self.N = len(self.doc_lengths)
        print(f"Index built: {self.N} docs, {len(self.index)} unique terms")

    # Doc frequency of term
    def df(self, term: str) -> int:
        return len(self.index.get(term, {}))

    # Smoothed IDF
    def idf(self, term: str) -> float:
        return math.log((self.N + 1) / (self.df(term) + 1)) + 1

    @property
    def avg_dl(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
