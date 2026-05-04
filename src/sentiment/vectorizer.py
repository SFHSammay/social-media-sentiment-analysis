import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_FEAT,
    TFIDF_NGRAM,
    W2V_EPOCHS,
    W2V_MIN_COUNT,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_WORKERS,
)


# Featurizers for converting text to numerical representations
class TFIDFFeaturizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEAT,
            ngram_range=TFIDF_NGRAM,
            sublinear_tf=True,
        )

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)


# Featurizer for converting text using Word2Vec
class Word2VecFeaturizer:
    def __init__(self):
        self.model = None
        self.vector_size = W2V_VECTOR_SIZE

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return [t.split() for t in texts]

    def fit(self, texts: list[str]):
        sentences = self._tokenize(texts)
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=W2V_WINDOW,
            min_count=W2V_MIN_COUNT,
            workers=W2V_WORKERS,
            epochs=W2V_EPOCHS,
        )
        print(f"Word2Vec trained: vocab size = {len(self.model.wv)}")

    def _mean_vector(self, tokens: list[str]) -> np.ndarray:
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vecs:
            return np.zeros(self.vector_size)
        return np.mean(vecs, axis=0)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.array([self._mean_vector(t.split()) for t in texts])

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
