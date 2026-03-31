from sklearn.linear_model import LogisticRegression
from src.config import LR_MAX_ITER, LR_C
import numpy as np

class SentimentClassifier:
    def __init__(self):
        self.model   = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, class_weight="balanced")

    def fit(self, X, y: list[int]):
        self.model.fit(X, y)
        print("Classifier trained.")

    def predict(self, X) -> list[int]:
        return self.model.predict(X).tolist()

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
