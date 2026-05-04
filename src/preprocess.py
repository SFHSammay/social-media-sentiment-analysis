import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import (
    LABEL_MAP,
    PROCESSED_DIR,
    PROCESSED_TRAIN,
    PROCESSED_VAL,
    TRAIN_FILE,
    VAL_FILE,
)

nltk.download("stopwords", quiet=True)

_stemmer = PorterStemmer()
_stopwords = set(stopwords.words("english"))


# Removes URLs, metions, and nonaplha chars
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # keep hashtag word
    text = re.sub(r"[^a-z\s]", " ", text)  # non-alpha
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stopwords and len(t) > 1]
    return " ".join(tokens)


def map_label(label: str) -> int:
    return LABEL_MAP[label]


# Load and preprocess raw data
def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None, names=["id", "entity", "sentiment", "text"])
    df.dropna(subset=["text", "sentiment"], inplace=True)
    df = df[df["sentiment"] != "Irrelevant"]
    df["label"] = df["sentiment"].apply(map_label)
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["clean_text"].str.strip() != ""]
    return df


# Loads or builds data
def load_or_build(raw_path: str, processed_path: str) -> pd.DataFrame:
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        df["label"] = df["label"].astype(int)
        return df
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_raw(raw_path)
    df.to_csv(processed_path, index=False)
    print(f"  Saved processed data → {processed_path}")
    return df


def get_train_df() -> pd.DataFrame:
    return load_or_build(TRAIN_FILE, PROCESSED_TRAIN)


def get_val_df() -> pd.DataFrame:
    return load_or_build(VAL_FILE, PROCESSED_VAL)
