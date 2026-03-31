import re
import os
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from src.config import LABEL_MAP, TRAIN_FILE, VAL_FILE, PROCESSED_TRAIN, PROCESSED_VAL, PROCESSED_DIR

nltk.download("stopwords", quiet=True)

_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/hashtag, remove non letter and stopwords, stem."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # URLs
    text = re.sub(r"@\w+", "", text)                     # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # keep hashtag word
    text = re.sub(r"[^a-z\s]", " ", text)                # non-alpha
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stopwords and len(t) > 1]
    return " ".join(tokens)

def map_label(label: str) -> int:
    return LABEL_MAP[label]

def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None, names=["id", "entity", "sentiment", "text"])
    df.dropna(subset=["text", "sentiment"], inplace=True)
    df = df[df["sentiment"] != "Irrelevant"]
    df["label"] = df["sentiment"].apply(map_label)
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["clean_text"].str.strip() != ""]
    return df

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
