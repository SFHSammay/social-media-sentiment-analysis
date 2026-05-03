import os

# Paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")

TRAIN_FILE      = os.path.join(RAW_DIR, "twitter_training.csv")
VAL_FILE        = os.path.join(RAW_DIR, "twitter_validation.csv")

PROCESSED_TRAIN = os.path.join(PROCESSED_DIR, "train_clean.csv")
PROCESSED_VAL   = os.path.join(PROCESSED_DIR, "val_clean.csv")

REPORT_FILE     = os.path.join(BASE_DIR, "report.csv")

# Label Mapping
LABEL_MAP = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}
LABELS = [0, 1, 2]

# IR Settings
TOP_K = 10 # Precision@K

# Word2Vec Settings
W2V_VECTOR_SIZE = 100
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 2
W2V_EPOCHS      = 10
W2V_WORKERS     = 4

# Classifier Settings 
LR_MAX_ITER     = 1000
LR_C            = 1.0
TFIDF_MAX_FEAT  = 10000
TFIDF_NGRAM     = (1, 2)

# Queries for Testing (Milestone)
QUERIES = [
    "amazon delivery",
    "facebook privacy",
    "microsoft word",
    "windows update",
    "xbox game pass",
    "csgo skins",
    "league of legends",
    "genshin impact",
    "epic games refund",
    "google search",
    "nvidia graphics card",
    "apple iphone battery",
    "netflix recommendations"
]

# BM25 Settings
BM25_K1 = 1.2
BM25_B  = 0.75

# Query Likelihood Settings (Dirichlet Smoothing)
QL_MU = 1000

# LLM API Settings
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", None)
GEMMA_MODEL = "gemma-4-31b-it"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
MAX_RETRIES = 6

