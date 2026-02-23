import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = "backend/safety_model.joblib"
VECTORIZER_PATH = "backend/vectorizer.joblib"

# ==========================
# TRAIN MODEL (runs once)
# ==========================
def train_safety_model():
    print("🛡️ Training safety filter...")

    # LOAD REDBENCH (unsafe)
    red_df = pd.read_parquet(
        "hf://datasets/knoveleng/redbench/AdvBench/train-00000-of-00001.parquet"
    )
    red_df = red_df[["prompt"]].dropna()
    red_df["label"] = 1  # unsafe

    # LOAD OPENASSISTANT (safe)
    oa_df = pd.read_parquet(
        "hf://datasets/OpenAssistant/oasst1/data/train-00000-of-00001-b42a775f407cee45.parquet"
    )
    oa_df = oa_df[oa_df["role"] == "prompter"]
    oa_df = oa_df[["text"]].dropna()
    oa_df.rename(columns={"text": "prompt"}, inplace=True)
    oa_df["label"] = 0  # safe

    # Balance dataset
    oa_sampled = oa_df.sample(n=len(red_df), random_state=42)

    df = pd.concat([red_df, oa_sampled])

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["prompt"])
    y = df["label"]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("✅ Safety model trained & saved")


# ==========================
# LOAD MODEL (startup)
# ==========================
def load_safety_model():
    if not os.path.exists(MODEL_PATH):
        train_safety_model()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# ==========================
# PREDICT FUNCTION
# ==========================
model, vectorizer = load_safety_model()

def is_prompt_safe(prompt: str):
    vec = vectorizer.transform([prompt])
    pred = model.predict(vec)[0]
    return pred == 0   # True = safe