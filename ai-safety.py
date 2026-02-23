import os
import random
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack

random.seed(42)
np.random.seed(42)

print("Loading Red-AVA dataset...")
red_df = pd.read_parquet(
    "hf://datasets/knoveleng/redbench/AdvBench/train-00000-of-00001.parquet"
)
red_df = red_df[["prompt", "category"]].dropna()
red_df.rename(columns={"category": "label"}, inplace=True)

counts = red_df["label"].value_counts()
rare   = counts[counts < 2].index
red_df = red_df[~red_df["label"].isin(rare)]
print(f"   Unsafe samples (raw): {len(red_df)}")
print(red_df["label"].value_counts().to_string())

TARGET_PER_CLASS = 300

def aug_drop_words(text, rate=0.12):
    words = text.split()
    result = " ".join(w for w in words if random.random() > rate)
    return result if result else text

def aug_swap_words(text):
    words = text.split()
    if len(words) < 4:
        return text
    i = random.randint(0, len(words) - 2)
    words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)

def aug_duplicate_phrase(text):
    words = text.split()
    if len(words) < 6:
        return text
    mid   = len(words) // 2
    chunk = words[mid:mid+3]
    return " ".join(words[:mid] + chunk + words[mid:])

def aug_prefix(text):
    prefixes = [
        "Tell me ", "Explain how to ", "I want to know how to ",
        "Give me instructions to ", "Describe how to ", "Help me ",
        "Walk me through how to ", "Show me how to ",
    ]
    return random.choice(prefixes) + text.lower()

def aug_suffix(text):
    suffixes = [
        " step by step.", " in detail.", " without getting caught.",
        " quickly.", " as an expert.", " for educational purposes.",
        " in simple terms.", " thoroughly.",
    ]
    return text.rstrip(".") + random.choice(suffixes)

AUGMENTERS = [aug_drop_words, aug_swap_words, aug_duplicate_phrase, aug_prefix, aug_suffix]

print(f"\n Augmenting each unsafe class to {TARGET_PER_CLASS} samples...")
augmented_rows = []
for label, group in red_df.groupby("label"):
    needed  = max(0, TARGET_PER_CLASS - len(group))
    prompts = group["prompt"].tolist()
    for _ in range(needed):
        base    = random.choice(prompts)
        new_txt = random.choice(AUGMENTERS)(base)
        augmented_rows.append({"prompt": new_txt, "label": label})

if augmented_rows:
    red_df = pd.concat([red_df, pd.DataFrame(augmented_rows)], ignore_index=True)

print(f"   Added {len(augmented_rows)} augmented samples")
print(f"   Total unsafe samples: {len(red_df)}")

print("\nLoading OpenAssistant dataset...")
oa_df = pd.read_parquet(
    "hf://datasets/OpenAssistant/oasst1/data/train-00000-of-00001-b42a775f407cee45.parquet"
)
oa_df = oa_df[oa_df["role"] == "prompter"][["text"]].dropna()
oa_df.rename(columns={"text": "prompt"}, inplace=True)
oa_df["label"] = "safe"

n_safe     = min(int(len(red_df) * 1.5), len(oa_df))
oa_sampled = oa_df.sample(n=n_safe, random_state=42)
print(f"   Safe samples: {n_safe}")

df = pd.concat(
    [red_df[["prompt", "label"]], oa_sampled[["prompt", "label"]]],
    ignore_index=True
).sample(frac=1, random_state=42)

print(f"\nFinal dataset: {len(df)} samples")
print(df["label"].value_counts().to_string())

df["label"]   = df["label"].astype("category")
label_mapping = dict(enumerate(df["label"].cat.categories))
df["y"]       = df["label"].cat.codes

print("\nLabel Mapping:")
for k, v in label_mapping.items():
    print(f"   {k}: {v}")

X_train, X_test, y_train, y_test = train_test_split(
    df["prompt"], df["y"],
    test_size=0.15,
    stratify=df["y"],
    random_state=42
)
print(f"\n   Train: {len(X_train)}  |  Test: {len(X_test)}")

print("\nFitting WORD TF-IDF...")
word_vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 3),
    stop_words="english",
    sublinear_tf=True,
    min_df=1
)

print("Fitting CHAR TF-IDF...")
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=50_000,
    sublinear_tf=True,
    min_df=1
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word  = word_vectorizer.transform(X_test)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char  = char_vectorizer.transform(X_test)

X_train_final = hstack([X_train_word, X_train_char])
X_test_final  = hstack([X_test_word,  X_test_char])

print("\nTraining LinearSVC...")
model = LinearSVC(C=1.2, max_iter=5000, class_weight="balanced")
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nFINAL MODEL METRICS")
print(f"   Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"   Precision : {precision:.4f}")
print(f"   Recall    : {recall:.4f}")
print(f"   F1-score  : {f1:.4f}")

os.makedirs("backend", exist_ok=True)

joblib.dump(model,           "backend/safety_model.joblib")
joblib.dump(word_vectorizer, "backend/vectorizer.joblib")
joblib.dump(char_vectorizer, "backend/char_vectorizer.joblib")
joblib.dump(label_mapping,   "backend/label_mapping.joblib")

print("\nSaved:")
print("   backend/safety_model.joblib")
print("   backend/vectorizer.joblib")
print("   backend/char_vectorizer.joblib")
print("   backend/label_mapping.joblib")
print("\nTraining complete. Run main.py to start the server.")