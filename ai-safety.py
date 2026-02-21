import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# ==========================
# 1️⃣ LOAD REDBENCH (UNSAFE)
# ==========================

red_df = pd.read_parquet(
    "hf://datasets/knoveleng/redbench/AdvBench/train-00000-of-00001.parquet"
)

# Keep only required columns
red_df = red_df[["prompt", "category"]].dropna()

# Convert all RedBench categories to "unsafe"
red_df["binary_label"] = "unsafe"

print("RedBench samples:", len(red_df))


# ==========================
# 2️⃣ LOAD OPENASSISTANT (SAFE)
# ==========================

splits = {
    'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet'
}

oa_df = pd.read_parquet(
    "hf://datasets/OpenAssistant/oasst1/" + splits["train"]
)

# Keep only user prompts
oa_df = oa_df[oa_df["role"] == "prompter"]

oa_df = oa_df[["text"]].dropna()
oa_df.rename(columns={"text": "prompt"}, inplace=True)

oa_df["binary_label"] = "safe"

print("OpenAssistant safe samples:", len(oa_df))


# ==========================
# 3️⃣ BALANCE DATASET
# ==========================

unsafe_count = len(red_df)

oa_sampled = oa_df.sample(n=unsafe_count, random_state=42)

combined_df = pd.concat(
    [
        red_df[["prompt", "binary_label"]],
        oa_sampled[["prompt", "binary_label"]]
    ],
    ignore_index=True
)

print("\nBalanced Dataset Distribution:")
print(combined_df["binary_label"].value_counts())


# ==========================
# 4️⃣ ENCODE LABELS (Binary)
# ==========================

combined_df["binary_label"] = combined_df["binary_label"].astype("category")

label_mapping = dict(enumerate(combined_df["binary_label"].cat.categories))
combined_df["label_encoded"] = combined_df["binary_label"].cat.codes

print("\nLabel Mapping:")
for k, v in label_mapping.items():
    print(f"{k}: {v}")


# ==========================
# 5️⃣ TRAIN / TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    combined_df["prompt"],
    combined_df["label_encoded"],
    test_size=0.3,
    random_state=42,
    stratify=combined_df["label_encoded"]
)


# ==========================
# 6️⃣ TF-IDF VECTORIZATION
# ==========================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================
# 7️⃣ TRAIN MODEL
# ==========================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=0.5
)

model.fit(X_train_tfidf, y_train)


# ==========================
# 8️⃣ EVALUATION
# ==========================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================
# 9️⃣ INTERACTIVE MODE
# ==========================

def predict_prompt(prompt):
    vectorized = vectorizer.transform([prompt])
    pred_label = model.predict(vectorized)[0]
    pred_category = label_mapping[pred_label]

    return pred_category.upper()


print("\n🔎 INTERACTIVE MODE")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a prompt: ")

    if user_input.lower() == "exit":
        break

    prediction = predict_prompt(user_input)

    print("\nPrediction:", prediction)
    print("-" * 40)