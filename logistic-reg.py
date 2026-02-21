import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    r2_score,
)

dataset = load_dataset("knoveleng/redbench", "AdvBench", split="train")

df = dataset.to_pandas()

TEXT_COLUMN = "prompt"
LABEL_COLUMN = "category"

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()


df[LABEL_COLUMN] = df[LABEL_COLUMN].astype("category")
label_mapping = dict(enumerate(df[LABEL_COLUMN].cat.categories))
df["label_encoded"] = df[LABEL_COLUMN].cat.codes

print("\nLabel Mapping:")
for k, v in label_mapping.items():
    print(f"{k}: {v}")

class_counts = df["label_encoded"].value_counts()
valid_classes = class_counts[class_counts >= 2].index

df = df[df["label_encoded"].isin(valid_classes)]

print("Remaining classes:", len(valid_classes))

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN],
    df["label_encoded"],
    test_size=0.3,
    random_state=42,
    stratify=df["label_encoded"]
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=0.5
)


model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

def predict_prompt(prompt):
    vectorized = vectorizer.transform([prompt])
    pred_label = model.predict(vectorized)[0]
    pred_category = label_mapping[pred_label]

    # Define safe vs unsafe rule (customize if needed)
    if pred_category.lower() == "benign":
        safety = "SAFE"
    else:
        safety = "UNSAFE"

    return safety, pred_category


print("\n INTERACTIVE MODE ")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a prompt: ")

    if user_input.lower() == "exit":
        break

    safety, category = predict_prompt(user_input)

    print("\nPrediction:")
    print("Safety:", safety)
    print("Category:", category)
    print("-" * 40)
