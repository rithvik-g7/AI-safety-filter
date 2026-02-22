import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

print("Loading datasets...")

# ==========================
# 1️⃣ SAFE DATA (OpenAssistant)
# ==========================

splits = {'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet', 'validation': 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'}
oasst_df = pd.read_parquet("hf://datasets/OpenAssistant/oasst1/" + splits["train"])

oasst_df = oasst_df[oasst_df["role"] == "prompter"]
oasst_df = oasst_df[["text"]].dropna()
oasst_df.rename(columns={"text": "text"}, inplace=True)
oasst_df["label"] = 0  # SAFE

print("OpenAssistant safe samples:", len(oasst_df))

# ==========================
# 2️⃣ UNSAFE DATA (ToxiGen)
# ==========================

# ==========================
# 2️⃣ UNSAFE DATA (ToxiGen - Annotated Version)
# ==========================

splits = {
    'train': 'annotated/train-00000-of-00001.parquet',
    'test': 'annotated/test-00000-of-00001.parquet'
}

toxigen_df = pd.read_parquet(
    "hf://datasets/toxigen/toxigen-data/" + splits["train"]
)

print("ToxiGen columns:", toxigen_df.columns)

# Filter highly toxic samples (human annotated)
toxigen_df = toxigen_df[toxigen_df["toxicity_human"] > 0.7]

toxigen_df = toxigen_df[["text"]].dropna()
toxigen_df["label"] = 1  # UNSAFE

print("ToxiGen unsafe samples:", len(toxigen_df))

# ==========================
# 3️⃣ UNSAFE DATA (RealToxicityPrompts)
# ==========================

rtp_df = pd.read_json(
    "hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl",
    lines=True
)

print("RTP columns:", rtp_df.columns)

# Extract safely
rtp_df["text"] = rtp_df["prompt"].apply(lambda x: x.get("text"))
rtp_df["toxicity"] = rtp_df["prompt"].apply(lambda x: x.get("toxicity"))

# Filter high toxicity
rtp_df = rtp_df[rtp_df["toxicity"] > 0.8]

rtp_df = rtp_df[["text"]].dropna()
rtp_df["label"] = 1

print("RealToxicityPrompts unsafe samples:", len(rtp_df))

# ==========================
# 4️⃣ BALANCE DATASET
# ==========================

unsafe_df = pd.concat([toxigen_df, rtp_df], ignore_index=True)

safe_sample = oasst_df.sample(n=15000, random_state=42)
unsafe_sample = unsafe_df.sample(n=15000, random_state=42)

df = pd.concat([safe_sample, unsafe_sample], ignore_index=True)

print("\nFinal dataset distribution:")
print(df["label"].value_counts())

# ==========================
# 5️⃣ TRAIN / TEST SPLIT
# ==========================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# ==========================
# 6️⃣ TOKENIZATION
# ==========================

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ==========================
# 7️⃣ MODEL
# ==========================

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# ==========================
# 8️⃣ METRICS
# ==========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ==========================
# 9️⃣ TRAINING ARGS
# ==========================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("\nTraining...\n")
trainer.train()

# ==========================
# 🔟 EVALUATION
# ==========================

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["SAFE", "UNSAFE"]))

# ==========================
# 1️⃣1️⃣ INTERACTIVE MODE
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model.to(device)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "UNSAFE" if prediction == 1 else "SAFE"

print("\n🔎 INTERACTIVE MODE")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter prompt: ")
    if user_input.lower() == "exit":
        break
    result = predict(user_input)
    print("\nPrediction:", result)
    print("-" * 40)