import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================
# 1️⃣ LOAD DATASETS
# ==========================

red_df = pd.read_parquet(
    "hf://datasets/knoveleng/redbench/AdvBench/train-00000-of-00001.parquet"
)
red_df = red_df[["prompt"]].dropna()
red_df["label"] = 1  # unsafe

oa_df = pd.read_parquet(
    "hf://datasets/OpenAssistant/oasst1/data/train-00000-of-00001-b42a775f407cee45.parquet"
)
oa_df = oa_df[oa_df["role"] == "prompter"]
oa_df = oa_df[["text"]].dropna().rename(columns={"text":"prompt"})
oa_df["label"] = 0  # safe

# balance
oa_df = oa_df.sample(n=len(red_df), random_state=42)
df = pd.concat([red_df, oa_df], ignore_index=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

# ==========================
# 2️⃣ TOKENIZER
# ==========================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class PromptDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PromptDataset(train_df["prompt"], train_df["label"])
test_dataset = PromptDataset(test_df["prompt"], test_df["label"])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ==========================
# 3️⃣ LOAD MODEL
# ==========================

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ==========================
# 4️⃣ TRAIN LOOP 🔥
# ==========================

epochs = 1   # keep 1 for quick demo (increase later)

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# ==========================
# 5️⃣ EVALUATION
# ==========================

model.eval()
preds, true = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

        preds.extend(predictions)
        true.extend(batch["labels"].numpy())

print("\nAccuracy:", accuracy_score(true, preds))
print(classification_report(true, preds))

# ==========================
# 6️⃣ INTERACTIVE TEST
# ==========================

import torch.nn.functional as F

def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    safe_prob = probs[0][0].item()
    unsafe_prob = probs[0][1].item()

    print(f"\nConfidence -> SAFE: {safe_prob:.3f} | UNSAFE: {unsafe_prob:.3f}")

    # 🔥 threshold trick (important)
    if unsafe_prob > 0.50:
        return "UNSAFE PROMPT"
    else:
        return "SAFE PROMPT"

while True:
    txt = input("Enter prompt: ")
    if txt == "exit":
        break
    print("Prediction:", predict(txt))
    print("-"*40)