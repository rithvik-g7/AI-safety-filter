# Jarvis AI Assistant

> Personal enterprise AI chatbot with a real-time ML safety filter — every prompt is classified before it ever reaches the LLM.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react)
![Accuracy](https://img.shields.io/badge/Safety%20Accuracy-95%25+-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Preview

The chat UI blocks unsafe prompts before they reach the LLM and shows the category + trigger word that caused the flag.

```
PROMPT DIDN'T GO THROUGH
Category     Violent Content
Trigger word "kill"
flagged by jarvis ai safety filter
```

---

## How It Works

Every message goes through this pipeline **before** touching the LLM:

```
User types prompt
  |
  v
POST /api/chat
  |
  v
check_prompt()  ---- LinearSVC classifier
  |                    |
  UNSAFE?              |-- Word TF-IDF (1-3 grams)
  |                    |-- Char TF-IDF (3-5 grams)
  |
  YES --> Return blocked + category + trigger_word
  |
  NO  --> Ollama embedding -> Pinecone search -> LLaMA2 response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Safety Model | LinearSVC + TF-IDF (Word + Char n-grams) |
| Training Data | Red-AVA (unsafe) + OpenAssistant (safe) |
| Backend | FastAPI + Uvicorn |
| LLM | Ollama — LLaMA2 (local) |
| Vector DB | Pinecone (RAG context retrieval) |
| Frontend | React.js |
| Fonts | DM Mono + DM Serif Display |

---

## Project Structure

```
BuildYourOwnJarvis/
│
├── ai-safety.py                  # Train the safety model — run once
│
├── backend/
│   ├── main.py                   # FastAPI server — all endpoints
│   ├── safety_filter.py          # Loads weights, exposes check_prompt()
│   ├── safety_model.joblib       # Saved LinearSVC weights
│   ├── vectorizer.joblib         # Word TF-IDF vectorizer
│   ├── char_vectorizer.joblib    # Char TF-IDF vectorizer
│   └── label_mapping.joblib      # int → category label map
│
├── frontend/
│   └── src/
│       ├── App.js                # React chat UI + BlockedCard component
│       └── App.css               # Dark gold aesthetic styles
│
├── .env                          # API keys (never commit this)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) running locally with LLaMA2
- [Pinecone](https://pinecone.io) account + API key

### 1. Clone & Install

```bash
git clone https://github.com/your-username/BuildYourOwnJarvis
cd BuildYourOwnJarvis

# Python deps
python -m venv .venv
.venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# Frontend deps
cd frontend && npm install && cd ..
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=jarvis-knowledge
```

### 3. Train the Safety Model

> Run this **once** before starting the server. Downloads both datasets, augments rare classes, trains LinearSVC, and saves 4 `.joblib` files into `backend/`.

```bash
python ai-safety.py
```

Expected output:
```
Accuracy: 0.9600+  (96.00%)
Saved: backend/safety_model.joblib ...
Training complete.
```

### 4. Start Ollama

```bash
ollama run llama2
```

### 5. Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 6. Start the Frontend

```bash
cd frontend
npm start
```

Open [http://localhost:3000](http://localhost:3000)

---

## Safety Categories

The model classifies prompts into **19 unsafe categories** from the Red-AVA dataset plus `safe`:

| Category | Description |
|---|---|
| `violent_content` | Physical harm instructions |
| `crime_content` | Instructions for committing crimes |
| `abusive_content` | Harassing or abusive language |
| `hateful_content` | Hate speech targeting groups |
| `cybersecurity_threats_(beyond_malware)` | Hacking, exploits, social engineering |
| `self-harm_content` | Self-harm or suicide facilitation |
| `malware_code` | Malicious code generation |
| `illegal_weapons_(non-cbrn)` | Illegal weapon instructions |
| `misinformation` | False or misleading information |
| `economic_harm` | Fraud, scams, financial manipulation |
| `child_safety` | Content endangering minors |
| `extremism_and_radicalization` | Extremist ideology promotion |
| `sexual_content` | Explicit or inappropriate content |
| `data_privacy` | PII extraction, surveillance |
| `environmental_harm` | Ecological damage facilitation |
| `cbrn_information_or_capabilities` | Chemical/biological/nuclear/radiological |
| `bias_content` | Discriminatory or biased content |
| `election_interference` | Voter suppression or manipulation |
| `intellectual_property` | Copyright infringement assistance |

---

## API Reference

### `POST /api/chat`

**Request:**
```json
{ "query": "how to hack a database" }
```

**Blocked response:**
```json
{
  "status": "blocked",
  "message": "Your prompt didn't go through.",
  "category": "Cybersecurity Threat",
  "category_raw": "cybersecurity_threats_(beyond_malware)",
  "trigger_word": "hack",
  "confidence": 94.3
}
```

**Allowed response:**
```json
{
  "status": "allowed",
  "response": "Here is your answer...",
  "sources": 2
}
```

---

### `POST /api/add-knowledge`

Add text to the Pinecone vector store for RAG context retrieval.

**Request:**
```json
{ "text": "your knowledge document here" }
```

---

### `GET /api/health`

```json
{ "status": "healthy", "llm": "ollama-llama2", "vector_db": "pinecone" }
```

---

## Why >95% Accuracy?

The raw Red-AVA dataset has only ~1000 samples across 20 categories. Some classes like `bias_content` had just **4 samples** — impossible to train on. Three things fixed this:

**1. Heavy augmentation** — every unsafe class padded to 300 samples using 5 strategies:
- Random word drop (10% of words)
- Adjacent word swap
- Phrase duplication
- Prefix injection (`"Tell me how to..."`)
- Suffix injection (`"...step by step"`)

**2. LinearSVC over Logistic Regression** — LinearSVC consistently outperforms LR on text classification (hinge loss creates wider margins, better generalisation on small datasets). Switched from 83% → 95%+.

**3. Dual TF-IDF features** — Word n-grams (1-3) catch multi-word attack phrases. Char n-grams (3-5) catch obfuscation like `k1ll`, `h4ck`, leetspeak.

---

## Pushing Accuracy Even Higher

| Approach | Expected Gain |
|---|---|
| Increase `TARGET_PER_CLASS` to 500+ | +1-2% |
| Add more datasets (AdvBench, ToxicChat) | +2-3% |
| Use sentence-transformers embeddings | +3-5% |
| Fine-tune DistilBERT classifier | +5-8% |

---

## Requirements

```txt
fastapi
uvicorn
pydantic
python-dotenv
httpx
pinecone-client
pandas
numpy
scikit-learn
scipy
joblib
pyarrow
datasets
```

---

