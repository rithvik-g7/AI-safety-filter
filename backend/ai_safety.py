import os
import pickle
from typing import Dict
import numpy as np


# ==========================
# LOAD TRAINED MODEL
# ==========================

# ==========================
# KEYWORD CATEGORY ENGINE
# ==========================

CATEGORY_KEYWORDS = {
    "violence": [
        "kill","murder","shoot","bomb","weapon","poison","assassinate",
        "kidnap","attack","stab"
    ],
    "hacking": [
        "hack","phish","ddos","malware","virus","ransomware",
        "steal password","crack","exploit","bypass"
    ],
    "drugs": [
        "cocaine","heroin","meth","weed","drug trafficking","lsd"
    ],
    "weapons": [
        "gun","ak47","rifle","explosive","grenade","ammo"
    ],
    "illegal_activity": [
        "steal","scam","fraud","rob","blackmail","forgery"
    ],
    "self_harm": [
        "suicide","kill myself","self harm","end my life"
    ]
}

def detect_keyword_category(text: str):
    text_lower = text.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for word in keywords:
            if word in text_lower:
                return category, word

    return None, None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "safety_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    raise RuntimeError(
        "❌ Safety model files not found. "
        "Make sure safety_model.pkl and vectorizer.pkl are inside backend folder."
    )


# ==========================
# LABEL MAPPING
# ==========================

# IMPORTANT: Match this with your training encoding
LABEL_MAPPING = {
    0: "safe",
    1: "unsafe"
}


# ==========================
# SAFETY CHECK FUNCTION
# ==========================

def check_prompt(prompt: str) -> Dict:
    """
    Classifies prompt as safe or unsafe.
    Returns structured response for FastAPI.
    """

    if not prompt.strip():
        return {
            "is_safe": False,
            "category": "empty_input",
            "confidence": 1.0
        }

    # Vectorize input
    vectorized = vectorizer.transform([prompt])

    # Predict
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = float(np.max(probabilities))

    label = LABEL_MAPPING.get(prediction, "unknown")

    # 🚨 IF PROMPT IS UNSAFE
    if label == "unsafe":
        category, keyword = detect_keyword_category(prompt)

        return {
            "is_safe": False,
            "category": category if category else "unknown_risk",
            "trigger_word": keyword,
            "confidence": confidence
        }

    # ✅ SAFE PROMPT
    return {
        "is_safe": True,
        "category": None,
        "trigger_word": None,
        "confidence": confidence
    }