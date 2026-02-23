import os
import pickle
from typing import Dict
import numpy as np


# ==========================
# LOAD TRAINED MODEL
# ==========================

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

    if label == "unsafe":
        return {
            "is_safe": False,
            "category": "unsafe_content",
            "confidence": confidence
        }

    return {
        "is_safe": True,
        "category": None,
        "confidence": confidence
    }