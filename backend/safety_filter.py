

import os
import numpy as np
import joblib
from scipy.sparse import hstack

MODEL_PATH       = "backend/safety_model.joblib"
VECTORIZER_PATH  = "backend/vectorizer.joblib"
CHAR_VEC_PATH    = "backend/char_vectorizer.joblib"
LABEL_MAP_PATH   = "backend/label_mapping.joblib"
missing = [p for p in [MODEL_PATH, VECTORIZER_PATH, CHAR_VEC_PATH, LABEL_MAP_PATH]
           if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing model files: {missing}\n"
        "   Run  python ai-safety.py  first to train and save the model."
    )

_model        = joblib.load(MODEL_PATH)
_word_tfidf   = joblib.load(VECTORIZER_PATH)
_char_tfidf   = joblib.load(CHAR_VEC_PATH)
_int_to_label = joblib.load(LABEL_MAP_PATH)

print("Safety filter loaded successfully")


def _get_features(prompt: str):
    word_vec = _word_tfidf.transform([prompt])
    char_vec = _char_tfidf.transform([prompt])
    return hstack([word_vec, char_vec])


def _trigger_word(prompt: str) -> str | None:
    feature_names = _word_tfidf.get_feature_names_out()
    vec = _word_tfidf.transform([prompt]).toarray()[0]
    if vec.sum() == 0:
        return None
    return str(feature_names[int(np.argmax(vec))])


def _confidence(features) -> float:
    scores = _model.decision_function(features)[0]
    e = np.exp(scores - np.max(scores))
    proba = e / e.sum()
    return float(np.max(proba))


def check_prompt(prompt: str) -> dict:
    features = _get_features(prompt)
    pred_int = int(_model.predict(features)[0])
    category = _int_to_label[pred_int]

    return {
        "is_safe":      category == "safe",
        "category":     category,
        "trigger_word": _trigger_word(prompt),
        "confidence":   _confidence(features),
    }