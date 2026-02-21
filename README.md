# 🛡️ AI Safety Filter

An AI Safety Filter that classifies user prompts as **Safe** or **Unsafe** and, if unsafe, identifies the **specific risk category**.  
This project uses **Machine Learning and Natural Language Processing (NLP)** to support responsible and secure AI usage.

---

## 📌 Overview

With the growing use of AI systems, prompt safety has become essential.  
This AI Safety Filter acts as a **pre-moderation layer**, analyzing text inputs before they are processed by an AI model.

The system can be used in:
- Chatbots
- AI assistants
- Educational platforms
- Content moderation pipelines

---

## 🚀 Features

- Text-based safety classification  
- Multi-class risk categorization  
- TF-IDF vectorization  
- Logistic Regression / SVM models  
- Model evaluation using standard metrics  
- Interactive prompt testing  

---

## 🧠 Machine Learning Pipeline

1. **Text Preprocessing**
   - Lowercasing
   - Tokenization
   - Noise removal
   - TF-IDF feature extraction

2. **Model Training**
   - Logistic Regression for baseline performance
   - Support Vector Machine for improved class separation

3. **Prediction**
   - Safe prompt  
   - Unsafe prompt → categorized into a specific risk class

---

## 📊 Dataset

- RED-EVAL / REDBENCH dataset (Hugging Face)
- Labeled prompts across multiple AI safety categories

Example categories:
- Hate / Harassment
- Violence
- Illegal Activities
- Self-harm
- Extremism

---

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- R² Score
- Precision, Recall, F1-score (optional)

---

## 🧩 Tech Stack

- Python
- scikit-learn
- pandas
- NumPy
- Hugging Face Datasets
- TF-IDF Vectorizer

