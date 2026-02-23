# chatbot.py

import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def generate_response(prompt: str) -> str:
    """
    Generates chatbot response.
    Replace this with your Pinecone + LLM logic.
    """

    if not PINECONE_API_KEY:
        return "⚠️ Pinecone API key not configured."

    # 🔥 Replace this section with your real RAG / LLM logic
    # -------------------------------------------------------
    # Example placeholder logic:
    response = f"🤖 AI Response to: '{prompt}'"

    return response