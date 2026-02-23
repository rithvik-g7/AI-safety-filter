import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def generate_response(prompt: str) -> str:
    if not PINECONE_API_KEY:
        return "Pinecone API key not configured."

    response = f"AI Response to: '{prompt}'"

    return response