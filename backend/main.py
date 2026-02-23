from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import httpx
from pinecone import Pinecone, ServerlessSpec
import json
from backend.ai_safety import check_prompt

load_dotenv()

# 🔑 FORCE LOAD ENV VARIABLES (Windows-safe)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jarvis-knowledge")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "❌ PINECONE_API_KEY not found.\n"
        "Make sure your .env file is in project root and contains:\n"
        "PINECONE_API_KEY=your_key_here"
    )

app = FastAPI(title="Jarvis AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = PINECONE_INDEX_NAME

index = None

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    text: str

async def get_ollama_embedding(text: str):
    """Get embeddings from Ollama"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "llama2", "prompt": text}
        )
        return response.json()["embedding"]

async def get_ollama_response(prompt: str):
    """Get response from Ollama"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]

@app.on_event("startup")
async def startup_event():
    global index
    
    # create index if doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)

@app.get("/")
async def root():
    return {"message": "Jarvis AI Assistant", "status": "online"}

@app.post("/api/chat")
async def chat(request: QueryRequest):
    try:
        # 1️⃣ Safety Check
        safety_result = check_prompt(request.query)

        if not safety_result["is_safe"]:
            return {
                 "status": "blocked",
                 "message": "Prompt did not pass safety check",
                 "category": safety_result["category"],
                 "trigger_word": safety_result.get("trigger_word"),
                 "confidence": safety_result["confidence"]
                 }

        # 2️⃣ Generate Embedding
        query_embedding = await get_ollama_embedding(request.query)

        # 3️⃣ Search Vector DB
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        # 4️⃣ Build Context
        context = ""
        if results.matches:
            context = "\n".join(
                [match.metadata.get("text", "") for match in results.matches]
            )

        # 5️⃣ Construct Prompt
        prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, answer based on general knowledge.

Context:
{context}

Question:
{request.query}

Answer:
"""

        # 6️⃣ Generate LLM Response
        response = await get_ollama_response(prompt)

        return {
            "status": "allowed",
            "response": response,
            "sources": len(results.matches) if results.matches else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-knowledge")
async def add_knowledge(request: DocumentRequest):
    try:
        # split text into chunks
        chunks = [request.text[i:i+500] for i in range(0, len(request.text), 450)]
        
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = await get_ollama_embedding(chunk)
            vectors.append({
                "id": f"doc_{hash(chunk)}_{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            })
        
        index.upsert(vectors=vectors)
        
        return {
            "message": "Knowledge added successfully",
            "chunks_added": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "llm": "ollama-llama2",
        "vector_db": "pinecone"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)