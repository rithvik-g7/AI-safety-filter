from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = FastAPI(title="Jarvis AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# setup pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# use ollama for embeddings too - completely free
embeddings = OllamaEmbeddings(model="llama2")

llm = Ollama(
    model="llama2",
    temperature=0.7
)

vector_store = None
qa_chain = None

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global vector_store, qa_chain
    
    index_name = os.getenv("PINECONE_INDEX_NAME", "jarvis-knowledge")
    
    # create index if it doesn't exist
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=4096,  # ollama llama2 embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    index = pc.Index(index_name)
    
    vector_store = LangchainPinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    # setup qa chain with retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

@app.get("/")
async def root():
    return {"message": "Jarvis AI Assistant API", "status": "online"}

@app.post("/api/chat")
async def chat(request: QueryRequest):
    try:
        if not qa_chain:
            raise HTTPException(status_code=500, detail="QA chain not initialized")
        
        result = qa_chain({"query": request.query})
        
        return {
            "response": result["result"],
            "sources": len(result.get("source_documents", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-knowledge")
async def add_knowledge(request: DocumentRequest):
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(request.text)
        
        vector_store.add_texts(chunks)
        
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