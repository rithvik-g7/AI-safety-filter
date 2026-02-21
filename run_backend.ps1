$env:PINECONE_API_KEY="pcsk_1YgxJ_3BPAPLgkUC8riR4tnJZDzNUVZVM2yWJ8acY9RM89w4Hs3P2LQts7tk2sqkbXGSr"
$env:PINECONE_INDEX_NAME="jarvis-knowledge"

.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload