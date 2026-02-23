$env:PINECONE_API_KEY="pcsk_66BKt3_BKJGuh9pzsnh753Ud72797kV8S7HekPQVdRiGUa69MeCxgA34FuRjxQBZpCAsNz"
$env:PINECONE_INDEX_NAME="jarvis-knowledge"

.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload