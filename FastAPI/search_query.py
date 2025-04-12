from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

app = FastAPI()

# Load the same embedding model used during indexing
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Path to your saved Chroma DB folder
CHROMA_DB_PATH = "C:/Users/harsh.u/Desktop/Rag_system/Vector-DB"

# Load the Chroma vector store
vector_db = Chroma( collection_name="Patient_records" , persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Request model
class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search_vector_db(req: QueryRequest):
    query = req.query
    docs = vector_db.similarity_search(query, k=3)  # Top 3 similar chunks
    print(docs)
    return {
        "query": query,
        "results": [doc.page_content for doc in docs]
    }
    
