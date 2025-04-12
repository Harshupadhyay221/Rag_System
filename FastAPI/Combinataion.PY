from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load enviornment variables 
load_dotenv()

# create fastapi app
app = FastAPI()

# Fetch OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load embedding model and Chroma DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DB_PATH = "C:/Users/harsh.u/Desktop/Rag_system/Vector-DB"
vector_db = Chroma(collection_name="Patient_records", persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Load OpenAI model
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2, model_name="gpt-3.5-turbo")

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    return_source_documents=True
)

# Request model
class QueryRequest(BaseModel):
    query: str
    
# Search endpoint (from before)
@app.post("/search")
def search_vector_db(req: QueryRequest):
    query = req.query
    docs = vector_db.similarity_search(query, k=3)
    return {
        "query": query,
        "results": [doc.page_content for doc in docs]
    }
    
# New QA endpoint
@app.post("/qa")
def qa_answer(req: QueryRequest):
    question = req.query
    result = qa_chain({"query": question})

    return {
        "question": question,
        "answer": result["result"],
        "source_documents": [doc.metadata for doc in result["source_documents"]]
    }