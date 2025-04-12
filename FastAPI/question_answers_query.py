from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Fetch OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load embedding model and Chroma DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DB_PATH = "C:/Users/harsh.u/Desktop/Rag_system/Vector-DB"

vector_db = Chroma(
    collection_name="Patient_records",
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# Load the OpenAI LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.2,
    model_name="gpt-3.5-turbo"
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    return_source_documents=True
)

# Define input schema for QA requests
class QueryRequest(BaseModel):
    query: str

# Define QA endpoint
@app.post("/qa")
def qa_answer(req: QueryRequest):
    question = req.query
    result = qa_chain({"query": question})

    return {
        "question": question,
        "answer": result["result"],
        "source_documents": [doc.metadata for doc in result["source_documents"]]
    }