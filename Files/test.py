# -------------------------------
# Import necessary libraries
# -------------------------------
from langchain_community.document_loaders import PyPDFLoader  # Load PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split large text into chunks
from langchain_openai import OpenAIEmbeddings  # Use OpenAI to create embeddings for text chunks
from langchain_community.vectorstores import FAISS  # FAISS is used for vector similarity search
from langchain_openai import ChatOpenAI  # Interface to OpenAI chat models (GPT-4 etc.)
import os
from dotenv import load_dotenv  # Load environment variables from a .env file
from langchain.chains.combine_documents import create_stuff_documents_chain  # Chain to combine context + prompt
from langchain.chains import create_retrieval_chain  # Combine retriever and document chain into one pipeline
from langchain_core.prompts import PromptTemplate  # Create a custom prompt template for answering questions

# -------------------------------
# Load environment variables (like OpenAI API key) from .env file
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_KEY")  # Get the OpenAI key from .env

# -------------------------------
# Step 1: Load and Preprocess Documents
# -------------------------------
def load_and_preprocess_documents(file_path):
    # Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the PDF into smaller chunks for embedding and retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    return docs

# -------------------------------
# Step 2: Generate Embeddings and Store in FAISS
# -------------------------------
def create_vector_store(documents, save_path="vector_store"):
    # Initialize the embedding model using your OpenAI API key
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Check if a previously saved FAISS vector store exists
    if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, "index.faiss")):
        print(f"Loading existing vector store from '{save_path}'...")
        vector_store = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True  # Required for local loading
        )
    else:
        print("Creating new vector store...")
        # Create a new FAISS vector store from the document embeddings
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(save_path)
        print(f"Saved vector store to '{save_path}'")

    return vector_store

# -------------------------------
# Step 3: Create a RAG (Retrieval-Augmented Generation) Pipeline
# -------------------------------
def create_rag_pipeline(vector_store):
    # Create a retriever to fetch relevant documents based on similarity
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Initialize the LLM (GPT-4) with a deterministic output (temperature=0)
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0,
        model="gpt-4"
    )

    # Define the prompt format used to ask the LLM a question based on context
    prompt = PromptTemplate.from_template(
        "Use the following pieces of context to answer the question:\n\n{context}\n\nQuestion: {input}\n\nAnswer:"
    )

    # Create a chain that combines the context from documents and the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Combine the retriever and the document_chain into a full RAG pipeline
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain

# -------------------------------
# Step 4: Ask Questions and Display Answers
# -------------------------------
def ask_questions(qa_chain):
    print("\nType your question or 'exit' to quit:\n")
    while True:
        # Get user input
        query = input("Your Question: ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        # Pass the question through the RAG pipeline
        response = qa_chain.invoke({"input": query})

        # Print the full response dictionary (optional for debugging)
        print(response, "\n")

        # Print only the final answer from the LLM
        print("\nAnswer:", response["answer"], "\n")

# -------------------------------
# Main function: Entry point of the script
# -------------------------------
if __name__ == "__main__":
    # Specify the path to your PDF document
    file_path = "pdf/Medical_document.pdf"

    print("Loading and preprocessing documents...")
    documents = load_and_preprocess_documents(file_path)

    print("Creating vector store...")
    vector_store = create_vector_store(documents)

    print("Initializing RAG pipeline...")
    qa_chain = create_rag_pipeline(vector_store)

    print("System is ready to answer your questions.")
    ask_questions(qa_chain)
