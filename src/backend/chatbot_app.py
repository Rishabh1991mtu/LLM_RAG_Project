# api.py
import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrievel import initialize_components

# Initialize FastAPI app
app = FastAPI()

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 3  # Number of documents to retrieve

# Load environment variables and initialize components
script_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(script_dir, "vector_db", "faiss_index.index")
csv_file = os.path.join(script_dir, "Metadata.csv")

# Model names
embedding_model = "all-MiniLM-L6-v2"
llm_model = "llama3.2:1b"

# Initialize retrieval components
retriever, response_generator = initialize_components(csv_file, index_path, embedding_model, llm_model)

# FastAPI endpoint for chatbot query
@app.post("/query/")
async def query_chatbot(request: QueryRequest):
    try:
        # Retrieve documents based on query
        retrieved_docs = retriever.retrieve_documents(request.query, k=request.k)
        # Generate response using retrieved documents
        response = response_generator.generate_response(request.query, retrieved_docs)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
