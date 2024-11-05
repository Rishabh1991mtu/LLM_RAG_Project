# api.py
import os 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retrievel import initialize_components
from fastapi.responses import StreamingResponse
import json

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        async def generate_response():
            for part in response_generator.stream_response(request.query, retrieved_docs):
                #yield json.dumps({"answer": part}) + "\n"  # Stream each part as a separate JSON object
                yield part
                
        #return StreamingResponse(generate_response(), media_type="application/json")
        return StreamingResponse(generate_response())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Notes : 

# End point for FastAPI : http://127.0.0.1:8000/docs
# End point for HTML page : http://localhost:8080/Chatbot_Inteface.html 
# Ctrl + C to shut down 
# Run API endpoint :  uvicorn chatbot_app:app --reload
# Chatbot Interface : python -m http.server 8080 (cd to frontend folder and run this command)