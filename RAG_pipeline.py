import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Step 1: Load the Sentence Transformer model to create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can change the model based on your needs

# Step 2: Example dataset (replace with actual data if needed)
documents = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Python is a widely used programming language.",
    "I enjoy hiking during the weekends.",
    "The Eiffel Tower is in Paris.",
    "Artificial Intelligence is transforming industries.",
    "Climate change is a major global issue.",
    "She loves painting and drawing in her free time.",
    "The stock market is highly volatile today.",
    "Football is the most popular sport in the world.",
    "Space exploration is a field full of opportunities.",
    "Reading books can improve your vocabulary."
]

# Step 3: Generate embeddings for the documents
embeddings = model.encode(documents, convert_to_numpy=True)

# Step 4: Create FAISS index for vector search
dimension = embeddings.shape[1]  # Get the dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # Use L2 distance (Euclidean distance)
index.add(embeddings)  # Add embeddings to the index
print(f"Total vectors in the FAISS index: {index.ntotal}")

# Step 5: Function to query FAISS and retrieve relevant documents
def retrieve_documents(query, k=3):
    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search in FAISS index for the k nearest neighbors
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the corresponding documents
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# Step 6: Function to use Ollama for generating answers based on the retrieved documents
def generate_response_with_ollama(query, retrieved_docs):
    prompt = f"Given the following documents:\n\n"
    for doc in retrieved_docs:
        prompt += f"- {doc}\n"
    prompt += f"\nAnswer the following question: {query}\n"
    
    # Use Ollama locally by sending an HTTP request (assuming Ollama is running locally)
    url = "http://localhost:11434/generate"  # Adjust this to your local Ollama endpoint
    data = {
        "prompt": prompt,
        "model": "llama2"  # Adjust this to the specific model you're using with Ollama
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['text']
    else:
        return "Failed to retrieve response from Ollama."

# Step 7: Example usage of the RAG pipeline
query = "What are the most popular programming languages?"
retrieved_docs = retrieve_documents(query, k=3)  # Retrieve top 3 relevant documents
print("Retrieved Documents:", retrieved_docs)

# Generate a response using Ollama based on the retrieved documents
response = generate_response_with_ollama(query, retrieved_docs)
print("Generated Response:", response)
