import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

def create_vector_index(embedding_model, documents):
    # Step 1: Load the Sentence Transformer model to create embeddings
    model = SentenceTransformer(embedding_model)

    # Step 3: Generate embeddings for the documents
    embeddings = model.encode(documents, convert_to_numpy=True)

    # Step 4: Create FAISS index for vector search
    dimension = embeddings.shape[1]  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance (Euclidean distance)
    index.add(embeddings)  # Add embeddings to the index
    print(f"Total vectors in the FAISS index: {index.ntotal}")
    
    return model, index

# Step 5: Function to query FAISS and retrieve relevant documents
def retrieve_documents(model, index, documents, query, k=3):
    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search in FAISS index for the k nearest neighbors
    distances, indices = index.search(query_embedding, k)
    print(distances, indices)

    # Retrieve the corresponding documents
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# Step 6: Function to use Ollama for generating answers based on the retrieved documents
def generate_response_with_ollama(query, retrieved_docs):
    prompt = f"Given the following documents:\n\n"
    for doc in retrieved_docs:
        prompt += f"- {doc}\n"
    prompt += f"\nAnswer the following question: {query}\n"
    
    stream = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

# Step 7: Example usage of the RAG pipeline
if __name__ == "__main__":
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
    
    embedding_model = "all-MiniLM-L6-v2"
    model, index = create_vector_index(embedding_model, documents)
    
    query = "Who will win US elections 2024 ?"
    retrieved_docs = retrieve_documents(model, index, documents, query, k=3)  # Retrieve top 3 relevant documents
    print("Retrieved Documents:", retrieved_docs)

    # Generate a response using Ollama based on the retrieved documents
    generate_response_with_ollama(query, retrieved_docs)
