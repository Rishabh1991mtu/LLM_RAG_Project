import faiss,os
from sentence_transformers import SentenceTransformer
import ollama
import pandas as pd

def load_vector_index(index_path):
    return faiss.read_index(index_path)

def retrieve_documents(model, index, document_texts, query, k=3):
    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search in FAISS index for the k nearest neighbors
    distances, indices = index.search(query_embedding, k)
    print("distances",distances)
    print("indices",indices)

    # Retrieve the corresponding documents
    retrieved_docs = [document_texts.iloc[i][1] for i in indices[0]]
    return retrieved_docs

def generate_response_with_ollama(query, retrieved_docs):
    prompt = f"Given the following documents:\n\n"
    for doc in retrieved_docs:
        prompt += f"- {doc}\n"
    prompt += f"\nAnswer the following question: {query}\n"
    
    stream = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

if __name__ == "__main__":
    # Get the absolute path of the script    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(script_dir, "vector_db", "faiss_index.index")
    csv_file = os.path.join(script_dir,"Metadata.csv")
    embedding_model = "all-MiniLM-L6-v2"

    # Load the model and FAISS index
    model = SentenceTransformer(embedding_model)
    index = load_vector_index(index_path)

    docs_data = pd.read_csv(csv_file)
    
    # Query the index
    query = "What are the requirements on door locks ? "
    retrieved_docs = retrieve_documents(model, index, docs_data, query, k=6)
    
    # Generate a response using Ollama based on the retrieved documents
    generate_response_with_ollama(query, retrieved_docs)