import faiss,os
from sentence_transformers import SentenceTransformer
import ollama
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_vector_index(index_path):
    return faiss.read_index(index_path)

def retrieve_documents(model, index, document_texts, query, k=3):
    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search in FAISS index for the k nearest neighbors
    distances, indices = index.search(query_embedding, k)
    
    print(f"Indexes for retrieved documents : {indices}")

    # Retrieve the corresponding documents
    retrieved_docs = [document_texts.iloc[i][1] for i in indices[0]]
    return retrieved_docs

def generate_response_with_ollama(query, retrieved_docs,llm_model):
    prompt = f"Given the following documents:\n\n"
    for doc in retrieved_docs:
        prompt += f"- {doc}\n"
    prompt += f"\nAnswer the following question by strictly following the context: {query}\n"
    
    stream = ollama.chat(
        model=llm_model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=False,
    )
    #for chunk in stream:
        #print(chunk['message']['content'], end='', flush=True)
    
    return stream['message']['content']

if __name__ == "__main__":
    # Get the absolute path of the script    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, "vector_db", "faiss_index.index")
    csv_file = os.path.join(script_dir,"Metadata.csv")
    # mpnet embedding model : 
    embedding_model = "all-MiniLM-L6-v2"
    # LLM model : 
    llm_model = 'llama3.2:1b'

    # Load the model and FAISS index
    model = SentenceTransformer(embedding_model)
    index = load_vector_index(index_path)

    docs_data = pd.read_csv(csv_file)
    
    # Query the index
    # k : Number of documents retrieved from user query. 
    user_query = '''Can you mention the code for invoking Digital Twin Actions DT actions can be invoked from a client via a Python/Js script or using the Web-UI client'''
    retrieved_docs = retrieve_documents(model, index, docs_data, user_query, k=6)
    
    print(f"User Question : {user_query}")
    
    # Generate a response using Ollama based on the retrieved documents
    print(generate_response_with_ollama(user_query, retrieved_docs,llm_model))