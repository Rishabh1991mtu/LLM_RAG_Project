import faiss
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd

def load_documents_from_files(directory_path):
    """
    Load text and PDF documents from the specified directory.
    """
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Load text files
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())  # Appending list of Document objects
        
        # Load PDF files
        elif filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())  # Appending list of Document objects
    
    return documents

def create_vector_index(embedding_model, documents, index_path, csv_path):
    # Load the Sentence Transformer model to create embeddings
    model = SentenceTransformer(embedding_model)

    # Split large documents into smaller chunks (optional)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_documents = text_splitter.split_documents(documents)

    # Generate embeddings for the documents
    document_texts = [doc.page_content for doc in split_documents]
    embeddings = model.encode(document_texts, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Total vectors in the FAISS index: {index.ntotal}")

    # Save the index to disk
    faiss.write_index(index, index_path)

    # Save index IDs and corresponding text chunks in a CSV file using pandas
    data = {"Index_ID": list(range(len(document_texts))), "Text_Chunk": document_texts}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return index, model, document_texts

if __name__ == "__main__":
    # Path to the directory with text files   
    
    # Get the absolute path of the script    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    directory_path = os.path.join(script_dir,"data")
    
    index_path = os.path.join(script_dir, "vector_db", "faiss_index.index")
    
    # Load and process documents
    documents = load_documents_from_files(directory_path)
    embedding_model = "all-MiniLM-L6-v2"
    
    # Save data in csv file : 
    csv_path = os.path.join(script_dir,"Metadata.csv")
    
    # Create and save FAISS index
    create_vector_index(embedding_model, documents, index_path, csv_path)
