# retrieval.py
import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import ollama

class VectorStore:
    """
    This class handles the loading and searching of the FAISS vector index.
    """
    def __init__(self, index_path):
        # Load the FAISS index from the specified path
        self.index = self.load_vector_index(index_path)

    @staticmethod
    def load_vector_index(index_path):
        """
        Loads a FAISS index from the specified file path.
        
        Args:
            index_path (str): Path to the FAISS index file.

        Returns:
            faiss.Index: Loaded FAISS index object.
        """
        return faiss.read_index(index_path)

    def search(self, query_embedding, k=3):
        """
        Searches for the k nearest neighbors in the vector index.

        Args:
            query_embedding (numpy.ndarray): The embedding of the query.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            tuple: Distances and indices of the nearest neighbors.
        """
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices


class Retriever:
    """
    This class retrieves relevant documents by embedding the query and
    searching for similar documents in the vector index.
    """
    def __init__(self, embedding_model_name, vector_store, document_texts):
        # Initialize the embedding model and vector store
        self.model = SentenceTransformer(embedding_model_name)
        self.vector_store = vector_store
        self.document_texts = document_texts  # DataFrame containing document texts

    def retrieve_documents(self, query, k=3):
        """
        Embeds the query, searches for the nearest documents, and retrieves them.

        Args:
            query (str): The user's query.
            k (int): Number of documents to retrieve.

        Returns:
            list: List of retrieved document texts.
        """
        # Convert query to an embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search the vector store for the k nearest documents
        distances, indices = self.vector_store.search(query_embedding, k)
        print(f"Indexes for retrieved documents : {indices}")
        
        # Retrieve the document texts based on the retrieved indices
        return [self.document_texts.iloc[i][1] for i in indices[0]]


class ResponseGenerator:
    """
    This class generates responses using the retrieved documents and the LLM model.
    """
    def __init__(self, llm_model):
        # Set the language model to be used for generating responses
        self.llm_model = llm_model

    def generate_response(self, query, retrieved_docs):
        """
        Generates a response based on the retrieved documents and the user's query.

        Args:
            query (str): The user's query.
            retrieved_docs (list): List of documents retrieved from the index.

        Returns:
            str: The generated response text.
        """
        # Formulate a prompt for the LLM using the retrieved documents
        prompt = "Given the following documents:\n\n"
        for doc in retrieved_docs:
            prompt += f"- {doc}\n"
        prompt += f"\nAnswer the following question by strictly following the context: {query}\n"

        # Generate the response using the standard method
        return self._generate_response_from_prompt(prompt)

    def stream_response(self, query, retrieved_docs):
        """
        Streams a response based on the retrieved documents and the user's query.

        Args:
            query (str): The user's query.
            retrieved_docs (list): List of documents retrieved from the index.

        Yields:
            str: Chunks of the generated response text.
        """
        # Formulate a prompt for the LLM using the retrieved documents
        prompt = "Given the following documents:\n\n"
        for doc in retrieved_docs:
            prompt += f"- {doc}\n"
        prompt += f"\nAnswer the following question by strictly following the context: {query}\n"

        # Generate the response using Ollama's chat model with streaming enabled
        stream = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )
        
        # Yield each chunk of the response as it becomes available
        for chunk in stream:
            yield chunk['message']['content']

    def _generate_response_from_prompt(self, prompt):
        """
        Generates a full response from a prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The generated response text.
        """
        # Directly get the full response without streaming
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,  # or omit stream altogether
        )
        return response['message']['content']


def initialize_components(csv_file, index_path, embedding_model, llm_model):
    """
    Initializes and returns instances of the VectorStore, Retriever, and ResponseGenerator.

    Args:
        csv_file (str): Path to the CSV file containing document data.
        index_path (str): Path to the FAISS index file.
        embedding_model (str): Name of the embedding model for SentenceTransformer.
        llm_model (str): Name of the language model for response generation.

    Returns:
        tuple: Initialized instances of Retriever and ResponseGenerator.
    """
    # Load document data from CSV file
    docs_data = pd.read_csv(csv_file)
    
    # Initialize the vector store with the FAISS index
    vector_store = VectorStore(index_path)
    
    # Initialize the retriever with the embedding model, vector store, and documents
    retriever = Retriever(embedding_model, vector_store, docs_data)
    
    # Initialize the response generator with the language model
    response_generator = ResponseGenerator(llm_model)
    
    return retriever, response_generator
