# Retrieval-Augmented Generation (RAG) Framework for Question Answering  

This project implements a **Retrieval-Augmented Generation (RAG)** framework that utilizes **FAISS** for vector search, **Sentence Transformers** for generating embeddings, and **Ollama's LLMs** to provide conversational answers based on retrieved documents. This repository is designed for efficient question answering using indexed text and PDF documents.  

---

## Table of Contents  
1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Usage](#usage)  
6. [How It Works](#how-it-works)  
7. [Customization](#customization)  
8. [License](#license)  

---

## Features  
- Supports **text (.txt)** and **PDF** documents for indexing.  
- Uses **FAISS** to build and store vector indices for fast retrieval.  
- **Sentence Transformers** model for document embedding.  
- **Ollama** LLMs for generating contextual responses based on retrieved documents.  
- Automatically splits long documents into smaller chunks for efficient indexing.  
- Saves metadata (text chunks and index IDs) in a CSV file for transparency.  

---

## Prerequisites  
Make sure the following dependencies are installed:  
- Python 3.7+  
- FAISS (`faiss-cpu`)  
- Sentence Transformers  
- LangChain  
- Ollama SDK  
- pandas  

---

## Installation  
1. Clone the repository:  
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
2. Download [Ollama](https://ollama.com/download) for acessing and running open source LLM models locally
   - Ollama documentation : https://github.com/ollama/ollama .
   - To download a model use ollama pull model .
   ```python
   ollama run llama3.2:1b 	
   
3. Install the python dependencies using the requirements.txt file : 
   ```python
   pip install -r requirements.txt

## Usage 
1. **Indexing Documents:**
	- To create a FAISS index from your text and PDF files, follow these steps:
	- Place your documents in the data/ directory : 
	- Run the script create_vector_index.py to index the documents
	- This will perform the following tasks :
		-Split documents into smaller chunks.
		-Create embeddings using an embedding model (by defualt : all-MiniLM-L6-v2 model)
		-Save the FAISS index in vector_db/faiss_index.index.
		-Store metadata in Metadata.csv.

2. **Querying the Index and Generating Responses:**
	- You can query the index by modifying the user_query variable in the RAG_pipeline.py script and running it
	- user_query = '''What is the name of the candidate?'''
	- The framework will:
		- Retrieve the most relevant documents based on your query.
		- Use Ollama's LLM to generate a response using the retrieved documents.

## How It Works  

1. **Document Loading and Splitting:**  
   - Loads **text** and **PDF** documents from the `data/` directory.  
   - Splits large documents into smaller chunks using **RecursiveCharacterTextSplitter** from LangChain to ensure efficient retrieval.  

2. **Embedding and Indexing:**  
   - Converts document chunks into numerical vectors (embeddings) using **Sentence Transformers** (e.g., `all-MiniLM-L6-v2`).  
   - Stores the embeddings in a **FAISS** index, which enables fast similarity-based searches.  

3. **Querying and Retrieval:**  
   - Converts the user query into an embedding using the same **Sentence Transformers** model.  
   - Searches the FAISS index to find the top `k` most relevant document chunks based on cosine similarity or L2 distance.  

4. **Response Generation:**  
   - Uses **Ollama's LLM** to generate a response based on the retrieved document chunks.  
   - The LLM is provided with both the retrieved documents and the user query to create a relevant and contextual response.  

## Customization

1. **Changing the Embedding Model:**

	- Update the embedding_model variable in the script. 
	```python
	embedding_model = "all-MiniLM-L6-v2" . 
2. **Changing the Number of Retrieved Documents**:
	- Adjust the k parameter in the retrieve_documents function:
	```python
	retrieved_docs = retrieve_documents(model, index, docs_data, user_query, k=6)
3. **Modifying the LLM Model:
	- Update the llm_model variable to use a different Ollama model:
	```python
	llm_model = 'llama3.2:1b'
	