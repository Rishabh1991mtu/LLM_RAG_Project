# RAG framework for Q/A chatbot 

### Installation instructions : 
1. Download [Ollama](https://ollama.com/download) for getting access ollama. It is package to run LLM locally.
2. Ollama documentation : https://github.com/ollama/ollama .
3. Create virtual python environment and download required python 
   packages from requirements.txt (pip install -r /path/to/requirements.txt)

### Ollama instructions : 
1. To download an open source model (ollama.com/library)-> run ollama model in terminal. For example : ollama run llama3.2
2. Code is using llama 3.2 which has to downloaded by the user by executing 

### Seach index generation :  
1. Add documents in data folder. Supported formats .txt and .pdf 
2. Run create_vector_index.py to create a search index (FAISS is being used to generate search index)
3. Parsed text from documents is saved in a .csv file which will be used to add in the contect window to generate an answer.

## Question Answer : 
1. Run RAG_pipeline.py to ask questions specific to documents. 
2. The questions can be added in the user query.   

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
2. Download [Ollama](https://ollama.com/download) for getting access ollama. It is package to run LLM locally.
   Ollama documentation : https://github.com/ollama/ollama .
3. pip install faiss-cpu sentence-transformers langchain ollama pandas

## Usage 


