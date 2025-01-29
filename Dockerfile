# Use Python 3.10 as base image to avoid compatibility issues with certain libraries
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies for Ollama, FAISS, and other Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Expose the Streamlit port
EXPOSE 8501

# Start both FastAPI (Uvicorn) and Streamlit in the background
CMD ["sh", "-c", "PYTHONPATH=/app uvicorn src.backend.chatbot_app:app --host 0.0.0.0 --port 8000 & streamlit run src/frontend/chatbot_frontend.py --server.port 8501 --server.address 0.0.0.0"]