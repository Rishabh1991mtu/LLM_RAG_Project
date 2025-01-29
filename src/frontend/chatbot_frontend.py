import streamlit as st
import requests

# Define the FastAPI endpoint
API_URL = "http://127.0.0.1:8000/query/"

# Streamlit UI for input
st.title("RAG chatbot app")

# User input for the query
query = st.text_input("Enter your query:")

# User input for the number of documents to retrieve
num_docs = st.number_input("Number of documents to retrieve", min_value=1, max_value=10, value=3)

# Button to submit the query
if st.button("Submit Query"):
    if query:
        # Send POST request to FastAPI backend
        response = requests.post(API_URL, json={"query": query, "k": num_docs})
        
        if response.status_code == 200:
            st.write("Response:")
            
            # Streaming logic - handle each part of the response
            for part in response.iter_lines():
                if part:
                    part = part.decode("utf-8")  # Decode byte response to string
                    st.write(part)  # Display each part as it streams
        else:
            st.write(f"Error: {response.status_code}")
            st.write(response.text)
    else:
        st.write("Please enter a query.")
