import streamlit as st
from docx import Document
import openai
import ollama

# Set OpenAI API Key
openai.api_key = "your-openai-api-key"

def generate_resume(job_description, previous_experience, skills):
    # Prompt for AI
    prompt = f"""
    Create a professional resume based on the following details:
    Job Description: {job_description}
    Previous Experience: {previous_experience}
    Skills: {skills}
    Use concise bullet points, a professional summary, and the correct format.
    """
    
    stream = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return stream['message']['content']

def save_to_word(resume_content, filename="Generated_Resume.docx"):
    # Create a Word Document
    document = Document()
    for line in resume_content.split("\n"):
        if line.strip():
            document.add_paragraph(line.strip())
    document.save(filename)
    return filename
    
def main():

    # Streamlit UI
    st.title("AI Resume Generator")
    st.write("Generate a tailored resume based on your job description, experience, and skills.")

    job_description = st.text_area("Enter the Job Description", "")
    previous_experience = st.text_area("Enter Your Previous Experience", "")
    skills = st.text_area("Enter Your Key Skills", "")

    if st.button("Generate Resume"):
        with st.spinner("Generating your resume..."):
            try:
                # Generate resume using AI
                resume_content = generate_resume(job_description, previous_experience, skills)
                # Save the resume to a Word file
                file_name = save_to_word(resume_content)
                st.success("Resume generated successfully!")
                st.download_button(
                    label="Download Resume",
                    data=open(file_name, "rb").read(),
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    
    main()