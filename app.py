import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Validate API keys
if not groq_api_key:
    print("Error: GROQ_API_KEY not found in environment variables.")
    sys.exit(1)

if not google_api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = google_api_key

try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
except Exception as e:
    print(f"Error initializing ChatGroq: {e}")
    sys.exit(1)

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are an AI Resume Evaluator.

<context>
{context}
</context>

Based on the above Resume and Job Description, provide:
1. A FIT SCORE from 0 to 100.
2. Key strengths match.
3. Skill gaps or missing experience.
4. Actionable feedback to improve resume for this job.

Respond in a clear and concise format.
Questions: {input}
""")

# Function to create vector store from PDFs
def create_chroma_vectorstore(resume_pdf, jd_pdf):
    # Validate file paths
    if not os.path.exists(resume_pdf):
        raise FileNotFoundError(f"Resume PDF not found: {resume_pdf}")
    if not os.path.exists(jd_pdf):
        raise FileNotFoundError(f"Job Description PDF not found: {jd_pdf}")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        resume_docs = PyPDFLoader(resume_pdf).load()
        jd_docs = PyPDFLoader(jd_pdf).load()

        for doc in resume_docs:
            doc.metadata["source"] = "Resume"
        for doc in jd_docs:
            doc.metadata["source"] = "Job Description"

        all_docs = resume_docs + jd_docs
        split_docs = text_splitter.split_documents(all_docs)

        vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_store")
        return vectordb
    except Exception as e:
        raise Exception(f"Error creating vector store: {e}")


if __name__ == "__main__":
    print("=== Resume Evaluation Console App ===")
    
    # Get file paths from user
    resume_path = input("Enter path to Resume PDF: ").strip()
    jd_path = input("Enter path to Job Description PDF: ").strip()
    
    # Validate paths exist
    if not resume_path or not jd_path:
        print("Error: Both resume and job description paths are required.")
        sys.exit(1)
    
    # Convert to absolute paths
    resume_path = os.path.abspath(resume_path)
    jd_path = os.path.abspath(jd_path)
    
    try:
        vectordb = create_chroma_vectorstore(resume_path, jd_path)
        retriever = vectordb.as_retriever()

        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        user_query = "Evaluate this resume against the job description."
        result = rag_chain.invoke({"input": user_query})

        print("\n=== Evaluation Result ===")
        print(result["answer"])
        
    except FileNotFoundError as e:
        print(f"File Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
