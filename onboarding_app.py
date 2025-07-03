import os
import sys
import streamlit as st
import tempfile
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

# Page configuration
st.set_page_config(
    page_title="Employee Onboarding Q&A System",
    page_icon="🏢",
    layout="wide"
)

# Initialize session state
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def initialize_llm():
    """Initialize the ChatGroq LLM"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not groq_api_key:
        st.error("Error: GROQ_API_KEY not found in environment variables.")
        st.stop()
    
    if not google_api_key:
        st.error("Error: GOOGLE_API_KEY not found in environment variables.")
        st.stop()
    
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
        return llm
    except Exception as e:
        st.error(f"Error initializing ChatGroq: {e}")
        st.stop()

def create_onboarding_vectorstore(uploaded_files):
    """Create vector store from uploaded onboarding documents"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        all_docs = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["document_type"] = get_document_type(uploaded_file.name)
                
                all_docs.extend(docs)
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        if not all_docs:
            raise Exception("No documents were successfully loaded")
        
        # Split documents
        split_docs = text_splitter.split_documents(all_docs)
        
        # Create vector store
        vectordb = Chroma.from_documents(
            split_docs, 
            embeddings, 
            persist_directory="./onboarding_chroma_store"
        )
        
        return vectordb, len(all_docs)
        
    except Exception as e:
        raise Exception(f"Error creating vector store: {e}")

def get_document_type(filename):
    """Determine document type based on filename"""
    filename_lower = filename.lower()
    if 'hr' in filename_lower or 'policy' in filename_lower:
        return "HR Policy"
    elif 'posh' in filename_lower:
        return "POSH Policy"
    elif 'handbook' in filename_lower:
        return "Employee Handbook"
    elif 'benefit' in filename_lower:
        return "Benefits"
    elif 'code' in filename_lower and 'conduct' in filename_lower:
        return "Code of Conduct"
    elif 'security' in filename_lower:
        return "Security Policy"
    else:
        return "General Document"

def create_qa_chain(llm, vectordb):
    """Create the Q&A chain"""
    # Enhanced prompt for onboarding Q&A
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful Employee Onboarding Assistant. You help new employees understand company policies, procedures, and guidelines.

    <context>
    {context}
    </context>

    Based on the company documents provided above, please answer the employee's question accurately and helpfully.

    Guidelines:
    - Provide clear, concise answers based on the company documents
    - If you need to reference specific policies, mention the document name
    - If the information is not available in the documents, clearly state that
    - Be friendly and professional in your responses
    - For policy-related questions, provide both the rule and the rationale when available

    Employee Question: {input}

    Answer:
    """)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    return qa_chain

def main():
    st.title("🏢 Employee Onboarding Q&A System")
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload Onboarding Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload HR policies, POSH guidelines, employee handbook, etc."
        )
        
        if uploaded_files and not st.session_state.documents_uploaded:
            with st.spinner("Processing documents..."):
                try:
                    vectordb, doc_count = create_onboarding_vectorstore(uploaded_files)
                    st.session_state.vectordb = vectordb
                    st.session_state.documents_uploaded = True
                    
                    st.success(f"Successfully processed {doc_count} documents!")
                    
                    # Show uploaded documents
                    st.subheader("📋 Uploaded Documents:")
                    for file in uploaded_files:
                        doc_type = get_document_type(file.name)
                        st.write(f"• **{file.name}** ({doc_type})")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        
        elif st.session_state.documents_uploaded:
            st.success("Documents are ready!")
            
            if st.button("🔄 Upload New Documents"):
                st.session_state.vectordb = None
                st.session_state.documents_uploaded = False
                st.session_state.chat_history = []
                st.rerun()
    
    # Main chat interface
    if st.session_state.documents_uploaded and st.session_state.vectordb:
        
        # Create Q&A chain
        qa_chain = create_qa_chain(llm, st.session_state.vectordb)
        
        # Question input at the top
        question = st.text_input(
            "💬 Ask a question about company policies:",
            placeholder="e.g., What is the company's leave policy?",
            key="question_input"
        )
        
        if st.button("Ask Question", type="primary"):
            if question.strip():
                # Add user question to chat
                st.session_state.chat_history.append((question, ""))
                
                # Generate answer
                with st.spinner("Searching documents..."):
                    try:
                        result = qa_chain.invoke({"input": question})
                        answer = result["answer"]
                        
                        # Update the last entry with the answer
                        st.session_state.chat_history[-1] = (question, answer)
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {e}"
                        st.session_state.chat_history[-1] = (question, error_msg)
                
                # Rerun to refresh the interface
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📝 Q&A History")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                if a:  # Only show if there's an answer
                    with st.expander(f"Q{i+1}: {q[:50]}..." if len(q) > 50 else f"Q{i+1}: {q}"):
                        st.write("**Answer:**")
                        st.write(a)
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                • What is the leave policy?
                • How do I report misconduct?
                • What are the working hours?
                • What benefits do I get?
                """)
            with col2:
                st.markdown("""
                • What is the dress code?
                • How do I apply for reimbursement?
                • What are security guidelines?
                • Performance review process?
                """)
    
    else:
        # Welcome screen - simplified
        st.info("👈 Please upload your company documents using the sidebar to get started!")

if __name__ == "__main__":
    main()
