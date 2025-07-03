import os
import streamlit as st
import tempfile
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
    page_title="Document Q&A System",
    page_icon="💰",
    layout="centered"
)

# Initialize session state
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

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

def create_vectorstore(uploaded_files=None):
    """Create vector store from uploaded files"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Slightly larger chunks for better context
            chunk_overlap=300,  # More overlap to preserve policy continuity
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        all_docs = []
        
        # Process uploaded PDF files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    for i, doc in enumerate(docs):
                        doc.metadata.update({
                            "source": uploaded_file.name,
                            "document_type": "Uploaded Document",
                            "page_number": i + 1
                        })
                    
                    all_docs.extend(docs)
                finally:
                    os.unlink(tmp_file_path)
        
        if not all_docs:
            raise Exception("No documents were loaded")
        
        # Split documents
        split_docs = text_splitter.split_documents(all_docs)
        
        # Create vector store
        vectordb = Chroma.from_documents(
            split_docs, 
            embeddings, 
            persist_directory="./document_chroma_store"
        )
        
        return vectordb, len(all_docs)
        
    except Exception as e:
        raise Exception(f"Error creating vector store: {e}")

def create_qa_chain(llm, vectordb):
    """Create Q&A chain for any uploaded document"""
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert document analyst that provides comprehensive answers based ONLY on the provided document content. Your role is to help users understand all aspects of their uploaded documents, including policies, procedures, calculations, and specific scenarios.

    <context>
    {context}
    </context>

    IMPORTANT GUIDELINES:
    - Answer questions based ONLY on the content from the uploaded documents
    - For policy questions, explain both what happens in different scenarios (eligibility vs non-eligibility cases)
    - For calculation questions, provide step-by-step explanations with examples if available in the document
    - For scenario-based questions (like "what happens if..."), search for specific conditions, exceptions, and edge cases mentioned in the document
    - If specific information is not available in the documents, clearly state: "This specific scenario/information is not covered in the uploaded documents"
    - When discussing eligibility or requirements, mention both qualifying and non-qualifying conditions
    - Include relevant policy details, exceptions, and special circumstances when they exist in the document
    - For benefits/compensation questions, explain both entitlements and limitations
    - Provide clear, detailed answers that address the full scope of the question

    QUESTION HANDLING APPROACH:
    - For "what happens if..." questions: Look for specific scenarios, conditions, exceptions, and consequences
    - For eligibility questions: Detail both qualifying and disqualifying criteria
    - For calculation questions: Provide formulas, examples, and any limitations or special cases
    - For process questions: Explain procedures, timelines, requirements, and any variations
    - For policy questions: Cover rules, exceptions, special circumstances, and different scenarios

    Question: {input}

    Provide a comprehensive answer based on the uploaded document content, addressing all relevant aspects and scenarios mentioned in the documents:
    """)
    
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 8}  # Increased to get more relevant context
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    return qa_chain
def main():
    st.title("💰 Document Q&A System")
    st.markdown("Ask questions about your uploaded document")
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your document (PDF format)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to analyze and ask questions about"
        )
        
        if uploaded_files and not st.session_state.documents_loaded:
            if st.button("📚 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        vectordb, doc_count = create_vectorstore(uploaded_files=uploaded_files)
                        st.session_state.vectordb = vectordb
                        st.session_state.documents_loaded = True
                        st.success(f"✅ Processed {doc_count} document(s)!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif st.session_state.documents_loaded:
            st.success("📚 Documents ready!")
            if st.button("🔄 Upload New"):
                st.session_state.vectordb = None
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.rerun()
    
    # Main Q&A interface
    if st.session_state.documents_loaded and st.session_state.vectordb:
        
        # Question input
        question = st.text_input(
            "💬 Ask your question:",
            placeholder="e.g., What happens if I'm terminated before 5 years? How is gratuity calculated?",
            key="question_input"
        )
        
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                # Create Q&A chain
                qa_chain = create_qa_chain(llm, st.session_state.vectordb)
                
                # Add to chat history
                st.session_state.chat_history.append((question, ""))
                
                # Generate answer
                with st.spinner("🔍 Analyzing the document..."):
                    try:
                        result = qa_chain.invoke({"input": question})
                        answer = result["answer"]
                        
                        # Update chat history with answer
                        st.session_state.chat_history[-1] = (question, answer)
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {e}"
                        st.session_state.chat_history[-1] = (question, error_msg)
                
                # Clear input and rerun
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📝 Q&A History")
            
            for i, (q, a) in enumerate(st.session_state.chat_history):
                if a:  # Only show if there's an answer
                    with st.expander(f"Q{i+1}: {q[:80]}..." if len(q) > 80 else f"Q{i+1}: {q}"):
                        st.write("**Answer:**")
                        st.write(a)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
    
    else:
        # Simple message when no documents are loaded
        st.info("� Please upload a document using the sidebar to get started!")

if __name__ == "__main__":
    main()
