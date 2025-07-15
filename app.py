import streamlit as st
import os
from models.qa_manager import QAManager
from utils.document_processor import process_pdf, add_document_to_knowledge_base
from config.settings import GRATUITY_DB_PATH, LEAVE_DB_PATH, UPSKILLING_DB_PATH, DATA_DIR

def initialize_session_state():
    """Initialize session state variables"""
    if 'qa_manager' not in st.session_state:
        st.session_state.qa_manager = QAManager()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = set()
    if 'show_upload' not in st.session_state:
        st.session_state.show_upload = True

def handle_document_upload(uploaded_file, doc_type):
    """Handle document upload and processing"""
    try:
        # Check if file was already processed
        if uploaded_file.name in st.session_state.uploaded_files:
            st.warning(f"'{uploaded_file.name}' was already processed. Each file should only be uploaded once.")
            return

        # Extract text and create document chunks
        with st.status("Processing document...", expanded=True) as status:
            status.write("Extracting text from PDF...")
            combined_text, docs = process_pdf(uploaded_file, doc_type)
            
            status.write("Previewing extracted text...")
            with st.expander("📄 Text Preview", expanded=False):
                st.text_area("Extracted Content", combined_text, height=200)
            
            # Button to proceed with processing
            if st.button("✨ Process and Store Document", type="primary"):
                status.write("Adding to knowledge base...")
                add_document_to_knowledge_base(uploaded_file.name, combined_text)
                
                status.write("Creating vector store...")
                if "gratuity" in doc_type.lower():
                    store_path = GRATUITY_DB_PATH
                elif "leave" in doc_type.lower():
                    store_path = LEAVE_DB_PATH
                elif "upskilling" in doc_type.lower():
                    store_path = UPSKILLING_DB_PATH
                else:
                    store_path = LEAVE_DB_PATH  # Default fallback
                vectordb, doc_count = st.session_state.qa_manager.vector_store.create_vectorstore(docs, store_path)
                
                # Mark file as processed
                st.session_state.uploaded_files.add(uploaded_file.name)
                st.session_state.show_upload = False  # Hide upload section after processing
                
                status.update(label="✅ Document processed!", state="complete", expanded=False)
                st.success(f"Successfully processed {uploaded_file.name} into {doc_count} chunks!")
                st.rerun()

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        if os.getenv('DEBUG', 'false').lower() == 'true':
            st.exception(e)

def display_system_status():
    """Display the current system status"""
    loaded_dbs = st.session_state.qa_manager.vector_store.load_existing_stores()
    
    cols = st.columns(5)
    with cols[0]:
        st.write("📚 Gratuity DB:", "✅" if loaded_dbs['gratuity'] else "❌")
    with cols[1]:
        st.write("📚 Leave DB:", "✅" if loaded_dbs['leave'] else "❌")
    with cols[2]:
        st.write("📚 Upskilling DB:", "✅" if loaded_dbs['upskilling'] else "❌")
    with cols[3]:
        kb_exists = os.path.exists(os.path.join(DATA_DIR, "knowledge_base.json"))
        st.write("📖 Knowledge Base:", "✅" if kb_exists else "❌")
    with cols[4]:
        if st.button("📤 Show/Hide Upload Section"):
            st.session_state.show_upload = not st.session_state.show_upload
            st.rerun()

def main():
    st.set_page_config(
        page_title="HR Policy Q&A System",
        page_icon="💼",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header section
    st.title("💼 HR Policy Q&A System")
    
    # Status bar
    display_system_status()
    
    # Collapsible upload section
    if st.session_state.show_upload:
        with st.expander("📤 Document Upload", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                doc_type = st.selectbox(
                    "Select Document Type",
                    ["Gratuity Policy", "Leave Policy", "Upskilling Policy"],
                    help="Choose the type of policy document"
                )
            with col2:
                uploaded_file = st.file_uploader(
                    f"Upload {doc_type}",
                    type=["pdf"],
                    help="Upload a PDF document"
                )
            
            if uploaded_file:
                handle_document_upload(uploaded_file, doc_type)
    
    # Main Q&A interface
    st.header("💭 Ask Questions About HR Policies")
    
    # Question input with better styling
    question = st.text_input(
        "Your question:",
        placeholder="E.g., What are the eligibility criteria for gratuity?",
        help="Ask any question about the uploaded policies"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question!")
            else:
                with st.spinner("Finding answer..."):
                    try:
                        answer, source, context = st.session_state.qa_manager.get_answer(question)
                        
                        # Display answer in a nice card-like container
                        st.markdown("### Answer")
                        st.markdown(f"{answer}")
                        st.caption(f"Source: {source}")
                        
                        # Show context in expander
                        if context:
                            with st.expander("👀 View Context", expanded=False):
                                st.text(context)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "source": source
                        })
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
                        if os.getenv('DEBUG', 'false').lower() == 'true':
                            st.exception(e)
    
    # Chat history with better styling
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📝 Recent Questions")
        for chat in reversed(st.session_state.chat_history[-5:]):
            with st.container():
                st.write("🤔 " + chat['question'])
                st.write("💡 " + chat['answer'])
                st.caption(f"Source: {chat['source']}")
                st.markdown("---")

if __name__ == "__main__":
    main()
