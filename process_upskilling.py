#!/usr/bin/env python3
"""
Script to process the upskilling document and create vector store
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.document_processor import process_pdf_from_path, add_document_to_knowledge_base
from models.vector_store import VectorStoreManager
from config.settings import UPSKILLING_DB_PATH, BASE_DIR

def main():
    """Process the upskilling document"""
    
    # Path to the upskilling document
    upskilling_pdf_path = BASE_DIR / "Documents_Policies" / "Upskilling_and_Continuous_Learning_Policy.pdf"
    
    if not upskilling_pdf_path.exists():
        print(f"Error: Upskilling document not found at {upskilling_pdf_path}")
        return
    
    print("Processing Upskilling and Continuous Learning Policy...")
    
    try:
        # Extract text and create document chunks
        print("1. Extracting text from PDF...")
        combined_text, docs = process_pdf_from_path(str(upskilling_pdf_path), "upskilling")
        
        print(f"   Extracted {len(combined_text)} characters of text")
        print(f"   Created {len(docs)} document chunks")
        
        # Add to knowledge base
        print("2. Adding to knowledge base...")
        add_document_to_knowledge_base("Upskilling_and_Continuous_Learning_Policy.pdf", combined_text)
        
        # Create vector store
        print("3. Creating vector store...")
        vector_store = VectorStoreManager()
        vectordb, doc_count = vector_store.create_vectorstore(docs, UPSKILLING_DB_PATH)
        
        print(f"✅ Successfully processed upskilling document!")
        print(f"   Created vector store with {doc_count} chunks")
        print(f"   Vector store saved to: {UPSKILLING_DB_PATH}")
        
        # Test the vector store
        print("\n4. Testing vector store...")
        docs_found, source = vector_store.get_relevant_documents("What are the upskilling opportunities?")
        if docs_found:
            print(f"   Found {len(docs_found)} relevant documents from source: {source}")
        else:
            print("   No relevant documents found in test query")
            
    except Exception as e:
        print(f"Error processing upskilling document: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
