#!/usr/bin/env python3
"""
Quick test to verify the multi-document system is working
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.qa_manager import QAManager

def test_multi_document_system():
    """Test the multi-document Q&A system"""
    print("🧪 Testing Multi-Document HR Policy Q&A System\n")
    
    # Initialize QA Manager
    qa_manager = QAManager()
    
    # Load vector stores
    loaded_dbs = qa_manager.vector_store.load_existing_stores()
    print("📚 Database Status:")
    for db_name, status in loaded_dbs.items():
        print(f"  {db_name.capitalize()}: {'✅' if status else '❌'}")
    
    if not any(loaded_dbs.values()):
        print("❌ No databases loaded. Please process documents first.")
        return
    
    # Test questions for each document type
    test_questions = [
        ("Gratuity", "What is the gratuity calculation formula?"),
        ("Leave", "How many annual leave days do I get?"),
        ("Upskilling", "What training opportunities are available?")
    ]
    
    print(f"\n🔍 Testing {len(test_questions)} queries:\n")
    
    for doc_type, question in test_questions:
        print(f"Q: {question}")
        try:
            docs, source = qa_manager.vector_store.get_relevant_documents(question)
            if docs:
                print(f"✅ Found {len(docs)} relevant documents from '{source}' source")
                # Preview first document
                preview = docs[0].page_content[:100] + "..." if len(docs[0].page_content) > 100 else docs[0].page_content
                print(f"   Preview: {preview}")
            else:
                print(f"❌ No relevant documents found")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        print("-" * 60)
    
    print("✅ Multi-document system test completed!")

if __name__ == "__main__":
    test_multi_document_system()
