#!/usr/bin/env python3

import os
import sys
import time
import pytest
import re
import warnings

# Suppress the UserWarning about relevance scores
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

# Add the project root to the path (parent directory of Main_app)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.qa_manager import QAManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory and all dependencies are installed")
    sys.exit(1)

@pytest.fixture(scope="module")
def qa_manager():
    """Initialize QA Manager for testing"""
    print("Initializing QA Manager...")
    manager = QAManager()
    
    if not manager.initialize_llm():
        pytest.fail("Error: Could not initialize LLM - check GROQ_API_KEY")
    
    return manager

@pytest.fixture(scope="module")
def test_questions():
    """Predefined test questions for HR policy Q&A"""
    return [
        "What is the gratuity policy?",
        "How many days of annual leave am I entitled to?",
        "What is the notice period for resignation?",
        "Can I encash my leave?",
        "What are the maternity leave benefits?",
        "How is gratuity calculated?",
        "What is the sick leave policy?",
        "Can I carry forward unused leave?",
        "What documents are required for leave application?",
        "What is the maximum gratuity amount?",
        "What is the annual leave quota?",
        "Up to how many optional leaves one can avail?",
        "What is the gratuity contribution calculation?",
        "What are the upskilling opportunities available?",
        "How can I apply for training programs?",
        "What is the learning and development policy?",
        "Are there any certifications the company supports?",
        "What budget is allocated for employee learning?",
        "How often can I attend training sessions?"
    ]

def validate_hr_answer(question, answer, source):
    """Validate HR policy answer quality and relevance"""
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Check for question-specific content
    if "gratuity" in question_lower:
        assert any(keyword in answer_lower for keyword in ["gratuity", "4.81%", "5 years", "basic salary"]), \
            f"Gratuity answer should contain relevant terms. Answer: {answer[:100]}..."
    
    elif "annual leave" in question_lower or "leave quota" in question_lower:
        assert any(keyword in answer_lower for keyword in ["18 days", "annual", "leave", "yearly"]), \
            f"Annual leave answer should mention leave entitlement. Answer: {answer[:100]}..."
    
    elif "maternity" in question_lower:
        assert any(keyword in answer_lower for keyword in ["26 weeks", "maternity", "childbirth", "female"]), \
            f"Maternity answer should contain maternity leave details. Answer: {answer[:100]}..."
    
    elif "carry forward" in question_lower:
        assert any(keyword in answer_lower for keyword in ["9 days", "carry", "forward", "next year"]), \
            f"Carry forward answer should mention carry forward limits. Answer: {answer[:100]}..."
    
    elif "optional" in question_lower:
        assert any(keyword in answer_lower for keyword in ["3", "optional", "holidays"]), \
            f"Optional leave answer should mention optional holidays. Answer: {answer[:100]}..."
    
    elif "upskilling" in question_lower or "training" in question_lower or "learning" in question_lower:
        assert any(keyword in answer_lower for keyword in ["learning", "training", "development", "skill", "course"]), \
            f"Upskilling answer should contain learning/training terms. Answer: {answer[:100]}..."
    
    elif "certification" in question_lower:
        assert any(keyword in answer_lower for keyword in ["certification", "certificate", "accredited", "professional"]), \
            f"Certification answer should mention certification details. Answer: {answer[:100]}..."
    
    elif "encash" in question_lower:
        assert any(keyword in answer_lower for keyword in ["encash", "no encashment", "not encashed"]), \
            f"Encashment answer should address leave encashment policy. Answer: {answer[:100]}..."

def validate_response_quality(answer, source, response_time):
    """Validate overall response quality"""
    # Basic validation
    assert answer is not None, "Answer should not be None"
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer.strip()) > 0, "Answer should not be empty"
    assert source is not None, "Source should not be None"
    
    # Quality checks
    assert len(answer.strip()) >= 20, f"Answer too short ({len(answer)} chars): {answer}"
    assert not answer.strip().startswith("Error"), f"Answer indicates error: {answer[:100]}..."
    assert "unable to" not in answer.lower(), f"Answer indicates inability to respond: {answer[:100]}..."
    
    # Performance checks
    assert response_time < 30, f"Response took too long: {response_time:.2f} seconds"
    
    # Source validation
    valid_sources = ["Direct Match", "Vector Search", "LLM with Full Context"]
    assert any(valid_source in source for valid_source in valid_sources), \
        f"Source should indicate retrieval method: {source}"

def validate_numerical_answers(question, answer):
    """Validate numerical information in answers"""
    question_lower = question.lower()
    
    if "how many" in question_lower or "quota" in question_lower:
        # Should contain numbers
        numbers = re.findall(r'\d+', answer)
        assert len(numbers) > 0, f"Numerical question should contain numbers. Answer: {answer[:100]}..."
    
    if "gratuity" in question_lower and "calculation" in question_lower:
        # Should mention percentage or calculation method
        assert any(keyword in answer.lower() for keyword in ["4.81%", "percentage", "basic salary", "calculate"]), \
            f"Gratuity calculation should mention calculation method. Answer: {answer[:100]}..."

@pytest.mark.parametrize("question", [
    "What is the gratuity policy?",
    "How many days of annual leave am I entitled to?", 
    "What is the notice period for resignation?",
    "Can I encash my leave?",
    "What are the maternity leave benefits?",
    "How is gratuity calculated?",
    "What is the sick leave policy?",
    "Can I carry forward unused leave?",
    "What documents are required for leave application?",
    "What is the maximum gratuity amount?",
    "What is the annual leave quota?",
    "Up to how many optional leaves one can avail?",
    "What is the gratuity contribution calculation?"
])
def test_hr_policy_questions(qa_manager, question):
    """Test HR policy Q&A system and return structured Q&A"""
    try:
        result = qa_manager.get_answer(question)
        
        # Handle both tuple and single return formats
        if isinstance(result, tuple):
            answer = result[0]  # First element is the answer
        else:
            answer = result
        
        # Basic validation
        assert answer is not None, "Answer should not be None"
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer.strip()) > 0, "Answer should not be empty"
        
        # Clean up the answer - remove \n tags and extra whitespace
        clean_answer = answer.replace('\\n', ' ').replace('\n', ' ').strip()
        clean_answer = ' '.join(clean_answer.split())  # Remove extra spaces
        
        # Print structured Q&A
        print(f"\nQ: {question}")
        print(f"A: {clean_answer}")
        print("-" * 60)
        
    except Exception as e:
        pytest.fail(f"Error getting answer: {str(e)}")

def test_all_questions_summary(qa_manager, test_questions):
    """Test all questions and display structured Q&A format"""
    print("\n" + "="*60)
    print("HR POLICY Q&A - STRUCTURED OUTPUT")
    print("="*60)
    
    successful_queries = 0
    failed_queries = 0
    
    for i, question in enumerate(test_questions, 1):
        try:
            result = qa_manager.get_answer(question)
            
            # Handle both tuple and single return formats
            if isinstance(result, tuple):
                answer = result[0]  # First element is the answer
            else:
                answer = result
            
            # Basic validation
            assert answer is not None and isinstance(answer, str) and len(answer.strip()) > 0
            
            # Clean up the answer - remove \n tags and extra whitespace
            clean_answer = answer.replace('\\n', ' ').replace('\n', ' ').strip()
            clean_answer = ' '.join(clean_answer.split())  # Remove extra spaces
            
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {clean_answer}")
            print("-" * 60)
            
            successful_queries += 1
            
        except Exception as e:
            print(f"\nQ{i}: {question}")
            print(f"A{i}: Error - {str(e)}")
            print("-" * 60)
            failed_queries += 1
    
    # Final assertion
    assert failed_queries == 0, f"{failed_queries} out of {len(test_questions)} queries failed"

if __name__ == "__main__":
    # Run pytest programmatically if executed directly
    pytest.main([__file__, "-v", "-s"])