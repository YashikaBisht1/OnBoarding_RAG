# OnBoarding RAG

A Retrieval-Augmented Generation (RAG) system for HR policy question-answering, supporting multi-document search and interactive Q&A. This project helps organizations onboard new employees and answer HR-related queries efficiently using AI and document-based knowledge.

## Features
- **Multi-Document Q&A:** Handles questions across multiple HR policy documents (gratuity, leave, upskilling, etc.)
- **Intelligent Source Selection:** Automatically selects the most relevant document for each query
- **Streamlit Web Interface:** User-friendly UI for real-time Q&A and document uploads
- **Extensible Test Suite:** Easily add or modify test questions and documents
- **Source Attribution:** Shows which document provided each answer

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation
```bash
# Clone the repository
https://github.com/yashika-005/OnBoarding_RAG.git
cd OnBoarding_RAG

# Install dependencies
pip install -r requirements.txt
```

### Usage
#### Run the Test Suite
```bash
pytest test_simple_qa.py -v -s
# OR
python test_simple_qa.py
```

#### Run the Web Application
```bash
streamlit run app.py
```
- Open your browser to `http://localhost:8501`
- Upload HR policy documents and ask questions interactively

## File Structure
- `app.py` — Streamlit web application
- `test_simple_qa.py` — Main test script
- `models/` — Q&A logic and models
- `utils/` — Document processing utilities
- `Documents_Policies/` — Folder for HR policy documents
- `requirements.txt` — Python dependencies
## Demo
<img width="1071" height="830" alt="Screenshot 2025-07-14 123350" src="https://github.com/user-attachments/assets/a4a64231-f8fb-4a5f-be7f-e809329210de" />
<img width="1506" height="731" alt="Screenshot 2025-07-09 124427" src="https://github.com/user-attachments/assets/7ac56832-7887-4f72-a398-3842c3d6fd33" />


