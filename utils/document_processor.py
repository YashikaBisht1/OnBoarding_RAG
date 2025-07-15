import json
import tempfile
import logging
import re
import os
from datetime import datetime
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import KNOWLEDGE_BASE_JSON, CHUNK_SIZE, CHUNK_OVERLAP, LOG_FILE, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def load_knowledge_base():
    """Load the knowledge base from JSON file"""
    try:
        with open(KNOWLEDGE_BASE_JSON, "r", encoding="utf-8") as f:
            kb = json.load(f)
            logging.info("Knowledge base loaded successfully")
            return kb
    except FileNotFoundError:
        logging.warning("Knowledge base file not found, returning empty dict")
        return {}
    except Exception as e:
        logging.error(f"Error loading knowledge base: {str(e)}")
        return {}

def save_knowledge_base(data: dict):
    """Save the knowledge base to JSON file"""
    try:
        with open(KNOWLEDGE_BASE_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info("Knowledge base saved successfully")
    except Exception as e:
        logging.error(f"Error saving knowledge base: {str(e)}")
        raise

def extract_meaningful_sentences(text: str) -> List[str]:
    """Extract meaningful sentences from text, handling various punctuation."""
    import re
    logging.info("Extracting meaningful sentences")
    
    # First clean up the text
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\f', ' ')  # Form feed
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    logging.debug(f"Cleaned text sample: {text[:200]}")
    
    # Split text into potential sentences using more lenient pattern
    raw_sentences = re.split(r'(?<=[.!?])\s*', text)
    
    # Clean and filter sentences
    sentences = []
    for sentence in raw_sentences:
        # Clean the sentence
        cleaned = sentence.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Replace multiple spaces with single
        
        # Filter out non-meaningful sentences
        if (len(cleaned) > 10 and  # Reasonable length
            any(char.isalpha() for char in cleaned) and  # Contains letters
            cleaned.count(' ') >= 2 and  # At least 3 words
            not cleaned.endswith('.pdf') and  # Not just a filename
            not cleaned.startswith('Figure') and  # Skip figure captions
            not cleaned.startswith('Table')):  # Skip table headers
            sentences.append(cleaned)
    
    logging.info(f"Extracted {len(sentences)} meaningful sentences")
    if not sentences:
        logging.warning("No meaningful sentences were extracted from the text!")
        
    return sentences

def add_document_to_knowledge_base(doc_name: str, doc_text: str):
    """Add a document to the knowledge base with line-by-line sentences"""
    logging.info(f"Adding document to knowledge base: {doc_name}")
    
    try:
        # Input validation
        if not doc_text or not doc_text.strip():
            raise ValueError("Empty document text provided")
            
        if len(doc_text.strip()) < 100:  # Arbitrary minimum length for a policy document
            raise ValueError(f"Document text too short ({len(doc_text)} chars)")
            
        kb = load_knowledge_base()
        
        # Determine document type
        doc_type = "gratuity" if "gratuity" in doc_name.lower() else "leave"
        logging.info(f"Document type determined: {doc_type}")
        
        # Log the document text length for debugging
        logging.info(f"Document text length: {len(doc_text)} characters")
        logging.info(f"Preview of first 200 chars: {doc_text[:200]}...")
        
        # Extract meaningful sentences with better preprocessing
        # Clean up common PDF artifacts
        cleaned_text = doc_text.replace('\x0c', '\n')  # Form feed
        cleaned_text = re.sub(r'\s*\n\s*', '\n', cleaned_text)  # Clean up newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Normalize paragraph breaks
        
        sentences = extract_meaningful_sentences(cleaned_text)
        
        # Validate we got meaningful content
        if not sentences:
            logging.error(f"No meaningful sentences extracted from {doc_name}!")
            logging.error(f"Cleaned text preview: {cleaned_text[:500]}")
            raise ValueError("No meaningful sentences could be extracted")
            
        # Additional validation of extracted sentences
        valid_sentences = []
        for sentence in sentences:
            # Skip likely headers, footers, and metadata
            if re.match(r'^(page|section|chapter|\d+|\s*$)', sentence.lower()):
                continue
            # Skip very short sentences that might be artifacts
            if len(sentence.split()) < 4:
                continue
            valid_sentences.append(sentence)
        
        if not valid_sentences:
            raise ValueError("No valid sentences after filtering")
            
        # Create simple numbered sentences list
        numbered_sentences = []
        for idx, sentence in enumerate(valid_sentences, 1):
            numbered_sentences.append({
                "line_no": idx,
                "text": sentence
            })
        
        # Store in knowledge base with minimal structure
        kb[doc_name] = {
            "doc_type": doc_type,
            "sentences": numbered_sentences
        }
        
        save_knowledge_base(kb)
        logging.info(f"Document {doc_name} added to knowledge base with {len(numbered_sentences)} valid sentences")
        return kb
        
    except Exception as e:
        logging.error(f"Error processing document {doc_name}: {str(e)}")
        raise

def process_pdf(uploaded_file, doc_type: str) -> Tuple[str, List[Document]]:
    """Process a PDF file and return its text content and document chunks"""
    logging.info(f"Processing PDF file: {uploaded_file.name}")
    
    # Validate input file
    if not uploaded_file or not hasattr(uploaded_file, 'name'):
        raise ValueError("Invalid file uploaded")
        
    if not uploaded_file.name.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")
    
    # Configure text splitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]  # Added spaces after punctuation
    )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

            # Try multiple PDF extraction methods
            combined_text = ""
            docs = []
            
            # First try PyPDF
            try:
                logging.info("Attempting PyPDF extraction...")
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                
                # Debug log the raw content
                for i, doc in enumerate(docs):
                    logging.debug(f"Page {i+1} content length: {len(doc.page_content)}")
                    logging.debug(f"Page {i+1} preview: {doc.page_content[:100]}...")
                
                # Verify we got actual content
                combined_text = "\n\n".join([doc.page_content for doc in docs])
                if not combined_text.strip():
                    raise ValueError("No text content extracted")
                    
                logging.debug(f"Raw extracted text sample: {combined_text[:200]}")
                
                logging.info(f"Successfully extracted {len(docs)} pages with PyPDF")
                
            except Exception as pdf_error:
                logging.warning(f"PyPDF extraction failed: {str(pdf_error)}")
                # Fallback to alternative extraction if needed
                try:
                    from pypdf import PdfReader
                    logging.info("Attempting direct PdfReader extraction...")
                    reader = PdfReader(tmp_file_path)
                    text_content = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(text)
                    
                    if not text_content:
                        raise ValueError("No text content extracted with PdfReader")
                    
                    combined_text = "\n\n".join(text_content)
                    docs = [Document(page_content=text, metadata={"page": i}) for i, text in enumerate(text_content)]
                    logging.info(f"Successfully extracted {len(docs)} pages with PdfReader")
                    
                except Exception as reader_error:
                    logging.error(f"All PDF extraction methods failed: {str(reader_error)}")
                    raise
            
            # Log some content for verification
            preview = combined_text[:200].replace('\n', ' ')
            logging.info(f"Content preview: {preview}...")
            
            # Validate the extracted text
            if not combined_text or len(combined_text.strip()) < 100:
                raise ValueError("Insufficient text extracted from PDF")
                
            # Split documents into chunks
            logging.info("Splitting document into chunks")
            split_docs = text_splitter.split_documents(docs)
            
            # Validate chunks
            if not split_docs:
                raise ValueError("Document splitting produced no chunks")
                
            # Log chunk information
            for i, doc in enumerate(split_docs):
                logging.debug(f"Chunk {i+1} size: {len(doc.page_content)} chars")
                logging.debug(f"Chunk {i+1} preview: {doc.page_content[:100]}...")
            
            # Add metadata
            for doc in split_docs:
                doc.metadata.update({
                    "source": uploaded_file.name,
                    "doc_type": doc_type,
                    "chunk_size": len(doc.page_content),
                    "extraction_date": datetime.now().isoformat(),
                    "chunk_word_count": len(doc.page_content.split())
                })
            
            logging.info(f"PDF processed successfully: {len(split_docs)} chunks created")
            return combined_text.strip(), split_docs
            
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

def process_pdf_from_path(file_path: str, doc_type: str) -> Tuple[str, List[Document]]:
    """Process a PDF file from file path and return its text content and document chunks"""
    logging.info(f"Processing PDF file from path: {file_path}")
    
    # Validate input file
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
        
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")
    
    # Configure text splitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]  # Added spaces after punctuation
    )
    
    try:
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Combine all pages into single text
        combined_text = ""
        for page in pages:
            page_text = page.page_content.strip()
            if page_text:
                combined_text += page_text + "\n\n"
        
        if not combined_text.strip():
            raise ValueError("No text content extracted from PDF")
        
        # Create Document objects for text splitting
        documents = [Document(
            page_content=combined_text,
            metadata={
                "source": os.path.basename(file_path),
                "doc_type": doc_type,
                "total_pages": len(pages),
                "extraction_date": datetime.now().isoformat()
            }
        )]
        
        # Split into chunks
        split_docs = text_splitter.split_documents(documents)
        
        if not split_docs:
            raise ValueError("Document splitting produced no chunks")
            
        # Log chunk information
        for i, doc in enumerate(split_docs):
            logging.debug(f"Chunk {i+1} size: {len(doc.page_content)} chars")
            logging.debug(f"Chunk {i+1} preview: {doc.page_content[:100]}...")
        
        # Add metadata
        for doc in split_docs:
            doc.metadata.update({
                "source": os.path.basename(file_path),
                "doc_type": doc_type,
                "chunk_size": len(doc.page_content),
                "extraction_date": datetime.now().isoformat(),
                "chunk_word_count": len(doc.page_content.split())
            })
        
        logging.info(f"PDF processed successfully: {len(split_docs)} chunks created")
        return combined_text.strip(), split_docs
        
    except Exception as e:
        error_msg = f"Error processing PDF from path: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
