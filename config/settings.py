import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# File paths
KNOWLEDGE_BASE_JSON = str(DATA_DIR / "knowledge_base.json")
LOG_FILE = str(LOGS_DIR / "qa_log.txt")

# Vector DB paths
GRATUITY_DB_PATH = str(BASE_DIR / "document_chroma_store_gratuity")
LEAVE_DB_PATH = str(BASE_DIR / "document_chroma_store_leave")
UPSKILLING_DB_PATH = str(BASE_DIR / "document_chroma_store_upskilling")

# Text splitting settings
CHUNK_SIZE = 500  # Smaller chunks for better context
CHUNK_OVERLAP = 100  # Overlap to maintain context between chunks

# Model settings
LLM_MODEL = "gemma2-9b-it"  # Using Gemma model for chat
TEMPERATURE = 0.7  # Add temperature for more natural responses
MAX_TOKENS = 2048  # Maximum response length

# Retrieval settings
TOP_K_MATCHES = 3  # Number of similar documents to retrieve
SIMILARITY_THRESHOLD = -1.0  # Minimum similarity score for matches (allowing negative scores)

# Logging settings
LOG_LEVEL = "INFO"
ENABLE_DEBUG = False

# API Keys (from environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
