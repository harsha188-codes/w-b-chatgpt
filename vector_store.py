import os
import logging
from typing import List, Optional
from pathlib import Path
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import necessary packages
try:
    from langchain.document_loaders import (
        TextLoader, 
        CSVLoader, 
        PyPDFLoader, 
        DirectoryLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain packages not found. Please install with: pip install langchain faiss-cpu pypdf pandas")

# Configuration
VECTOR_STORE_PATH = Path("faiss_index")
VECTOR_INDEX_PATH = VECTOR_STORE_PATH / "index.faiss"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # Smaller, faster model from HuggingFace
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize embeddings model
def get_embeddings():
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("Required packages not installed")
    
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# Initialize or load vector store
def get_vector_store():
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("Required packages not installed")
    
    # Create vector store directory if it doesn't exist
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    
    embeddings = get_embeddings()
    
    # Check if vector store exists
    if VECTOR_INDEX_PATH.exists():
        try:
            logger.info(f"Loading existing vector store from {VECTOR_INDEX_PATH}")
            return FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, "index")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.info("Creating new vector store")
            return FAISS.from_texts(["This is an initialization document."], embeddings)
    else:
        # Create a new empty vector store
        logger.info("Creating new vector store")
        return FAISS.from_texts(["This is an initialization document."], embeddings)

# Load and split documents
def process_documents(file_paths: List[str]):
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("Required packages not installed")
    
    documents = []
    
    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    for file_path in file_paths:
        try:
            file_path = Path(file_path)
            
            # Use appropriate loader based on file extension
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.csv':
                loader = CSVLoader(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path))
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                continue
                
            # Load and split the document
            loaded_docs = loader.load()
            split_docs = text_splitter.split_documents(loaded_docs)
            documents.extend(split_docs)
            
            logger.info(f"Processed {file_path.name}: {len(split_docs)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return documents

# Add documents to vector store
def add_documents(file_paths: List[str]):
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("Required packages not installed")
    
    try:
        # Process documents
        documents = process_documents(file_paths)
        
        if not documents:
            logger.warning("No documents were processed")
            return False
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Add documents to vector store
        vector_store.add_documents(documents)
        
        # Save updated vector store
        vector_store.save_local(str(VECTOR_STORE_PATH), "index")
        
        logger.info(f"Successfully added {len(documents)} chunks to vector store")
        return True
    
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {e}")
        return False

# Query vector store for relevant documents
def query_documents(query: str, k: int = 3):
    if not LANGCHAIN_AVAILABLE:
        logger.error("Required packages not installed")
        return []
    
    try:
        # Get vector store
        vector_store = get_vector_store()
        
        # Search for relevant documents
        results = vector_store.similarity_search(query, k=k)
        
        return results
    
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return []

# Test the vector store
if __name__ == "__main__":
    # Create a temporary text file for testing
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        f.write("This is a test document about Python programming. Python is a high-level, interpreted programming language.")
        test_file = f.name
    
    try:
        # Test adding documents
        add_documents([test_file])
        
        # Test querying
        results = query_documents("What is Python?")
        print(f"Query results: {results}")
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)