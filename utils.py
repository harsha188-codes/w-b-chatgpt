import os
import logging
from pathlib import Path
from prompt_tracker import initialize_storage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory structure
DIRS = [
    Path("data/documents"),
    Path("faiss_index"),
    Path("models")
]

def create_initial_dirs():
    """
    Create all required directories for the application if they don't exist
    """
    for dir_path in DIRS:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"Error creating directory {dir_path}: {e}")
    
    # Initialize storage for user prompt tracking
    initialize_storage()

def get_available_models():
    """
    Check for available local models in the models directory
    """
    model_dir = Path("models")
    model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.gguf"))
    return [model.name for model in model_files]

def check_dependencies():
    """
    Check if all required packages are installed
    """
    missing_packages = []
    
    # Essential packages
    try:
        import streamlit
    except ImportError:
        missing_packages.append("streamlit")
    
    # Vector store packages
    try:
        import faiss
    except ImportError:
        missing_packages.append("faiss-cpu")
    
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        missing_packages.append("langchain")
    
    # Document processing
    try:
        import PyPDF2
    except ImportError:
        missing_packages.append("pypdf2")
    
    # Embedding models
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing_packages.append("sentence-transformers")
    
    # Local LLM options
    local_llm_found = False
    
    try:
        from langchain.llms import GPT4All
        local_llm_found = True
    except ImportError:
        pass
    
    try:
        from llama_cpp import Llama
        local_llm_found = True
    except ImportError:
        pass
    
    if not local_llm_found:
        missing_packages.append("gpt4all or llama-cpp-python")
    
    return missing_packages

def download_sample_model(model_name="ggml-gpt4all-j-v1.3-groovy.bin"):
    """
    Utility function to download a sample model for testing
    Note: In a real app, you would want to show progress and handle this better
    """
    import requests
    from tqdm import tqdm
    
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / model_name
    
    # Skip if model already exists
    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
        return str(model_path)
    
    # GPT4All model URLs
    model_urls = {
        "ggml-gpt4all-j-v1.3-groovy.bin": "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
    }
    
    if model_name not in model_urls:
        logger.error(f"Model {model_name} not found in available models")
        return None
    
    url = model_urls[model_name]
    
    try:
        logger.info(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Model downloaded to {model_path}")
        return str(model_path)
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

# Run tests if executed directly
if __name__ == "__main__":
    # Create directories
    create_initial_dirs()
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
    else:
        print("All dependencies are installed!")
    
    # List available models
    models = get_available_models()
    if models:
        print(f"Available models: {', '.join(models)}")
    else:
        print("No models found. You'll need to download a model to use the local LLM feature.")