import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing different model options
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not found. Falling back to local models.")

try:
    from langchain.llms import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False
    logger.warning("GPT4All package not found.")

# Import vector store for context retrieval
from vector_store import query_documents

# Configuration (in real app, load from .env or config file)
MODEL_CONFIG = {
    "use_openai": False,  # Set to True to use OpenAI, False for local model
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "openai_model": "gpt-3.5-turbo",
    "local_model_path": "models/ggml-gpt4all-j-v1.3-groovy.bin",  # Download from https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin
    "temperature": 0.7,
    "max_tokens": 500
}

# The number of context documents to retrieve for RAG
NUM_CONTEXT_DOCS = 3

# Load a local LLM using GPT4All
def load_local_model():
    if not GPT4ALL_AVAILABLE:
        raise ImportError("GPT4All is not installed. Please run: pip install gpt4all")
    
    model_path = Path(MODEL_CONFIG["local_model_path"])
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"Local model not found at {model_path}. Please download it first.")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    try:
        model = GPT4All(model=str(model_path), 
                       max_tokens=MODEL_CONFIG["max_tokens"],
                       temp=MODEL_CONFIG["temperature"])
        return model
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        raise

# Initialize OpenAI client
def get_openai_client():
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not installed. Please run: pip install openai")
    
    if not MODEL_CONFIG["openai_api_key"]:
        raise ValueError("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
    
    # Initialize client
    client = openai.OpenAI(api_key=MODEL_CONFIG["openai_api_key"])
    return client

# Generate response using OpenAI
def get_openai_response(prompt: str) -> str:
    client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_CONFIG["openai_model"],
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=MODEL_CONFIG["max_tokens"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# Generate response using local model
def get_local_model_response(prompt: str) -> str:
    try:
        model = load_local_model()
        response = model.generate(prompt, max_tokens=MODEL_CONFIG["max_tokens"])
        return response.strip()
    except Exception as e:
        logger.error(f"Local model error: {e}")
        # Fallback to a simple response if model fails
        return f"Sorry, I encountered an error with the local model: {str(e)}"

# Build enhanced prompt with context
def build_rag_prompt(query: str, username: str) -> str:
    # Retrieve relevant documents for the query
    context_docs = query_documents(query, k=NUM_CONTEXT_DOCS)
    
    # If no relevant documents found
    if not context_docs:
        return f"Question: {query}\nAnswer: "
    
    # Build context string from retrieved documents
    context_str = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create RAG prompt with context
    rag_prompt = f"""You are a helpful AI assistant. Answer the question based on the context provided below.
If the answer cannot be determined from the context, say "I don't have enough information to answer that."

Context:
{context_str}

Question: {query}

Answer: """
    
    return rag_prompt

# Main response generation function
def get_response(query: str, username: str) -> str:
    try:
        # Build RAG-enhanced prompt
        enhanced_prompt = build_rag_prompt(query, username)
        
        # Generate response based on configuration
        if MODEL_CONFIG["use_openai"]:
            response = get_openai_response(enhanced_prompt)
        else:
            response = get_local_model_response(enhanced_prompt)
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I'm having trouble generating a response right now. Please try again later."

# Function to test the chat engine directly
if __name__ == "__main__":
    test_query = "What are the key features of Python?"
    response = get_response(test_query, "test_user")
    print(f"Query: {test_query}")
    print(f"Response: {response}")