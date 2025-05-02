import streamlit as st
import os
import datetime
from pathlib import Path
from chat_engine import get_response
from vector_store import add_documents
from prompt_tracker import check_prompt_limit, increment_prompt_count
from utils import create_initial_dirs

# Initialize the application
create_initial_dirs()

# Streamlit app configuration
st.set_page_config(
    page_title="Private GPT",
    page_icon="🤖",
    layout="wide"
)

# Simple authentication
def authenticate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        st.title("Private GPT Login")
        
        # In a real application, you'd use a more secure approach
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # For demo purposes, using a simple password
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        # Stop further execution if not authenticated
        if not st.session_state.authenticated:
            st.stop()
    
    return st.session_state.username

# Document uploader for feeding documents into the vector database
def document_uploader():
    st.sidebar.title("Document Management")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents for RAG", 
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"]
    )
    
    if uploaded_files and st.sidebar.button("Process Documents"):
        with st.sidebar.status("Processing documents..."):
            file_paths = []
            
            # Save uploaded files temporarily
            for file in uploaded_files:
                file_path = Path(f"data/documents/{file.name}")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                file_paths.append(str(file_path))
            
            # Process and add to vector store
            add_documents(file_paths)
            st.sidebar.success(f"Added {len(file_paths)} documents to knowledge base")

# Main application
def main():
    # Simple auth - can be disabled by commenting this line
    username = authenticate()
    
    st.title("🤖 Private GPT")
    
    # Sidebar with document uploader
    document_uploader()
    
    # Usage metrics
    remaining_prompts = 20 - check_prompt_limit(username)
    st.sidebar.metric("Remaining Prompts Today", remaining_prompts)
    
    # Display usage warning when approaching limit
    if remaining_prompts < 5:
        st.sidebar.warning(f"You only have {remaining_prompts} prompts left today!")
    
    # Reset time information
    now = datetime.datetime.now()
    midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), 
                                         datetime.time.min)
    reset_seconds = (midnight - now).total_seconds()
    hours, remainder = divmod(int(reset_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    st.sidebar.info(f"Prompt limit resets in: {hours}h {minutes}m")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input area
    if prompt := st.chat_input("Ask something..."):
        # Check if user has used their daily limit
        if check_prompt_limit(username) >= 20:
            st.error("You've reached your daily prompt limit. Please try again tomorrow.")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from model
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt, username)
                st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Increment the user's prompt count
        increment_prompt_count(username)

if __name__ == "__main__":
    main()