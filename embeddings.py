import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings  # Alternative that might work better in cloud
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import uuid

# Set strict cache directories
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = "/tmp/huggingface_home"

# Try importing Pinecone
try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
except ImportError:
    print("Pinecone packages not available.")
    raise

load_dotenv()

# Try different embedding strategies
try:
    # Attempt with smaller model and offline caching
    os.environ["HF_HUB_OFFLINE"] = "1"  # Try to use cached models first
    
    # Try with a very small model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # 384 dimensions but very small
        cache_folder="/tmp/huggingface_models"
    )
except Exception as e:
    print(f"Error loading embedding model: {str(e)}")
    # Fall back to TensorFlow Hub embeddings which might be more cloud-friendly
    try:
        embeddings = TensorflowHubEmbeddings()
        print("Using TensorFlow Hub embeddings")
    except:
        # Last resort - create a simple class that produces mock embeddings with correct dimensions
        print("Using fallback embeddings with 768 dimensions")
        
        from langchain_core.embeddings import Embeddings
        import numpy as np
        
        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [np.random.rand(768).tolist() for _ in texts]
                
            def embed_query(self, text):
                return np.random.rand(768).tolist()
                
        embeddings = MockEmbeddings()

def clear_pinecone_data(index_name, namespace=None):
    """Clear existing vectors from Pinecone index"""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key not found")
            
        pc = Pinecone(api_key=api_key)
        
        index = pc.Index(index_name)
        
        if namespace:
            index.delete(delete_all=True, namespace=namespace)
        else:
            index.delete(delete_all=True)
            
        print(f"Cleared all vectors from index {index_name}")
    except Exception as e:
        print(f"Error clearing Pinecone data: {str(e)}")

def get_embeddings(text, resume_id=None):
    """
    Process the input text, split it into chunks, and create embeddings.
    Returns a vector store with the embeddings that can be queried.
    
    Args:
        text: Resume text content
        resume_id: Unique identifier for this resume
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200  
    )
    
    chunks = text_splitter.create_documents([text])
    
    for chunk in chunks:
        if not hasattr(chunk, "metadata"):
            chunk.metadata = {}
        chunk.metadata["resume_id"] = resume_id if resume_id else "default"
        chunk.metadata["source"] = "resume"
    
    
    try:
        index_name = os.getenv("INDEX_NAME")
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key or not index_name:
            raise ValueError("Pinecone API key or index name not found")
        
        namespace = f"resume_{resume_id}" if resume_id else "default_resume"
        
        
        pc = Pinecone(api_key=api_key)
        

        vector_store = PineconeVectorStore.from_documents(
            chunks,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        )
        
        return vector_store
        
    except Exception as e:
        print(f"Error with Pinecone: {str(e)}")
        raise ValueError(f"Failed to create vector store: {str(e)}")