import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import uuid

# Set cache directories for cloud environment
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = "/tmp/huggingface_home"

# Try importing Pinecone with new API
try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
except ImportError:
    print("Pinecone packages not available. Please install with 'pip install pinecone langchain-pinecone'.")
    raise

load_dotenv()

# Try loading models that produce 768-dimensional vectors to match Pinecone
try:
    # First try a smaller 768-dimension model (compatible with your Pinecone index)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-albert-small-v2"  # 768 dimensions
    )
    print("Successfully loaded smaller 768-dimension embedding model")
except Exception as e:
    print(f"Error loading first model: {str(e)}")
    try:
        # Fall back to the original model if needed
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
        )
        print("Successfully loaded original embedding model")
    except Exception as e2:
        # Last resort - distilbert which also produces 768 dimensions
        print(f"Trying last resort model: {str(e2)}")
        from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityHFEmbeddings
        embeddings = CommunityHFEmbeddings(
            model_name="distilbert-base-uncased"  # Also 768 dimensions
        )

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