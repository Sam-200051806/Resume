from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import uuid

# Try importing Pinecone with new API
try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone  # Updated import
except ImportError:
    print("Pinecone packages not available. Please install with 'pip install pinecone langchain-pinecone'.")
    raise

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
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