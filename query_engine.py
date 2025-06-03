from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()

class QueryEngine:
    def __init__(self, index_name):
        self.index_name = index_name
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)
        

        self.llm = ChatGroq(
            model="llama3-3.1-8b-instant",
            temperature=0.2,
        )
        
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def query(self, user_input):
        response = self.retrieval_chain.run(user_input)
        return response


def query_resume(query, vector_store, chat_history=None, model="llama3-8b-8192", temperature=0.2, max_tokens=1000, resume_id=None):
    """
    Query the resume using the provided vector store and chat history
    """
    
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    
    system_template = """You are an AI assistant analyzing a resume. Your task is to answer questions about the candidate based on their resume.
    
    Answer questions ONLY based on the context provided below. Be specific and provide details from the resume.
    If information isn't explicitly mentioned in the context, say "I don't see that information in the resume."
    
    For common resume questions:
    - When asked about name, look at the top sections or contact information
    - When asked about skills, look for sections titled "Skills", "Technical Skills", or similar
    - When asked about experience, focus on work history and job descriptions
    - When asked about education, find degree information and educational institutions
    
    Resume Context:
    {context}
    """
    
    human_template = "{question}"
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])
    
    
    search_kwargs = {"k": 8}  
    

    if resume_id:
        try:
            search_kwargs["filter"] = {"resume_id": resume_id}
        except Exception as e:
            print(f"Warning: Could not apply filter with resume_id: {e}")
    
    
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(docs)} documents for query: {query}")
    
    
    if len(docs) > 0:
        # print("First chunk content preview:")
        pass
        # print(docs[0].page_content[:200] + "...")
    else:
        print("WARNING: No documents retrieved! Check your vector store.")
    
    context = "\n\n".join([doc.page_content for doc in docs])
    if chat_history and len(chat_history) > 0:
        history_context = "\nPrevious conversation:\n"
        for i in range(0, len(chat_history)-1, 2):
            if i+1 < len(chat_history):
                history_context += f"User: {chat_history[i]['content']}\n"
                history_context += f"Assistant: {chat_history[i+1]['content']}\n"
    
    chain = chat_prompt | llm
    response = chain.invoke({"context": context, "question": query})
    
    return response.content