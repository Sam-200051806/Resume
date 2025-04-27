import streamlit as st
from document_loader import load_resume
from embeddings import get_embeddings
from query_engine import query_resume
from typing import Set
import hashlib
from datetime import datetime
import os 
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

if (
    "chat_answers_history" not in st.session_state
    or "user_prompt_history" not in st.session_state
    or "chat_history" not in st.session_state
    or "resume_vector_store" not in st.session_state
    or "resume_hashes" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state["resume_vector_store"] = None
    st.session_state["resume_hashes"] = {}  


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

def compute_file_hash(file_content):
    """Compute a hash for the file content to identify duplicate uploads"""
    return hashlib.md5(file_content).hexdigest()

def process_resume(uploaded_file, force_refresh=False):
    """Process a resume and store in vector DB, with option to force refresh"""
    
    file_content = uploaded_file.getvalue()
    file_hash = compute_file_hash(file_content)
    
  
    is_new_upload = file_hash not in st.session_state["resume_hashes"]
    
    if is_new_upload or force_refresh:
        with st.spinner("Processing resume..."):
            resume_id = str(file_hash)
            
            if not is_new_upload and force_refresh:
                st.info("Refreshing existing resume data...")
                try:
                    old_namespace = f"resume_{resume_id}"
                    
                    from embeddings import clear_pinecone_data
                    index_name = os.getenv("INDEX_NAME")
                    clear_pinecone_data(index_name, old_namespace)
                except Exception as e:
                    st.warning(f"Could not clear old data: {str(e)}")
            
            resume_text = load_resume(uploaded_file)
            if not resume_text or len(resume_text) < 10:
                st.error("Could not extract text from this PDF. Please try another file.")
                return None
                
            st.success(f"Resume loaded successfully! {len(resume_text)} characters extracted.")

            if st.checkbox("Show extracted resume text"):
                st.text_area("Extracted text", resume_text, height=200)
            
            vector_store = get_embeddings(resume_text, resume_id=resume_id)
            
            st.session_state["resume_vector_store"] = vector_store
            st.session_state["resume_hashes"][file_hash] = {
                "name": uploaded_file.name, 
                "vector_store": vector_store,
                "resume_id": resume_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return vector_store, resume_id
    else:
       
        st.success("Resume recognized! Using existing vector data.")
        st.session_state["resume_vector_store"] = st.session_state["resume_hashes"][file_hash]["vector_store"]
        resume_id = st.session_state["resume_hashes"][file_hash]["resume_id"]
        return st.session_state["resume_vector_store"], resume_id

def main():
    st.title("Resume Analyzer")
    
    with st.sidebar:
        st.title("⚙️ Settings")
        
        model = st.selectbox(
            "Select Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        max_tokens = st.slider(
            "Max Response Length",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum number of tokens in the response"
        )
        
        st.subheader("Display Options")
        show_sources = st.toggle("Show Sources", value=True)
        
        if st.button("Clear Conversation", type="primary"):
            st.session_state["chat_answers_history"] = []
            st.session_state["user_prompt_history"] = []
            st.session_state["chat_history"] = []
            st.session_state["last_query"] = ""
            st.rerun()
        
        st.divider()
        st.caption("About")
        st.markdown("""
        This app uses LangChain and Retrieval Augmented Generation to answer 
        questions about your resume.
        """)

    st.write("Upload your resume in PDF format to get insights.")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        force_refresh = False
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Force Refresh"):
                force_refresh = True
        

        result = process_resume(uploaded_file, force_refresh)
        
        if result:
            vector_store, resume_id = result

            st.header("Ask about this resume")

            if st.session_state["chat_answers_history"]:
                for generated_response, user_query in zip(
                    st.session_state["chat_answers_history"],
                    st.session_state["user_prompt_history"]
                ):
                    st.chat_message("user").write(user_query)
                    st.chat_message("assistant").write(generated_response)
                    
            with st.form(key="query_form", clear_on_submit=True):
                user_query = st.text_input("Ask a question about the resume:", key="query_input")
                submit_button = st.form_submit_button("Submit")

                if submit_button and user_query:
                    with st.spinner("Analyzing resume..."):
                        try:
                            response = query_resume(
                                query=user_query, 
                                vector_store=st.session_state["resume_vector_store"],
                                chat_history=st.session_state["chat_history"],
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                resume_id=resume_id
                            )
                            
                            st.session_state["user_prompt_history"].append(user_query)
                            st.session_state["chat_answers_history"].append(response)
                            st.session_state["chat_history"].append({"type": "human", "content": user_query})
                            st.session_state["chat_history"].append({"type": "ai", "content": response})
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()