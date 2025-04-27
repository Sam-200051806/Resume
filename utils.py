import streamlit as st
def format_response(response):
    return f"**Response:** {response}"

def handle_error(error_message):
    return f"**Error:** {error_message}"

def manage_session_state(key, value=None):
    if value is not None:
        st.session_state[key] = value
    return st.session_state.get(key, None)