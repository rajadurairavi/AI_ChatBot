import streamlit as st
from llm_chain import get_chain
 
st.title("Smart Chatbot")
 
# keep the same chain (and its memory) across button clicks
# streamlit reruns the script every click, so without this it forgets
if "chain" not in st.session_state:
    st.session_state.chain = get_chain()
 
# input box + button
user_input = st.text_input("Enter your question")
search_button = st.button("Search")
 
# handle button click
if search_button:
    if user_input.strip():
        chain = st.session_state.chain  # reuse same chain every time
        response = chain.invoke({"question": user_input})
        # chain returns {"text": "...answer..."}
        st.write(response["text"])
    else:
        st.warning("Please enter a question to get a response.")