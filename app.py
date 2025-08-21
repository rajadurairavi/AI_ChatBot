import streamlit as st
from llm_chain import get_chain
from langchain_core.messages import HumanMessage, AIMessage  # ADDED
 
st.title("Smart Chatbot")
 
# ADDED: init memory once
if "history" not in st.session_state:
    st.session_state.history = []
 
user_input = st.text_input("Please Enter Your Question Here")
search_button = st.button("Search")
 
if search_button:
    if user_input.strip() != "":
        chain = get_chain()
        with st.spinner("Thinking..."):
            # ADDED: pass history to the chain
            response = chain.invoke({
                "question": user_input,
                "history": st.session_state.history
            })
 
        st.write(response.content)
 
        # ADDED: update memory after each turn
        st.session_state.history.append(HumanMessage(content=user_input))
        st.session_state.history.append(AIMessage(content=response.content))
    else:
        st.warning("Please enter a question to get a response.")