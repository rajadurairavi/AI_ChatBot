from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
 
def get_chain():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )
 
    # History-aware prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise and factual."),
        MessagesPlaceholder(variable_name="history"),  # memory goes here
        ("human", "{question}")
    ])
 
    chain = prompt | llm
    return chain
 