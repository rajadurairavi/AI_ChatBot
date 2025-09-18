#pip install langchain langchain-groq langchain-core streamlit
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
 
# this function builds a full LLM chain (LLM + prompt + memory)
def get_chain():
    # 1) connect to Groq LLM using API key from environment
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),  # never hardcode keys
        model_name="llama-3.1-8b-instant",             # Groq-hosted model
    )
 
    # 2) build a chat-style prompt (professional best practice)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise and factual."),  
        # system role sets the bot’s overall behavior
        MessagesPlaceholder(variable_name="history"),  
        # memory will inject past turns here automatically
        ("human", "{question}"),  
        # latest user input goes here
    ])
 
    # 3) attach LangChain memory (stores conversation turns)
    memory = ConversationBufferMemory(
        return_messages=True,   # keep chat messages in structured format
        memory_key="history"    # must match the placeholder above
    )
 
    # 4) final chain = llm + prompt + memory
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    return chain