import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun

from langchain.agents import create_agent

load_dotenv()

st.title("LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your GROQ API key:", type="password"
)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm a chatbot who can search the web. How can i help you?"}
    ]

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Get user prompt
prompt = st.chat_input("Ask your question...")

if prompt:
    # Save user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Define LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        streaming=True,
    )

    # Tools
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    )
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    )
    search = DuckDuckGoSearchRun(name="Search")

    tools = [search, arxiv, wiki]

    # Build agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant that can use tools to answer queries.",
    )

    # Run agent with .invoke(...) using the chat messages
    response = agent.invoke(
        {"messages": st.session_state.messages}
    )

    # Get the final assistant output
    if "messages" in response and len(response["messages"]) > 0:
        output = response["messages"][-1].content
    else:
        output = "No response generated."

    st.write(output)
    st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": output})
