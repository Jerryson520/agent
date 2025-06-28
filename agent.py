import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json

import cmath
import pandas as pd
import numpy as np

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools import Tool

from tools import (
        web_search, wiki_search, arxiv_search, 
        operate, square_root, save_path_read_file, 
        download_file_from_url, extract_text_from_image, 
        analyze_csv_file, analyze_excel_file,
    )
from retriever import retriever
load_dotenv()
root = os.getenv("PROJECT_ROOT")
    
def get_llm(provider: str = "groq"):
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        # Groq https://console.groq.com/docs/models
        return ChatGroq(model="qwen-qwq-32b", temperature=0) # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        # TODO: Add huggingface endpoint
        return ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")

# Node
def assistant(state: MessagesState):
    """Assistant node"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def retrieve_assistant(state: MessagesState):
    """Retriever Node"""
    similar_question = retriever(state["messages"][0].content) 
    
    if similar_question:  # Check if the list is not empty
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}
    else:
        # Handle the case when no similar questions are found
        return {"messages": [sys_msg] + state["messages"]}
    

# load the system prompt from the file
with open(os.path.join(root, "prompts/system_prompt.txt"), "r", encoding="utf-8") as f:
    system_prompt = f.read()
# System message
sys_msg = SystemMessage(content=system_prompt)

tools = [
    web_search,
    wiki_search,
    arxiv_search,
    operate,
    square_root, 
    save_path_read_file, 
    download_file_from_url, 
    extract_text_from_image, 
    analyze_csv_file, 
    analyze_excel_file,
]

llm = get_llm()

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
    

# Build graph function
def build_graph():
    """Build the graph"""
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("retrieve_assistant", retrieve_assistant)
    builder.add_node("tools", ToolNode(tools))

    # add edges
    builder.add_edge(START, "retrieve_assistant")
    builder.add_edge("retrieve_assistant", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()


# test
if __name__ == "__main__":
    # question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    question = "What is the hometown of this year’s 16th overall pick in the NBA draft?"
    graph = build_graph(provider="groq")
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()