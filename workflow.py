import os
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import json
import cmath
import pandas as pd
import numpy as np

from config import root, sys_msg
from state import AgentState

from llm_provider import get_llm
from langchain_core.messages import BaseMessage
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


# Node
def assistant(state: AgentState):
    """Assistant node"""
    result = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [result],
    }


def retrieve_assistant(state: AgentState):
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
llm_with_tools = llm.bind_tools(tools)
    

# Build graph function
def build_graph():
    """Build the graph"""
    builder = StateGraph(AgentState)
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
    # question = "What is the hometown of the 16th overall pick in the NBA draft in 2025?"
    # question = "Who was the 16th overall pick in 2025 NBA Draft and where is he from?"
    question = "Who is the only Chinese NBA draft 1st pick and where is he from?"
    graph = build_graph()
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()