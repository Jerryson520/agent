import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import json

import cmath
import pandas as pd
import numpy as np

from langgraph.graph import START, StateGraph, MessagesState
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader

# Import VecDB 
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth
from langchain_openai import OpenAIEmbeddings

from langgraph.prebuilt import ToolNode, tools_condition

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from tools import web_search, wiki_search, arxiv_search, operate
load_dotenv()


# Build RAG
metadata_path = "./data/metadata.jsonl"
documents = []
with open(metadata_path, "r", encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        questions = item['Question']
        steps = item['Annotator Metadata'].get('Steps', "")
        final_answer = item['Final answer']            
        content = f"{questions}\n\n{steps}\n\nFinal Answer: {final_answer}"
        # contents.append(content)
        metadata = {
            "task_id": item.get("task_id", ""),
            "level": item.get("Level", ""),
            "final_answer": item.get("Final answer", ""),
            "num_steps": item["Annotator Metadata"].get("Number of steps", ""),
            "tools": item["Annotator Metadata"].get("Tools", ""),
            "num_tools": item["Annotator Metadata"].get("Number of tools", ""),
            "time_taken": item["Annotator Metadata"].get("How long did this take?"),
        }
        documents.append(Document(page_content=content, metadata=metadata))

embeddings = OpenAIEmbeddings()
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
)
db = WeaviateVectorStore.from_documents(documents, embeddings, client=weaviate_client)


tools = [
    web_search,
    wiki_search,
    arxiv_search,
    operate,
]

# Build graph function
def build_graph(provider: str = "groq"):
    """Build the graph"""
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        # Groq https://console.groq.com/docs/models
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0) # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        # TODO: Add huggingface endpoint
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()



# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    graph = build_graph(provider="groq")
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()