import json
import os 
from dotenv import load_dotenv
from langchain.schema import Document
from contextlib import contextmanager
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import streamlit as st
from llm_provider import get_llm

@st.cache_resource
def _get_vectorstore():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    embeddings = OpenAIEmbeddings()
    
    return WeaviateVectorStore(
        client=client,
        embedding=embeddings,
        index_name="GAIA_val_vecdb",
        text_key='text'
    )

vectorstore = _get_vectorstore()

def retriever(query: str) -> str:
    return vectorstore.similarity_search(query)


llm = get_llm()

def generator(query: str, context: str) -> str:
    """LLM-based generation given query and retrieved context"""
    messages = [
        SystemMessage(
            content="Here is a similar context for reference, and try to plan your own logic for the question."
        ),
        HumanMessage(content=f"Context:\n{context}"),
        HumanMessage(content=f"Question:\n{query}")
    ]
    return llm.invoke(messages).content

if __name__ == "__main__":
    pass