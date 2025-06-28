import json
import os 
from dotenv import load_dotenv
from langchain.schema import Document
from contextlib import contextmanager
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
import streamlit as st

@st.cache_resource
def get_vectorstore():
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

vectorstore = get_vectorstore()

def retriever(query: str) -> str:
    return vectorstore.similarity_search(query)

if __name__ == "__main__":
    pass