import json
import os 
from dotenv import load_dotenv
from langchain.schema import Document
from contextlib import contextmanager
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
)

vectorstore = WeaviateVectorStore(
    client=weaviate_client,
    embedding=embeddings,
    index_name="GAIA_val_vecdb",
    text_key='text'
)

def retriever(query: str) -> str:
    try:
        return vectorstore.similarity_search(query)
    finally:
        weaviate_client.close()


if __name__ == "__name__":
    pass