import json
import os 
from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain.schema import Document
from contextlib import contextmanager
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
load_dotenv()
root = os.getenv("PROJECT_ROOT")


# Build RAG
metadata_path = os.path.join(root, "data/metadata.jsonl")
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
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
)
try:
    db = WeaviateVectorStore.from_documents(
        documents, 
        embeddings, 
        client=weaviate_client,
        index_name="GAIA_val_vecdb")
finally:
    weaviate_client.close()
    
    
if __name__ == "__main__":
    pass