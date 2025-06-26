# Basic Import
import json
import os 
from dotenv import load_dotenv
# Import Tools
from langchain_core.tools import tool

@tool
def question_retriever(query: str) -> str:
    """A tool to retrieve similar questions or similar reasoning steps from a vector store."""
    
    
    