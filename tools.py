# tools.py
import os
from dotenv import load_dotenv
from typing import List, Dict, Union


from langchain_core.documents import Document
from langchain_core.tools import tool
# Tools: web_search, wiki_search, code_analysis, mathematical tools, Document processing tools, Image Loading and processing tools
from huggingface_hub import list_models


# Tools
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import ArxivLoader
import cmath

# =============== BROWSER TOOLS =============== 
@tool
def wiki_search(query: str,
                max_docs: int = 2,
                char_limit: int = 1_000,
                ) -> Dict[str, List[Dict]]:
    """
    Fetch up to ``max_docs`` Wikipedia pages that match *query* and return a
    JSON-serialisable dict with title, url and a truncated page snippet.

    Args:
        query: Search string.
        max_docs: How many Wikipedia pages to load (default 2).
        char_limit: Maximum characters per snippet (default 1 000).
    """
    loader = WikipediaLoader(
        query, 
        load_max_docs=max_docs,
    )
        
    response: List[Document] = loader.load()
    
    results = [
        {
            "title": d.metadata.get("title", "Unknown"), 
            "url": d.metadata.get("source", "N/A"),
            "snippet": d.page_content[:char_limit] + "…" if len(d.page_content) > char_limit else d.page_content,
        }
        for d in response
    ]
    return {"query": query, "results": results}

@tool
def web_search(query: str,
               max_results: int = 3,
               include_answer: bool = False,
               include_images: bool = False) -> Dict[str, List[Dict]]:
    """
    Search Tavily and return structured JSON (titles, urls, snippets).

    Args:
        query:  The search query
        max_results:  How many results to fetch (default 3)
        include_answer / include_images:  Pass-through flags
    """
    tavily_tool = TavilySearch(
        max_results=max_results,
        include_answer=include_answer,
        include_images=include_images,
        include_raw_content=False,  # usually not needed
        search_depth="basic"
    )
    response = tavily_tool.invoke({"query": query})

    results = [
        {
            "title": item["title"],
            "url": item["url"],
            "snippet": item["content"]      # or item.get("raw_content")
        }
        for item in response["results"]
    ]

    # return a JSON-serialisable object (StructuredTool will serialise for you)
    return {"query": response["query"], "results": results}


@tool
def arxiv_search(query: str,
                 max_results: int = 3,
                ) -> Dict[str, List[Dict]]:
    """
    Search arXiv for papers matching a query and return structured results.

    Args:
        query (str): The search query string, following arXiv API syntax
            (e.g., "quantum computing" or an arXiv ID like "0710.5765v1").
        max_results (int, optional): Maximum number of papers to retrieve.
            Defaults to 3. The API allows up to ~300 000 but typical usage
            is much smaller.
    """
    loader = ArxivLoader(
        query=query,
        load_max_docs=max_results
    )
    response: List[Document] = loader.load()
    results = [
        {
            "publish_date": doc.metadata.get("Published", "Unknown"),
            "Title": doc.metadata.get("Title", "Unknown"),
            "Authors": doc.metadata.get("Authors", "Unknown"),
            "Summary": doc.metadata.get("Summary", "Unknown"),
        }
        for doc in response
    ]
    
    return {"query": query, "results": results}


### =============== CODE INTERPRETER TOOLS =============== ###



### =============== MATHEMATICAL TOOLS =============== ###
@tool
def operate(
    operand: str,
    a: float, 
    b: float,
) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Supported operations:
        - 'add': a + b
        - 'subtract': a - b
        - 'multiply': a * b
        - 'divide': a / b
        - 'modulus': a % b
        - 'power': a ** b

    Args:
        operand (str): The operation to perform.
        a (float): The first operand.
        b (float): The second operand.

    Returns:
        float: The result of the arithmetic operation.

    Raises:
        ZeroDivisionError: If dividing or taking modulus by zero.
        ValueError: If the operand is not supported.
    """
    if operand == "add":
        return a + b
    elif operand == "subtract":
        return a - b
    elif operand == "multiply":
        return a * b
    elif operand == "divide":
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return a / b
    elif operand == "modulus":
        if b == 0:
            raise ValueError("Cannot take modulus by zero.")
        return a % b
    elif operand == "power":
        return a ** b
    else:
        raise ValueError(f"Unsupported operand: {operand}")
    
    
@tool
def square_root(a: float) -> Union[float, complex]:
    """
    Compute the square root of a number, supporting both real and complex results.

    Args:
        a (float or complex): The number to compute the square root of. 
            If a is negative, the result will be a complex number.

    Returns:
        complex: The square root of the input number. Always returned as a complex type.
    """
    if a >= 0:
        return a**0.5
    return cmath.sqrt(a)

### =============== DOCUMENT PROCESSING TOOLS =============== ###



### ============== IMAGE PROCESSING AND GENERATION TOOLS =============== ###

if __name__ == "__main__":
    load_dotenv()
    
    query = "Help me find the latest AI agent papers."
    # print(wiki_search(query))
    # print(web_search.invoke({'query':query}))
    print(arxiv_search.invoke({'query':query}))
    