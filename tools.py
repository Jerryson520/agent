# tools.py
import os
from dotenv import load_dotenv
from typing import List, Dict, Union, Optional, Any
import tempfile
import uuid

from langchain_core.documents import Document
from langchain_core.tools import tool
# Tools: web_search, wiki_search, code_analysis, mathematical tools, Document processing tools, Image Loading and processing tools
from huggingface_hub import list_models


# Tools
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import ArxivLoader
import cmath
from urllib.parse import urlparse
import requests
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import pandas as pd

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
@tool
def save_path_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(content)
    
    return f"File saved to {filepath}. You can read this file to process its contents."


@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                file_name = f"download_{uuid.uuid4().hex[:8]}" 
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)
        
        # download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"File downloaded to {file_path}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    """
    try:
        # Open the image
        image = Image.open(image_path)

        text = pytesseract.image_to_string(image)
        
        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"
    
@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"
    
@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

### ============== IMAGE PROCESSING AND GENERATION TOOLS =============== ###
# @tool
# def analyze_image(image_base64: str) -> Dict[str: Any]:
#     """
#     Analyze basic properties of an image (size, mode, color analysis, thumbnail preview).
#     Args:
#         image_base64 (str): Base64 encoded image string
#     Returns:
#         Dictionary with analysis result
#     """
#     try:
#         img = decode_image(image_base64)
#         width, height = img.size
#         mode = img.mode
        
#         if mode in ("RGB")


if __name__ == "__main__":
    load_dotenv()
    
    query = "Help me find the latest AI agent papers."
    # print(wiki_search(query))
    # print(web_search.invoke({'query':query}))
    print(arxiv_search.invoke({'query':query}))
    