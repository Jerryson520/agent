# AI Agent System

A powerful AI agent system built with LangChain and LangGraph that combines RAG (Retrieval-Augmented Generation) capabilities with a suite of specialized tools for enhanced reasoning and task execution.

## Overview

This project implements an AI agent system that can:

- Answer questions using retrieval-augmented generation (RAG)
- Execute a variety of tools for web search, document processing, mathematical operations, and more
- Process and respond to queries through both API and web interface
- Utilize different LLM providers (Groq, Google Gemini, HuggingFace)

The system uses a graph-based workflow architecture powered by LangGraph to route queries through different processing nodes, enabling complex reasoning and tool use.

## Architecture

The system consists of several key components:

### Core Components

- **Workflow Engine**: Built with LangGraph, manages the flow of queries through different nodes
- **RAG System**: Uses Weaviate as a vector store for similarity search and context retrieval
- **LLM Provider**: Supports multiple LLM backends (Groq, Google Gemini, HuggingFace)
- **Tools System**: Provides a variety of tools for enhanced capabilities

### Workflow Nodes

- **RAG Assistant**: Retrieves relevant context from the vector store
- **Assistant**: Processes queries and determines if tools are needed
- **Tools**: Executes various tools based on the query requirements

### Tools

The system includes various tools:

- **Web Search**: Search the web using Tavily
- **Wikipedia Search**: Retrieve information from Wikipedia
- **arXiv Search**: Find research papers on arXiv
- **Mathematical Tools**: Perform calculations and mathematical operations
- **Document Processing**: Process CSV/Excel files, extract text from images, etc.

### Interfaces

- **FastAPI Backend**: Provides a REST API for interacting with the agent
- **Streamlit Frontend**: Offers a user-friendly web interface

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Jerryson520/agent.git
   cd agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following variables:
   ```
   PROJECT_ROOT=<path_to_project_root>
   WEAVIATE_URL=<your_weaviate_url>
   WEAVIATE_API_KEY=<your_weaviate_api_key>
   OPENAI_API_KEY=<your_openai_api_key>  # For embeddings
   TAVILY_API_KEY=<your_tavily_api_key>  # For web search
   ```

## Usage

### Running the FastAPI Server

```bash
uvicorn app.fastapi_app:app --reload
```

### Running the Streamlit Interface

```bash
streamlit run app/streamlit_app.py
```

### Using the Agent Programmatically

```python
from workflow import build_graph
from langchain_core.messages import HumanMessage

# Build the agent graph
graph = build_graph()

# Create a query
question = "What is the hometown of the 16th overall pick in the NBA draft in 2025?"
messages = [HumanMessage(content=question)]

# Get the response
response = graph.invoke({"messages": messages})

# Print the result
for m in response["messages"]:
    m.pretty_print()
```

## Configuration

### LLM Providers

The system supports multiple LLM providers that can be configured in `llm_provider.py`:

- **Groq**: Default provider using the qwen-qwq-32b model
- **Google Gemini**: Using the gemini-2.0-flash model
- **HuggingFace**: Using models like llama-2-7b-chat-hf

To change the provider, modify the `get_llm()` function call in `workflow.py`.

### Vector Store

The system uses Weaviate as a vector store for RAG capabilities. The vector store is configured in `vectorstore.py` and used in `retriever.py`.

## Data

The system uses a dataset stored in `data/metadata.jsonl` for the RAG system. This dataset contains questions, steps, and final answers that are embedded and stored in the Weaviate vector store.

## Development

### Adding New Tools

To add new tools, modify the `tools.py` file and add your tool using the `@tool` decorator from LangChain:

```python
@tool
def my_new_tool(param1: str, param2: int) -> str:
    """
    Description of what the tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of the return value
    """
    # Tool implementation
    return result
```

Then add your tool to the `tools` list in `workflow.py`.

### Modifying the Workflow

To modify the workflow, edit the `build_graph()` function in `workflow.py`. This function defines the nodes and edges of the workflow graph.

## License

[Specify the license here]

## Contributors

- [Jerryson520](https://github.com/Jerryson520)
