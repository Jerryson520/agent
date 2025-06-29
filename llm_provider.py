from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def get_llm(provider: str = "groq"):
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        # Groq https://console.groq.com/docs/models
        return ChatGroq(model="qwen-qwq-32b", temperature=0) # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        # TODO: Add huggingface endpoint
        return ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")