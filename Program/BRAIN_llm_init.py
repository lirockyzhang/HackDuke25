import os
from langchain_groq import ChatGroq

# ———— API Key ————
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ———— Exported LLMs ————
llama_slow = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=1024,
    api_key=GROQ_API_KEY,
)

llama_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=10,
    api_key=GROQ_API_KEY,
)

__all__ = ["llama_slow", "llama_fast"]