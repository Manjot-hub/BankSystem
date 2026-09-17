"""
Data Ingestion and Embedding Factory Module.

Provides document loading, preprocessing, and factory functions 
for generating text embeddings (OpenAI or Local HuggingFace).
"""
import os

# Limit BLAS/OMP thread memory allocation to prevent Windows OpenBLAS crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from typing import Any
from langchain_core.embeddings import Embeddings

# Safe import for local Hugging Face embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None


def get_embeddings(settings: Any) -> Embeddings:
    """
    Factory function to instantiate the appropriate Embedding model 
    based on application settings and available API keys.
    """
    model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openai_key = getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

    # Use Hugging Face if explicitly requested or if no OpenAI API Key is provided (e.g. in CI environments)
    if "all-MiniLM" in model_name or "sentence-transformers" in model_name or not openai_key:
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "HuggingFace Embeddings library is missing. "
                "Install it using: pip install -U langchain-huggingface sentence-transformers"
            )
        
        print(f"Loading local Hugging Face embeddings: sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            multi_process=False  # Prevents excessive virtual memory paging on Windows
        )

    # Use OpenAI Embeddings if model requires it and API key is present
    print(f"Using OpenAI embeddings: {model_name}")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=openai_key
    )