"""
Vector Store Manager

Handles ChromaDB operations for document storage and retrieval.
"""
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.bank_chatbot.config.settings import get_settings


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store with metadata filtering support."""

    def __init__(self, embeddings: Embeddings, persist_dir: str, collection_name: str):
        self._embeddings = embeddings
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.metadata.get("chunk_id", f"doc_{i}") for i, doc in enumerate(documents)]

        embeddings = self.embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] = None
    ) -> list[Document]:
        """Search for similar documents with optional metadata filter."""
        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        if results["documents"]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                metadata["_distance"] = distance
                documents.append(Document(page_content=doc, metadata=metadata))

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] = None
    ) -> list[tuple[Document, float]]:
        """Search with similarity scores."""
        docs = self.similarity_search(query, k, filter)
        return [(doc, 1 - doc.metadata.get("_distance", 0)) for doc in docs]

    def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        self.collection.delete(ids=ids)

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        count = self.collection.count()
        return {"count": count, "name": self.collection.name}

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: list[dict] = None,
        **kwargs
    ) -> "ChromaVectorStore":
        """Create vector store from texts."""
        persist_dir = kwargs.get("persist_dir", "./data/chroma_db")
        collection_name = kwargs.get("collection_name", "banking_knowledge")

        store = cls(embeddings, persist_dir, collection_name)
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas or [{}] * len(texts))
        ]
        store.add_documents(documents)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embeddings: Embeddings,
        persist_dir: str,
        collection_name: str
    ) -> "ChromaVectorStore":
        """Create vector store from documents."""
        store = cls(embeddings, persist_dir, collection_name)
        store.add_documents(documents)
        return store


def load_processed_chunks(data_dir: Path) -> list[Document]:
    """Load processed chunks from JSON file."""
    with open(data_dir / "processed_chunks.json") as f:
        data = json.load(f)

    documents = []
    for chunk in data["chunks"]:
        doc = Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)

    return documents


def build_vector_store(data_dir: Path = None) -> ChromaVectorStore:
    """Build vector store from processed chunks."""
    from src.bank_chatbot.data.ingestion import get_embeddings
    from src.bank_chatbot.config.settings import get_settings

    if data_dir is None:
        data_dir = Path("data/synthetic")

    settings = get_settings()
    embeddings = get_embeddings(settings)

    print("Loading processed chunks...")
    documents = load_processed_chunks(data_dir)
    print(f"Loaded {len(documents)} chunks")

    print("Building vector store...")
    vector_store = ChromaVectorStore.from_documents(
        documents=documents,
        embeddings=embeddings,
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.CHROMA_COLLECTION_NAME
    )

    stats = vector_store.get_collection_stats()
    print(f"Vector store built: {stats}")

    return vector_store


if __name__ == "__main__":
    build_vector_store()