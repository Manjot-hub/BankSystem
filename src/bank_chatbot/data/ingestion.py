"""
Document Ingestion Pipeline

Handles loading, chunking, and embedding of banking documents.
Supports both OpenAI embeddings and local sentence-transformers.
"""
import json
import hashlib
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.bank_chatbot.config.settings import get_settings


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: list[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = [
                "\n\n## ",      # Markdown headers
                "\n\n### ",
                "\n\n#### ",
                "\n\n",         # Paragraphs
                "\n",           # Lines
                ". ",           # Sentences
                " ",            # Words
                ""              # Characters
            ]


class LocalEmbeddings(Embeddings):
    """Local embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()


def get_embeddings(settings) -> Embeddings:
    """Get embeddings model based on available API keys."""
    if settings.OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        print(f"Using OpenAI embeddings: {settings.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    else:
        print("Using local sentence-transformers embeddings (all-MiniLM-L6-v2)")
        return LocalEmbeddings()


class DocumentIngestionPipeline:
    """Ingests and processes documents for RAG."""

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        self.settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )
        self.embeddings = get_embeddings(self.settings)

    def load_documents(self, data_dir: Path) -> list[Document]:
        """Load all JSON documents from data directory."""
        documents = []

        # Only process policies and FAQs for RAG
        # Transactions, users, accounts are handled via tool-calling
        target_files = ["policies.json", "faqs.json"]

        for json_file in data_dir.glob("*.json"):
            if json_file.name not in target_files:
                continue

            with open(json_file) as f:
                data = json.load(f)

            for item in data:
                doc_type = json_file.stem  # policies, faqs, etc.
                content = self._format_document(item, doc_type)
                metadata = self._extract_metadata(item, doc_type)

                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)

        return documents

    def _format_document(self, item: dict, doc_type: str) -> str:
        """Format document for embedding."""
        if doc_type == "policies":
            return f"POLICY: {item['title']}\n\n{item['content']}"
        elif doc_type == "faqs":
            return f"FAQ: {item['title']}\n\n{item['content']}"
        else:
            return json.dumps(item, indent=2)

    def _extract_metadata(self, item: dict, doc_type: str) -> dict[str, Any]:
        """Extract metadata for filtering and retrieval."""
        base_meta = {
            "doc_id": item.get("doc_id", ""),
            "doc_type": doc_type,
            "title": item.get("title", ""),
            "source": doc_type,
        }

        if doc_type == "policies":
            base_meta.update({
                "product": item.get("product", "all"),
                "jurisdiction": item.get("jurisdiction", "US"),
                "version": item.get("version", ""),
                "effective_date": item.get("effective_date", ""),
                "topic": item.get("metadata", {}).get("topic", ""),
                "keywords": item.get("metadata", {}).get("keywords", []),
            })
        elif doc_type == "faqs":
            base_meta.update({
                "product": item.get("product", "all"),
                "topic": item.get("metadata", {}).get("topic", ""),
                "keywords": item.get("metadata", {}).get("keywords", []),
            })

        return base_meta

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks with metadata preservation."""
        chunks = self.splitter.split_documents(documents)

        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = hashlib.md5(
                chunk.page_content.encode()
            ).hexdigest()[:12]
            chunk.metadata["chunk_index"] = i
            chunk.metadata["ingested_at"] = datetime.now().isoformat()

        return chunks

    def embed_chunks(self, chunks: list[Document]) -> list[list[float]]:
        """Generate embeddings for chunks."""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings

    def process(self, data_dir: Path) -> tuple[list[Document], list[list[float]]]:
        """Full ingestion pipeline."""
        print(f"Loading documents from {data_dir}...")
        documents = self.load_documents(data_dir)
        print(f"Loaded {len(documents)} documents")

        print("Chunking documents...")
        chunks = self.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print("Generating embeddings...")
        embeddings = self.embed_chunks(chunks)
        print(f"Generated {len(embeddings)} embeddings (dim: {len(embeddings[0]) if embeddings else 0})")

        return chunks, embeddings


def main():
    pipeline = DocumentIngestionPipeline()
    data_dir = Path("data/synthetic")
    chunks, embeddings = pipeline.process(data_dir)

    # Save for inspection
    output = {
        "chunks": [
            {
                "content": c.page_content,
                "metadata": c.metadata
            }
            for c in chunks
        ],
        "embeddings_shape": [len(embeddings), len(embeddings[0]) if embeddings else 0]
    }

    with open(data_dir / "processed_chunks.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved processed chunks to {data_dir / 'processed_chunks.json'}")


if __name__ == "__main__":
    main()