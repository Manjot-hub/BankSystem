"""
RAG Pipeline with LangGraph

Implements the RAG workflow: Query -> Retrieve -> Generate -> Guardrails -> Response
Supports Groq LLM primary execution with OpenAI fallback.
"""
import os
from typing import Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.rag.vector_store import ChromaVectorStore
from src.bank_chatbot.data.ingestion import get_embeddings


class RAGState(TypedDict):
    """State for the RAG pipeline."""
    query: str
    user_id: str
    session_id: str
    retrieved_docs: list[Document]
    context: str
    answer: str
    citations: list[dict]
    confidence: float
    guardrail_flags: list[str]
    error: str | None


class Citation(BaseModel):
    """Citation for a retrieved document."""
    doc_id: str
    chunk_id: str
    title: str
    snippet: str
    score: float


class RAGPipeline:
    """RAG Pipeline using LangGraph."""

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = get_embeddings(self.settings)
        self.vector_store = self._get_vector_store()
        self.llm = self._get_llm()
        self.graph = self._build_graph()

    def _get_vector_store(self) -> ChromaVectorStore:
        """Get or build vector store."""
        from src.bank_chatbot.rag.vector_store import build_vector_store
        return build_vector_store()

    def _get_llm(self):
        """Get LLM for generation (Groq primary, OpenAI fallback)."""
        groq_api_key = getattr(self.settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                print(f"Initializing Groq LLM: {getattr(self.settings, 'LLM_MODEL_PRIMARY', 'llama-3.3-70b-versatile')}")
                return ChatGroq(
                    model_name=getattr(self.settings, "LLM_MODEL_PRIMARY", "llama-3.3-70b-versatile"),
                    groq_api_key=groq_api_key,
                    temperature=getattr(self.settings, "LLM_TEMPERATURE", 0.1),
                    max_tokens=getattr(self.settings, "LLM_MAX_TOKENS", 2000),
                )
            except Exception as e:
                print(f"Warning: Failed to initialize ChatGroq: {e}")

        openai_api_key = getattr(self.settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            from langchain_openai import ChatOpenAI
            print(f"Initializing OpenAI LLM: {getattr(self.settings, 'LLM_MODEL_PRIMARY', 'gpt-4o-mini')}")
            return ChatOpenAI(
                model=getattr(self.settings, "LLM_MODEL_PRIMARY", "gpt-4o-mini"),
                temperature=getattr(self.settings, "LLM_TEMPERATURE", 0.1),
                max_tokens=getattr(self.settings, "LLM_MAX_TOKENS", 2000),
                openai_api_key=openai_api_key,
            )

        print("No valid API Key found for LLM generation. Running in retrieval-only mode.")
        return None

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph RAG workflow."""
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("guardrails", self._guardrails)
        workflow.add_node("format_response", self._format_response)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "guardrails")
        workflow.add_edge("guardrails", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        query = state["query"]
        docs = self.vector_store.similarity_search(query, k=5)
        return {**state, "retrieved_docs": docs}

    def _generate(self, state: RAGState) -> RAGState:
        """Generate answer from retrieved context."""
        query = state["query"]
        docs = state["retrieved_docs"]

        if not docs:
            return {
                **state,
                "answer": "I don't have enough information to answer that question. Please contact customer support for assistance.",
                "citations": [],
                "confidence": 0.0,
            }

        context_parts = []
        citations = []

        for i, doc in enumerate(docs):
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
            doc_id = doc.metadata.get("doc_id", "unknown")
            title = doc.metadata.get("title", "Unknown")
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content

            context_parts.append(f"[Source {i+1}: {title} (ID: {doc_id}:{chunk_id})]\n{doc.page_content}")
            citations.append(Citation(
                doc_id=doc_id,
                chunk_id=chunk_id,
                title=title,
                snippet=snippet,
                score=1.0 - doc.metadata.get("_distance", 0.0)
            ))

        context = "\n\n---\n\n".join(context_parts)

        if self.llm:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a banking customer support assistant. Answer ONLY from the provided context.
Rules:
1. Only use information from the provided sources
2. Cite sources using [Source X] format
3. If information is insufficient, say "I don't know" and suggest contacting support
4. Never provide financial advice
5. Be concise and professional
6. Include specific details from sources (fees, limits, policies)"""),
                ("human", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"),
            ])

            chain = prompt | self.llm
            response = chain.invoke({"context": context, "query": query})
            answer = response.content
        else:
            answer = f"Based on {len(docs)} retrieved documents:\n\n" + "\n\n".join([
                f"Source {i+1} ({c.title}): {c.snippet}" for i, c in enumerate(citations)
            ])
            answer += "\n\n[Retrieval-only mode: Set GROQ_API_KEY for full LLM response generation]"

        avg_score = sum(c.score for c in citations) / len(citations) if citations else 0.0

        return {
            **state,
            "context": context,
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "confidence": avg_score,
        }

    def _guardrails(self, state: RAGState) -> RAGState:
        """Apply guardrails to the generated answer."""
        flags = []
        answer = state["answer"]

        forbidden = [
            "guaranteed return", "risk-free", "investment advice",
            "tax advice", "legal advice", "definitely", "always"
        ]

        for pattern in forbidden:
            if pattern.lower() in answer.lower():
                flags.append(f"forbidden_phrase:{pattern}")

        if state["citations"] and "[Source" not in answer:
            flags.append("missing_citations")

        if state["confidence"] < 0.5:
            flags.append("low_confidence")

        return {**state, "guardrail_flags": flags}

    def _format_response(self, state: RAGState) -> RAGState:
        """Format final response."""
        answer = state["answer"]
        flags = state["guardrail_flags"]

        if "low_confidence" in flags or not state["citations"]:
            answer += "\n\nNote: This information is for general guidance only. Please verify with your account details or contact support for specific advice."

        return {**state, "answer": answer}

    def invoke(self, query: str, user_id: str = "anonymous", session_id: str = "default") -> dict[str, Any]:
        """Invoke the RAG pipeline."""
        initial_state = RAGState(
            query=query,
            user_id=user_id,
            session_id=session_id,
            retrieved_docs=[],
            context="",
            answer="",
            citations=[],
            confidence=0.0,
            guardrail_flags=[],
            error=None,
        )

        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial_state, config=config)

        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "guardrail_flags": result["guardrail_flags"],
            "retrieved_count": len(result["retrieved_docs"]),
            "retrieved_docs": result["retrieved_docs"],
        }