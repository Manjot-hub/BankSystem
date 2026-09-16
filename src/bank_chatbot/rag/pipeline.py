"""
RAG Pipeline with LangGraph

Implements the RAG workflow: Query -> Retrieve -> Generate -> Guardrails -> Response
"""
from typing import Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.rag.vector_store import ChromaVectorStore, build_vector_store
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
        """Get LLM for generation."""
        if self.settings.OPENAI_API_KEY:
            return ChatOpenAI(
                model=self.settings.LLM_MODEL_PRIMARY,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.LLM_MAX_TOKENS,
                openai_api_key=self.settings.OPENAI_API_KEY,
            )
        else:
            # Return None for retrieval-only mode
            return None

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph RAG workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("guardrails", self._guardrails)
        workflow.add_node("format_response", self._format_response)

        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "guardrails")
        workflow.add_edge("guardrails", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        query = state["query"]

        # Build filter based on user context (optional)
        filter_dict = {}
        # Could add: user segment, product, jurisdiction filters here

        docs = self.vector_store.similarity_search(query, k=5, filter=filter_dict if filter_dict else None)

        return {
            **state,
            "retrieved_docs": docs,
        }

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

        # Build context with citations
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
                score=1 - doc.metadata.get("_distance", 0)
            ))

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer if LLM available
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
            # Retrieval-only mode: return context summary
            answer = f"Based on {len(docs)} retrieved documents:\n\n" + "\n\n".join([
                f"Source {i+1} ({c.title}): {c.snippet}"
                for i, c in enumerate(citations)
            ])
            answer += "\n\n[Retrieval-only mode: Set OPENAI_API_KEY for full generation]"

        # Calculate confidence based on retrieval scores
        avg_score = sum(c.score for c in citations) / len(citations) if citations else 0

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

        # Check for forbidden patterns
        forbidden = [
            "guaranteed return", "risk-free", "investment advice",
            "tax advice", "legal advice", "definitely", "always"
        ]

        for pattern in forbidden:
            if pattern.lower() in answer.lower():
                flags.append(f"forbidden_phrase:{pattern}")

        # Check if answer contains citations
        if state["citations"] and "[Source" not in answer:
            flags.append("missing_citations")

        # Check confidence threshold
        if state["confidence"] < 0.5:
            flags.append("low_confidence")

        return {
            **state,
            "guardrail_flags": flags,
        }

    def _format_response(self, state: RAGState) -> RAGState:
        """Format final response."""
        answer = state["answer"]
        flags = state["guardrail_flags"]

        # Add disclaimer if needed
        if "low_confidence" in flags or not state["citations"]:
            answer += "\n\nNote: This information is for general guidance only. Please verify with your account details or contact support for specific advice."

        return {
            **state,
            "answer": answer,
        }

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


def test_rag():
    """Test the RAG pipeline."""
    pipeline = RAGPipeline()

    test_queries = [
        "What is the funds availability policy for check deposits?",
        "How do I report a lost debit card?",
        "What are the wire transfer fees?",
        "Can I get a loan with no credit check?",  # Should trigger guardrails
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = pipeline.invoke(query)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Citations: {len(result['citations'])}")
        print(f"Flags: {result['guardrail_flags']}")


if __name__ == "__main__":
    test_rag()