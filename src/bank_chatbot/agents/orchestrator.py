"""
Multi-Agent Orchestration

Implements hierarchical multi-agent architecture for the banking chatbot.
Supervisor routes queries to specialized agents:
- FAQ/RAG Agent
- Transaction Agent
- Fraud/Dispute Agent
- Escalation Agent
- Compliance Agent
"""
from typing import Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.rag.pipeline import RAGPipeline
from src.bank_chatbot.agents.tool_agent import ToolCallingAgent
from src.bank_chatbot.guardrails.guardrails import BankingGuardrails


class MultiAgentState(TypedDict):
    """State for the multi-agent supervisor."""
    query: str
    user_id: str
    session_id: str
    intent: str
    routed_agent: str
    agent_response: str
    guardrail_flags: list[str]
    error: str | None


class MultiAgentOrchestrator:
    """Hierarchical multi-agent orchestrator."""

    def __init__(self):
        self.settings = get_settings()
        self.rag_agent = RAGPipeline()
        self.transaction_agent = ToolCallingAgent()
        self.guardrails = BankingGuardrails()
        self.graph = self._build_graph()

    def _classify_intent(self, query: str) -> str:
        """Classify query intent for routing."""
        lower = query.lower()

        # High-risk intents
        if any(term in lower for term in ["fraud", "unauthorized", "stolen", "hacked", "compromised"]):
            return "fraud_agent"

        # Transaction intents
        if any(term in lower for term in ["balance", "transaction", "transfer", "payment", "spend", "account"]):
            return "transaction_agent"

        # Policy/FAQ intents
        if any(term in lower for term in ["fee", "policy", "deposit", "card", "wire", "ach", "overdraft", "atm", "password", "direct deposit", "funds availability", "limit"]):
            return "rag_agent"

        # Default to escalation
        return "escalation_agent"

    def _build_graph(self) -> StateGraph:
        """Build the multi-agent graph."""
        workflow = StateGraph(MultiAgentState)

        workflow.add_node("classify", self._classify)
        workflow.add_node("route", self._route)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("transaction_agent", self._transaction_agent)
        workflow.add_node("fraud_agent", self._fraud_agent)
        workflow.add_node("escalation_agent", self._escalation_agent)
        workflow.add_node("compliance_agent", self._compliance_agent)

        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "route")

        workflow.add_conditional_edges(
            "route",
            lambda state: state["routed_agent"],
            {
                "rag_agent": "rag_agent",
                "transaction_agent": "transaction_agent",
                "fraud_agent": "fraud_agent",
                "escalation_agent": "escalation_agent",
            },
        )

        workflow.add_edge("rag_agent", "compliance_agent")
        workflow.add_edge("transaction_agent", "compliance_agent")
        workflow.add_edge("fraud_agent", "compliance_agent")
        workflow.add_edge("escalation_agent", "compliance_agent")
        workflow.add_edge("compliance_agent", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _classify(self, state: MultiAgentState) -> MultiAgentState:
        """Classify the query intent."""
        intent = self._classify_intent(state["query"])
        return {**state, "intent": intent}

    def _route(self, state: MultiAgentState) -> MultiAgentState:
        """Route to the appropriate agent."""
        intent = state["intent"]

        routing_map = {
            "rag_agent": "rag_agent",
            "transaction_agent": "transaction_agent",
            "fraud_agent": "fraud_agent",
            "escalation_agent": "escalation_agent",
        }

        return {**state, "routed_agent": routing_map.get(intent, "escalation_agent")}

    def _rag_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Handle FAQ/policy queries with RAG."""
        result = self.rag_agent.invoke(
            query=state["query"],
            user_id=state["user_id"],
            session_id=state["session_id"],
        )
        return {
            **state,
            "agent_response": result["answer"],
            "guardrail_flags": result["guardrail_flags"],
        }

    def _transaction_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Handle account/transaction queries with tools."""
        result = self.transaction_agent.invoke(
            query=state["query"],
            user_id=state["user_id"],
            session_id=state["session_id"],
        )
        return {
            **state,
            "agent_response": result["answer"],
            "error": result["error"],
        }

    def _fraud_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Handle fraud/security queries."""
        response = (
            "For your security, I can't handle fraud or security issues through this chat. "
            "Please call our fraud hotline at 1-800-XXX-XXXX immediately, or visit your nearest branch "
            "with a valid ID. If you believe your card is compromised, report it lost or stolen through "
            "the mobile app immediately."
        )
        return {
            **state,
            "agent_response": response,
            "guardrail_flags": ["fraud_escalation"],
        }

    def _escalation_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Handle queries that need human assistance."""
        response = (
            "I want to make sure you get the right help. This request requires assistance from a human agent. "
            "Please call our support team at 1-800-XXX-XXXX or visit your nearest branch. "
            "For your reference, your session ID is " + state["session_id"] + "."
        )
        return {
            **state,
            "agent_response": response,
            "guardrail_flags": ["human_escalation"],
        }

    def _compliance_agent(self, state: MultiAgentState) -> MultiAgentState:
        """Final compliance check before responding."""
        response = state["agent_response"]

        # Validate response through guardrails
        validation = self.guardrails.validate_response(response)

        if not validation["allowed"]:
            return {
                **state,
                "agent_response": "I'm sorry, but I can't provide that information. Please contact our support team for assistance.",
                "guardrail_flags": state["guardrail_flags"] + validation["flags"],
                "error": "Response blocked by compliance guardrails",
            }

        return state

    def invoke(self, query: str, user_id: str, session_id: str = "default") -> dict[str, Any]:
        """Invoke the multi-agent orchestrator."""
        initial_state = MultiAgentState(
            query=query,
            user_id=user_id,
            session_id=session_id,
            intent="",
            routed_agent="",
            agent_response="",
            guardrail_flags=[],
            error=None,
        )

        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial_state, config=config)

        return {
            "answer": result["agent_response"],
            "intent": result["intent"],
            "routed_agent": result["routed_agent"],
            "guardrail_flags": result["guardrail_flags"],
            "error": result["error"],
        }


def test_orchestrator():
    """Test the multi-agent orchestrator."""
    orchestrator = MultiAgentOrchestrator()

    # Get first user
    user_id = orchestrator.rag_agent.vector_store.collection.count()
    user_id = "usr_000001"

    test_queries = [
        "What is the funds availability policy?",
        "What is my account balance?",
        "I think my account was hacked",
        "Can you help me with something unusual?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = orchestrator.invoke(query, user_id=user_id)
        print(f"Intent: {result['intent']}")
        print(f"Routed to: {result['routed_agent']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Flags: {result['guardrail_flags']}")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    test_orchestrator()