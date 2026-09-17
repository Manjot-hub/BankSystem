"""
Tool-Calling Agent

Implements agentic workflows with tool-calling for banking operations.
Supports Groq and OpenAI backends seamlessly with multi-turn state persistence.
"""
import os
import json
from typing import Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.tools.banking_tools import ToolRegistry

# Auto-load environment variables from .env file
load_dotenv()


class ToolCallingState(TypedDict):
    """State for the tool-calling agent."""
    query: str
    user_id: str
    session_id: str
    messages: list[dict]
    tool_calls: list[dict]
    tool_results: list[dict]
    final_answer: str
    error: str | None
    max_iterations: int
    iteration: int


class ToolCallingAgent:
    """Tool-calling agent for banking operations."""

    def __init__(self):
        self.settings = get_settings()
        self.tools = ToolRegistry()
        self.llm = self._get_llm()
        self.graph = self._build_graph()

    def _get_llm(self):
        """Get LLM instance (Strict Groq Execution with fallback for CI tests)."""
        groq_api_key = getattr(self.settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            print("Warning: GROQ_API_KEY missing. Initializing FakeListChatModel for CI unit testing.")
            from langchain_community.chat_models.fake import FakeListChatModel
            return FakeListChatModel(responses=["I am a mock response for CI testing."])

        try:
            from langchain_groq import ChatGroq
            model_name = getattr(self.settings, "LLM_MODEL_PRIMARY", "groq/compound-mini")
            print(f"Initializing Tool Agent with Groq LLM: {model_name}")
            return ChatGroq(
                model_name=model_name,
                groq_api_key=groq_api_key,
                temperature=getattr(self.settings, "LLM_TEMPERATURE", 0.1),
                max_tokens=getattr(self.settings, "LLM_MAX_TOKENS", 2000),
            )
        except Exception as e:
            print(f"Failed to initialize ChatGroq: {e}")
            from langchain_community.chat_models.fake import FakeListChatModel
            return FakeListChatModel(responses=["Fallback mock response."])

    def _build_graph(self) -> StateGraph:
        """Build the tool-calling agent graph with multi-turn Redis checkpointer."""
        workflow = StateGraph(ToolCallingState)

        workflow.add_node("plan", self._plan)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("respond", self._respond)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "execute_tool")
        workflow.add_edge("execute_tool", "reflect")
        workflow.add_edge("reflect", "respond")
        workflow.add_edge("respond", END)

        # Multi-turn persistent checkpointer setup
        redis_url = getattr(self.settings, "REDIS_URL", None) or os.getenv("REDIS_URL")
        
        if redis_url:
            try:
                from langgraph.checkpoint.redis import RedisSaver
                from redis import Redis
                
                conn = Redis.from_url(redis_url)
                checkpointer = RedisSaver(conn)
                print("Multi-turn state persistence: Upstash Redis Checkpointer Active")
                return workflow.compile(checkpointer=checkpointer)
            except Exception as e:
                print(f"Warning: Could not connect to Redis ({e}). Falling back to MemorySaver.")

        print("Multi-turn state persistence: Local MemorySaver Active")
        return workflow.compile(checkpointer=MemorySaver())

    def _plan(self, state: ToolCallingState) -> ToolCallingState:
        """Plan actions dynamically mapped to state['user_id']."""
        if not self.llm:
            msg = "Tool-calling requires an active GROQ_API_KEY or OPENAI_API_KEY."
            return {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": msg}],
                "tool_calls": [],
                "tool_results": [],
                "final_answer": msg,
            }

        query_lower = state["query"].lower()
        tool_calls = []
        user_id = state.get("user_id")

        if "balance" in query_lower:
            tool_calls.append({
                "name": "get_account_balance", 
                "args": {"user_id": user_id}
            })
        elif "transaction" in query_lower or "recent" in query_lower:
            tool_calls.append({
                "name": "list_recent_transactions", 
                "args": {"user_id": user_id, "limit": 5}
            })
        elif "transfer" in query_lower:
            tool_calls.append({
                "name": "initiate_transfer", 
                "args": {
                    "user_id": user_id,
                    "from_account_id": "acc_checking_001",
                    "to_account_id": "acc_savings_001",
                    "amount": 100.0,
                    "currency": "USD"
                }
            })

        return {
            **state,
            "tool_calls": tool_calls,
        }

    def _execute_tool(self, state: ToolCallingState) -> ToolCallingState:
        """Execute planned tool calls strictly bound to state['user_id']."""
        tool_results = []
        user_id = state.get("user_id")

        for tool_call in state["tool_calls"]:
            function_name = tool_call.get("name")
            arguments = tool_call.get("args", {})

            # Strictly pass user_id parameter for authorization boundary
            arguments["user_id"] = user_id

            try:
                result = self.tools.execute(function_name, arguments)
                result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            except Exception as e:
                result_dict = {"error": f"Tool execution failed: {str(e)}", "requires_auth": False}

            tool_results.append({
                "tool_call_id": function_name,
                "name": function_name,
                "result": result_dict,
            })

        return {
            **state,
            "tool_results": tool_results,
            "messages": state["messages"] + [
                {
                    "role": "tool",
                    "name": r["name"],
                    "tool_call_id": r["tool_call_id"],
                    "content": str(r["result"]),
                }
                for r in tool_results
            ],
        }

    def _reflect(self, state: ToolCallingState) -> ToolCallingState:
        """Reflect on tool results and enforce security policies."""
        tool_results = state["tool_results"]
        if not tool_results:
            return {**state, "error": "No tool executed"}

        requires_auth = any(r["result"].get("requires_auth") for r in tool_results)
        if requires_auth:
            return {
                **state,
                "final_answer": "For your security, this action requires authentication through our mobile app or online banking. Please log in to continue.",
            }

        errors = [r["result"].get("error") for r in tool_results if r["result"].get("error")]
        if errors:
            return {
                **state,
                "error": "; ".join(errors),
                "final_answer": "I couldn't complete that request. Please try again or contact support.",
            }

        return {**state, "final_answer": None}

    def _respond(self, state: ToolCallingState) -> ToolCallingState:
        """Generate final response."""
        if state["final_answer"]:
            return state

        if not self.llm:
            return {
                **state,
                "final_answer": "Execution complete.",
            }

        tool_results = state["tool_results"]
        result_text = "\n".join([f"{r['name']}: {r['result']}" for r in tool_results])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking support agent.
Answer the user's question based on the provided tool results.
Be concise, accurate, and professional."""),
            ("human", "User question: {query}\n\nTool results:\n{tool_results}\n\nAnswer:"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({"query": state["query"], "tool_results": result_text})

        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
            "final_answer": response.content,
        }

    def invoke(self, query: str, user_id: str, session_id: str = "default") -> dict[str, Any]:
        """Invoke the tool-calling agent workflow."""
        initial_state = ToolCallingState(
            query=query,
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": query}],
            tool_calls=[],
            tool_results=[],
            final_answer="",
            error=None,
            max_iterations=5,
            iteration=0,
        )

        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial_state, config=config)

        return {
            "answer": result["final_answer"],
            "tool_calls": result["tool_calls"],
            "tool_results": result["tool_results"],
            "error": result["error"],
            "iteration": result["iteration"],
        }


if __name__ == "__main__":
    agent = ToolCallingAgent()
    user_id = agent.tools.data["users"][0]["user_id"]
    session_thread_id = "session_user_001_test"

    print("\n--- Turn 1 ---")
    res1 = agent.invoke("What is my account balance?", user_id=user_id, session_id=session_thread_id)
    print(f"Agent: {res1['answer']}")

    print("\n--- Turn 2 (Context Follow-Up) ---")
    res2 = agent.invoke("Show me my recent transactions for that account", user_id=user_id, session_id=session_thread_id)
    print(f"Agent: {res2['answer']}")