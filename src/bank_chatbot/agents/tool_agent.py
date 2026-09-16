"""
Tool-Calling Agent

Implements agentic workflows with tool-calling for banking operations.
Supports ReAct and Plan-and-Execute patterns.
"""
from typing import Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.tools.banking_tools import ToolRegistry, BankingToolResult


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
        """Get LLM for tool-calling."""
        if self.settings.OPENAI_API_KEY:
            return ChatOpenAI(
                model=self.settings.LLM_MODEL_PRIMARY,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.LLM_MAX_TOKENS,
                openai_api_key=self.settings.OPENAI_API_KEY,
            ).bind_tools(self.tools.get_tool_definitions())
        return None

    def _build_graph(self) -> StateGraph:
        """Build the tool-calling agent graph."""
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

        return workflow.compile(checkpointer=MemorySaver())

    def _plan(self, state: ToolCallingState) -> ToolCallingState:
        """Plan the next action."""
        if not self.llm:
            # Retrieval-only mode: no tool calling
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": "Tool-calling requires an OpenAI API key. Please set OPENAI_API_KEY to enable banking tools."}
                ],
                "tool_calls": [],
                "tool_results": [],
                "final_answer": "Tool-calling requires an OpenAI API key. Please set OPENAI_API_KEY to enable banking tools.",
            }

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking support agent with access to tools.
You can help users with:
- Checking account balances
- Viewing recent transactions
- Initiating transfers (requires authentication)
- Disputing transactions (requires authentication)

Rules:
1. Always verify the user is authenticated before accessing account data
2. Never expose sensitive information
3. Use tools when the user asks about their account
4. If a tool requires authentication, explain that clearly
5. Never make up account information
6. If you need multiple tools, call them one at a time"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({"query": state["query"]})

        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = [tc for tc in response.tool_calls]

        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": response.content or ""},
            ],
            "tool_calls": tool_calls,
        }

    def _execute_tool(self, state: ToolCallingState) -> ToolCallingState:
        """Execute planned tool calls."""
        tool_results = []

        for tool_call in state["tool_calls"]:
            function_name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments", "{}")

            if isinstance(arguments, str):
                import json
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            result = self.tools.execute(function_name, arguments)
            tool_results.append({
                "tool_call_id": tool_call.get("id"),
                "name": function_name,
                "result": result.model_dump(),
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
        """Reflect on tool results and decide next step."""
        tool_results = state["tool_results"]

        if not tool_results:
            return {
                **state,
                "error": "No tool results available",
            }

        # Check if any tool requires authentication
        requires_auth = any(r["result"].get("requires_auth") for r in tool_results)

        if requires_auth:
            return {
                **state,
                "final_answer": "For your security, this action requires authentication through our mobile app or online banking. Please log in to continue.",
            }

        # Check for errors
        errors = [r["result"].get("error") for r in tool_results if r["result"].get("error")]
        if errors:
            return {
                **state,
                "error": "; ".join(errors),
                "final_answer": "I couldn't complete that request. Please try again or contact support.",
            }

        return {
            **state,
            "final_answer": None,
        }

    def _respond(self, state: ToolCallingState) -> ToolCallingState:
        """Generate final response."""
        if state["final_answer"]:
            return state

        if not self.llm:
            return {
                **state,
                "final_answer": "Tool-calling requires an OpenAI API key. Please set OPENAI_API_KEY to enable banking tools.",
            }

        tool_results = state["tool_results"]
        result_text = "\n".join([
            f"{r['name']}: {r['result']}"
            for r in tool_results
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking support agent.
Answer the user's question based on the tool results.
Be concise and professional.
Never expose sensitive information.
If the tool result requires authentication, explain that clearly."""),
            ("human", "User question: {query}\n\nTool results:\n{tool_results}\n\nAnswer:"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({"query": state["query"], "tool_results": result_text})

        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": response.content}
            ],
            "final_answer": response.content,
        }

    def invoke(self, query: str, user_id: str, session_id: str = "default") -> dict[str, Any]:
        """Invoke the tool-calling agent."""
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


def test_agent():
    """Test the tool-calling agent."""
    agent = ToolCallingAgent()

    # Get first user
    user_id = agent.tools.data["users"][0]["user_id"]

    test_queries = [
        "What is my account balance?",
        "Show me my recent transactions",
        "Transfer $100 to my savings account",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = agent.invoke(query, user_id=user_id)
        print(f"Answer: {result['answer']}")
        print(f"Tool calls: {len(result['tool_calls'])}")
        print(f"Tool results: {len(result['tool_results'])}")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    test_agent()