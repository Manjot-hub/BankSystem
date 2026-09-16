"""
Banking Tools

Tool definitions for the agentic banking chatbot.
All tools are read-only in PoC. Mutating tools require authentication.
"""
import json
import uuid
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field


class BankingToolResult(BaseModel):
    """Standard result from a banking tool."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    requires_auth: bool = False


class ToolRegistry:
    """Registry of available banking tools."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/synthetic")
        self.data = self._load_data()

    def _load_data(self) -> dict[str, Any]:
        """Load synthetic banking data."""
        data = {}
        for filename in ["users.json", "accounts.json", "transactions.json"]:
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    data[filename.split(".")[0]] = json.load(f)
        return data

    def get_account_balance(self, user_id: str, account_id: Optional[str] = None) -> BankingToolResult:
        """Get account balance for a user."""
        accounts = [a for a in self.data.get("accounts", []) if a["user_id"] == user_id]
        if not accounts:
            return BankingToolResult(success=False, error="No accounts found for user")

        if account_id:
            accounts = [a for a in accounts if a["account_id"] == account_id]
            if not accounts:
                return BankingToolResult(success=False, error="Account not found")

        return BankingToolResult(
            success=True,
            data=[
                {
                    "account_id": a["account_id"],
                    "account_type": a["account_type"],
                    "balance": a["balance"],
                    "available_balance": a["available_balance"],
                    "currency": a["currency"],
                    "status": a["status"],
                }
                for a in accounts
            ],
        )

    def list_recent_transactions(self, user_id: str, account_id: Optional[str] = None, limit: int = 10) -> BankingToolResult:
        """List recent transactions for a user."""
        accounts = [a["account_id"] for a in self.data.get("accounts", []) if a["user_id"] == user_id]
        if not accounts:
            return BankingToolResult(success=False, error="No accounts found for user")

        txns = [t for t in self.data.get("transactions", []) if t["user_id"] == user_id]
        if account_id:
            txns = [t for t in txns if t["account_id"] == account_id]

        txns = sorted(txns, key=lambda x: x["timestamp"], reverse=True)[:limit]

        return BankingToolResult(
            success=True,
            data=[
                {
                    "transaction_id": t["transaction_id"],
                    "account_id": t["account_id"],
                    "transaction_type": t["transaction_type"],
                    "category": t["category"],
                    "merchant": t["merchant"],
                    "amount": t["amount"],
                    "currency": t["currency"],
                    "description": t["description"],
                    "status": t["status"],
                    "timestamp": t["timestamp"],
                }
                for t in txns
            ],
        )

    def initiate_transfer(self, user_id: str, from_account_id: str, to_account_id: str, amount: float, currency: str = "USD") -> BankingToolResult:
        """Initiate a transfer between accounts (requires authentication)."""
        return BankingToolResult(
            success=False,
            requires_auth=True,
            error="Transfers require authentication through the mobile app or online banking",
        )

    def dispute_transaction(self, user_id: str, transaction_id: str, reason: str) -> BankingToolResult:
        """Dispute a transaction (requires authentication)."""
        return BankingToolResult(
            success=False,
            requires_auth=True,
            error="Disputes require authentication and identity verification",
        )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions for LLM function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_account_balance",
                    "description": "Get balance for one or all accounts for the authenticated user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "Authenticated user ID"},
                            "account_id": {"type": "string", "description": "Optional account ID"},
                        },
                        "required": ["user_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_recent_transactions",
                    "description": "List recent transactions for the authenticated user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "Authenticated user ID"},
                            "account_id": {"type": "string", "description": "Optional account ID"},
                            "limit": {"type": "integer", "description": "Number of transactions to return", "default": 10},
                        },
                        "required": ["user_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "initiate_transfer",
                    "description": "Initiate a transfer between two accounts (requires authentication)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "Authenticated user ID"},
                            "from_account_id": {"type": "string", "description": "Source account ID"},
                            "to_account_id": {"type": "string", "description": "Destination account ID"},
                            "amount": {"type": "number", "description": "Amount to transfer"},
                            "currency": {"type": "string", "description": "Currency code", "default": "USD"},
                        },
                        "required": ["user_id", "from_account_id", "to_account_id", "amount"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dispute_transaction",
                    "description": "Dispute a transaction (requires authentication)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "Authenticated user ID"},
                            "transaction_id": {"type": "string", "description": "Transaction ID to dispute"},
                            "reason": {"type": "string", "description": "Reason for dispute"},
                        },
                        "required": ["user_id", "transaction_id", "reason"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> BankingToolResult:
        """Execute a tool by name."""
        tools = {
            "get_account_balance": self.get_account_balance,
            "list_recent_transactions": self.list_recent_transactions,
            "initiate_transfer": self.initiate_transfer,
            "dispute_transaction": self.dispute_transaction,
        }

        if tool_name not in tools:
            return BankingToolResult(success=False, error=f"Unknown tool: {tool_name}")

        try:
            return tools[tool_name](**arguments)
        except TypeError as e:
            return BankingToolResult(success=False, error=f"Invalid arguments for {tool_name}: {str(e)}")
        except Exception as e:
            return BankingToolResult(success=False, error=f"Tool execution failed: {str(e)}")


def test_tools():
    """Test banking tools."""
    registry = ToolRegistry()

    # Get first user
    user_id = registry.data["users"][0]["user_id"]
    accounts = registry.get_account_balance(user_id)
    print(f"Balance: {accounts.model_dump()}")

    txns = registry.list_recent_transactions(user_id, limit=5)
    print(f"Transactions: {txns.model_dump()}")

    transfer = registry.initiate_transfer(
        user_id=user_id,
        from_account_id="acc_1",
        to_account_id="acc_2",
        amount=100,
    )
    print(f"Transfer: {transfer.model_dump()}")

    print(f"Tool definitions: {len(registry.get_tool_definitions())}")


if __name__ == "__main__":
    test_tools()