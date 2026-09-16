from bank_chatbot.tools.banking_tools import ToolRegistry


def test_get_account_balance():
    registry = ToolRegistry()
    user_id = registry.data["users"][0]["user_id"]
    result = registry.get_account_balance(user_id)

    assert result.success is True
    assert result.data


def test_list_recent_transactions():
    registry = ToolRegistry()
    user_id = registry.data["users"][0]["user_id"]
    result = registry.list_recent_transactions(user_id, limit=5)

    assert result.success is True
    assert len(result.data) <= 5


def test_transfer_requires_auth():
    registry = ToolRegistry()
    result = registry.initiate_transfer(
        user_id="usr_000001",
        from_account_id="acc_1",
        to_account_id="acc_2",
        amount=100,
    )

    assert result.requires_auth is True
    assert result.success is False
