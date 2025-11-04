"""Tests for tau2-bench integration"""

import pytest
from sabre.benchmarks.tau2.agent import SabreTau2Agent


def test_agent_initialization():
    """Test agent can be initialized with API key"""
    agent = SabreTau2Agent(openai_api_key="test-key")
    assert agent is not None
    assert agent.api_key == "test-key"
    assert agent.sabre_port == 8011


def test_agent_requires_api_key():
    """Test agent raises error without API key"""
    import os
    # Temporarily remove API key
    old_key = os.environ.pop('OPENAI_API_KEY', None)
    try:
        with pytest.raises(ValueError, match="OpenAI API key required"):
            SabreTau2Agent()
    finally:
        if old_key:
            os.environ['OPENAI_API_KEY'] = old_key


def test_tool_formatting():
    """Test tool formatting produces correct output"""
    agent = SabreTau2Agent(openai_api_key="test-key")

    tools = [{
        "function": {
            "name": "get_order",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID"},
                    "user_id": {"type": "string", "description": "The user ID"}
                },
                "required": ["order_id"]
            }
        }
    }]

    formatted = agent._format_tools(tools)

    # Check that function name and description are present
    assert "get_order" in formatted
    assert "Get order details" in formatted

    # Check that parameters are listed
    assert "order_id" in formatted
    assert "user_id" in formatted
    assert "required" in formatted or "optional" in formatted


def test_param_formatting():
    """Test parameter formatting"""
    agent = SabreTau2Agent(openai_api_key="test-key")

    params = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    formatted = agent._format_params(params)

    # Required param should not have default
    assert "name: string" in formatted
    # Optional param should have default
    assert "age: integer = None" in formatted


def test_prompt_building():
    """Test prompt building combines all components"""
    agent = SabreTau2Agent(openai_api_key="test-key")

    system = "You are a helpful agent."
    tools = "Available tools: get_user()"
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Help me"}
    ]

    prompt = agent._build_sabre_prompt(system, tools, conversation)

    # All components should be in the prompt
    assert "You are a helpful agent" in prompt
    assert "Available tools" in prompt
    assert "CONVERSATION HISTORY" in prompt
    assert "Hello" in prompt
    assert "Hi there!" in prompt
    assert "CURRENT USER REQUEST" in prompt
    assert "Help me" in prompt


def test_openai_response_format():
    """Test OpenAI response formatting"""
    agent = SabreTau2Agent(openai_api_key="test-key")

    content = "I can help you with that."
    tool_calls = []

    response = agent._format_openai_response(content, tool_calls)

    # Check structure
    assert "id" in response
    assert "object" in response
    assert response["object"] == "chat.completion"
    assert "choices" in response
    assert len(response["choices"]) == 1

    # Check message
    message = response["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == content
    assert response["choices"][0]["finish_reason"] == "stop"


def test_error_response_format():
    """Test error response formatting"""
    agent = SabreTau2Agent(openai_api_key="test-key")

    error_msg = "Connection failed"
    response = agent._error_response(error_msg)

    assert "error" in response
    assert response["error"]["message"] == error_msg
    assert response["error"]["type"] == "sabre_error"


@pytest.mark.asyncio
async def test_process_chat_completion_structure():
    """Test chat completion returns proper structure even on error"""
    agent = SabreTau2Agent(openai_api_key="test-key", sabre_port=9999)

    # This should fail to connect but return proper error format
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]

    response = await agent.process_chat_completion(messages)

    # Should return error in proper format
    assert "error" in response
    assert "Cannot connect" in response["error"]["message"]


@pytest.mark.asyncio
async def test_conversation_id_validation():
    """Test that conversation IDs are generated with valid 'conv' prefix"""
    from unittest.mock import AsyncMock, patch, MagicMock
    import asyncio

    agent = SabreTau2Agent(openai_api_key="test-key", sabre_port=8011)

    # Track the conversation_id used in the request
    captured_conv_id = None

    class MockResponse:
        """Mock HTTP response that returns early with error"""
        status_code = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    def mock_stream(*args, **kwargs):
        """Mock stream() - not async, returns async context manager"""
        nonlocal captured_conv_id
        # Capture the conversation_id from the json payload
        json_data = kwargs.get('json', {})
        captured_conv_id = json_data.get('conversation_id')

        # Return mock response that will trigger error path
        return MockResponse()

    messages = [{"role": "user", "content": "test"}]

    # Test with no conv_id provided - should generate valid one
    with patch('httpx.AsyncClient') as mock_client:
        client_instance = MagicMock()
        client_instance.stream = mock_stream
        mock_client.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        await agent.process_chat_completion(messages)

        # Verify conversation ID starts with 'conv'
        assert captured_conv_id is not None
        assert captured_conv_id.startswith('conv'), f"Expected conv_id to start with 'conv', got: {captured_conv_id}"
        assert 'tau2' in captured_conv_id

    # Test with invalid conv_id provided - should generate new valid one
    captured_conv_id = None
    with patch('httpx.AsyncClient') as mock_client:
        client_instance = MagicMock()
        client_instance.stream = mock_stream
        mock_client.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        await agent.process_chat_completion(messages, conv_id='invalid_id')

        # Verify it was replaced with valid one
        assert captured_conv_id is not None
        assert captured_conv_id.startswith('conv'), f"Expected conv_id to start with 'conv', got: {captured_conv_id}"

    # Test with valid conv_id provided - should keep it
    captured_conv_id = None
    with patch('httpx.AsyncClient') as mock_client:
        client_instance = MagicMock()
        client_instance.stream = mock_stream
        mock_client.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        await agent.process_chat_completion(messages, conv_id='conv_valid_123')

        # Verify it kept the valid one
        assert captured_conv_id == 'conv_valid_123'
