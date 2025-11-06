"""
Tests for external tool support in SABRE orchestrator.

Tests the enhanced orchestrator with ToolRegistry to support
both internal and external tool execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from sabre.server.tool_registry import ToolRegistry, ToolExecutionMode, ToolDefinition
from sabre.server.orchestrator import Orchestrator, OrchestrationResult
from sabre.server.python_runtime import PythonRuntime
from sabre.common.executors.response import ResponseExecutor
from sabre.common.models.execution_tree import ExecutionTree, ExecutionStatus


class TestToolRegistry:
    """Test ToolRegistry functionality."""

    def test_register_internal_tool(self):
        """Test registering an internal tool."""
        registry = ToolRegistry()

        def test_func(x):
            return x * 2

        registry.register_internal("test_func", test_func, "Test function")

        tool = registry.get("test_func")
        assert tool is not None
        assert tool.name == "test_func"
        assert tool.execution_mode == ToolExecutionMode.INTERNAL
        assert tool.callable == test_func
        assert tool.description == "Test function"

    def test_register_external_tool(self):
        """Test registering an external tool."""
        registry = ToolRegistry()
        schema = {"type": "function", "function": {"name": "external_func"}}

        registry.register_external("external_func", schema, "External function")

        tool = registry.get("external_func")
        assert tool is not None
        assert tool.name == "external_func"
        assert tool.execution_mode == ToolExecutionMode.EXTERNAL
        assert tool.callable is None
        assert tool.schema == schema
        assert tool.description == "External function"

    def test_is_external(self):
        """Test checking if a tool is external."""
        registry = ToolRegistry()

        registry.register_internal("internal", lambda x: x, "Internal")
        registry.register_external("external", None, "External")

        assert registry.is_external("internal") is False
        assert registry.is_external("external") is True
        assert registry.is_external("unknown") is False  # Unknown treated as internal

    def test_get_internal_tools(self):
        """Test getting all internal tools."""
        registry = ToolRegistry()

        registry.register_internal("func1", lambda x: x, "Func 1")
        registry.register_internal("func2", lambda x: x, "Func 2")
        registry.register_external("func3", None, "Func 3")

        internal_tools = registry.get_internal_tools()
        assert len(internal_tools) == 2
        assert all(t.execution_mode == ToolExecutionMode.INTERNAL for t in internal_tools)

    def test_get_external_tools(self):
        """Test getting all external tools."""
        registry = ToolRegistry()

        registry.register_internal("func1", lambda x: x, "Func 1")
        registry.register_external("func2", None, "Func 2")
        registry.register_external("func3", None, "Func 3")

        external_tools = registry.get_external_tools()
        assert len(external_tools) == 2
        assert all(t.execution_mode == ToolExecutionMode.EXTERNAL for t in external_tools)


class TestOrchestratorToolPartitioning:
    """Test Orchestrator tool partitioning logic."""

    def test_extract_function_name(self):
        """Test extracting function names from helper code."""
        runtime = PythonRuntime()
        executor = Mock(spec=ResponseExecutor)
        registry = ToolRegistry()
        orchestrator = Orchestrator(executor, runtime, tool_registry=registry)

        # Test simple function calls
        assert orchestrator._extract_function_name("result(123)") == "result"
        assert orchestrator._extract_function_name("find_user(id=456)") == "find_user"

        # Test method calls
        assert orchestrator._extract_function_name("Search.web_search('test')") == "Search.web_search"
        assert orchestrator._extract_function_name("Bash.execute('ls')") == "Bash.execute"

        # Test invalid code
        assert orchestrator._extract_function_name("invalid syntax!!") is None

    def test_partition_helpers(self):
        """Test partitioning helpers into internal vs external."""
        runtime = PythonRuntime()
        executor = Mock(spec=ResponseExecutor)
        registry = ToolRegistry()

        # Register some tools
        registry.register_internal("result", lambda x: x, "Result")
        registry.register_internal("Search.web_search", lambda x: x, "Search")
        registry.register_external("find_user", None, "Find user")
        registry.register_external("get_order", None, "Get order")

        orchestrator = Orchestrator(executor, runtime, tool_registry=registry)

        # Test partitioning
        helpers = [
            "result(123)",
            "find_user(id=456)",
            "Search.web_search('test')",
            "get_order(order_id='#123')",
        ]

        internal, external = orchestrator._partition_helpers(helpers)

        assert len(internal) == 2
        assert "result(123)" in internal
        assert "Search.web_search('test')" in internal

        assert len(external) == 2
        assert "find_user(id=456)" in external
        assert "get_order(order_id='#123')" in external

    def test_parse_external_helpers(self):
        """Test parsing external helpers into tool_call format."""
        runtime = PythonRuntime()
        executor = Mock(spec=ResponseExecutor)
        registry = ToolRegistry()
        orchestrator = Orchestrator(executor, runtime, tool_registry=registry)

        # Test parsing
        helpers = [
            "find_user(user_id=456)",
            'get_order(order_id="#W123")',
        ]

        tool_calls = orchestrator._parse_external_helpers(helpers)

        assert len(tool_calls) == 2

        # Check first tool call
        tc1 = tool_calls[0]
        assert tc1["name"] == "find_user"
        assert tc1["arguments"]["user_id"] == 456
        assert "id" in tc1
        assert tc1["id"].startswith("call_")

        # Check second tool call
        tc2 = tool_calls[1]
        assert tc2["name"] == "get_order"
        assert tc2["arguments"]["order_id"] == "#W123"

    def test_parse_external_helpers_with_positional_args(self):
        """Test parsing external helpers with positional arguments."""
        runtime = PythonRuntime()
        executor = Mock(spec=ResponseExecutor)
        registry = ToolRegistry()
        orchestrator = Orchestrator(executor, runtime, tool_registry=registry)

        # Test parsing with positional args
        helpers = [
            'find_user_by_name_zip("John", "Doe", "12345")',
        ]

        tool_calls = orchestrator._parse_external_helpers(helpers)

        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["name"] == "find_user_by_name_zip"
        # Positional args stored as arg0, arg1, arg2
        assert tc["arguments"]["arg0"] == "John"
        assert tc["arguments"]["arg1"] == "Doe"
        assert tc["arguments"]["arg2"] == "12345"


class TestOrchestrationResult:
    """Test enhanced OrchestrationResult."""

    def test_orchestration_result_with_pending_tools(self):
        """Test OrchestrationResult with pending tool calls."""
        result = OrchestrationResult(
            success=False,
            final_response="Pending...",
            conversation_id="conv_123",
            response_id="resp_456",
            status="awaiting_tool_results",
            pending_tool_calls=[
                {"id": "call_1", "name": "find_user", "arguments": {"id": 123}},
                {"id": "call_2", "name": "get_order", "arguments": {"order_id": "#W123"}},
            ]
        )

        assert result.status == "awaiting_tool_results"
        assert result.pending_tool_calls is not None
        assert len(result.pending_tool_calls) == 2
        assert result.pending_tool_calls[0]["name"] == "find_user"

    def test_orchestration_result_completed(self):
        """Test OrchestrationResult for completed orchestration."""
        result = OrchestrationResult(
            success=True,
            final_response="Done!",
            conversation_id="conv_123",
            response_id="resp_456",
            status="completed"
        )

        assert result.status == "completed"
        assert result.pending_tool_calls is None
        assert result.success is True


class TestExecutionTreePausedStatus:
    """Test ExecutionTree with PAUSED status."""

    def test_paused_status_exists(self):
        """Test that PAUSED status is available."""
        assert hasattr(ExecutionStatus, "PAUSED")
        assert ExecutionStatus.PAUSED.value == "paused"

    def test_paused_node(self):
        """Test creating a paused execution node."""
        from sabre.common.models.execution_tree import ExecutionNode, ExecutionNodeType
        from datetime import datetime

        node = ExecutionNode(
            id="node_123",
            parent_id=None,
            node_type=ExecutionNodeType.RESPONSE_ROUND,
            status=ExecutionStatus.PAUSED,
            start_time=datetime.now(),
            end_time=None
        )

        assert node.status == ExecutionStatus.PAUSED
        assert not node.is_complete()  # PAUSED is not complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
