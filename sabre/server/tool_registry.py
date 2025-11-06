"""
Tool Registry for managing internal and external tools.

This registry tracks which tools SABRE executes internally vs which tools
should be returned to the caller for external execution.

This enables SABRE to work with benchmarks like tau2-bench while maintaining
its orchestration architecture.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


class ToolExecutionMode(Enum):
    """Execution mode for a tool."""

    INTERNAL = "internal"  # Execute in SABRE's Python runtime
    EXTERNAL = "external"  # Return to caller for execution


@dataclass
class ToolDefinition:
    """Definition of a tool and how it should be executed."""

    name: str
    execution_mode: ToolExecutionMode
    callable: Callable | None  # Function for internal tools, None for external
    schema: dict | None  # Schema for external tools (OpenAI format)
    description: str


class ToolRegistry:
    """
    Registry for both internal and external tools.

    Internal tools are executed by SABRE's Python runtime.
    External tools are returned to the caller for execution.

    Example:
        registry = ToolRegistry()

        # Register SABRE's helpers as internal
        registry.register_internal("Search.web_search", Search.web_search, "Search the web")

        # Register benchmark tools as external
        registry.register_external("find_user", {...schema...}, "Find user by ID")

        # During orchestration
        if registry.is_external("find_user"):
            # Return to caller
        else:
            # Execute internally
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register_internal(self, name: str, callable: Callable, description: str = ""):
        """
        Register a tool that SABRE executes internally.

        Args:
            name: Tool name (e.g., "Search.web_search", "llm_call")
            callable: The actual Python function to execute
            description: Human-readable description
        """
        self._tools[name] = ToolDefinition(
            name=name,
            execution_mode=ToolExecutionMode.INTERNAL,
            callable=callable,
            schema=None,
            description=description
        )
        logger.debug(f"Registered internal tool: {name}")

    def register_external(self, name: str, schema: dict | None = None, description: str = ""):
        """
        Register a tool that external caller will execute.

        Args:
            name: Tool name (e.g., "find_user_id_by_name_zip")
            schema: OpenAI function schema (optional)
            description: Human-readable description
        """
        self._tools[name] = ToolDefinition(
            name=name,
            execution_mode=ToolExecutionMode.EXTERNAL,
            callable=None,  # External tools have no callable
            schema=schema,
            description=description
        )
        logger.debug(f"Registered external tool: {name}")

    def get(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        return self._tools.get(name)

    def is_external(self, name: str) -> bool:
        """
        Check if a tool should be executed externally.

        Args:
            name: Tool name

        Returns:
            True if tool is external, False if internal or not found
        """
        tool = self.get(name)
        if tool is None:
            # Unknown tool - treat as internal (will fail during execution if not available)
            logger.warning(f"Unknown tool: {name}, treating as internal")
            return False
        return tool.execution_mode == ToolExecutionMode.EXTERNAL

    def is_internal(self, name: str) -> bool:
        """Check if a tool should be executed internally."""
        return not self.is_external(name)

    def get_internal_tools(self) -> list[ToolDefinition]:
        """Get all internal tool definitions."""
        return [
            tool for tool in self._tools.values()
            if tool.execution_mode == ToolExecutionMode.INTERNAL
        ]

    def get_external_tools(self) -> list[ToolDefinition]:
        """Get all external tool definitions."""
        return [
            tool for tool in self._tools.values()
            if tool.execution_mode == ToolExecutionMode.EXTERNAL
        ]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def __repr__(self) -> str:
        internal_count = len(self.get_internal_tools())
        external_count = len(self.get_external_tools())
        return f"ToolRegistry(internal={internal_count}, external={external_count})"
