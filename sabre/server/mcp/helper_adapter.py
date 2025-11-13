"""
MCP Helper Adapter.

Bridges MCP tools into SABRE's helper system, making them available in the Python runtime.
"""

import asyncio
import logging
from typing import Any, Callable

from sabre.common.models import Content, TextContent, ImageContent
from .models import (
    MCPTool,
    MCPToolResult,
    MCPContent,
    MCPToolError,
    MCPServerNotFoundError,
)
from .client_manager import MCPClientManager

logger = logging.getLogger(__name__)


class MCPHelperAdapter:
    """
    Adapter that exposes MCP tools as SABRE helpers.

    Responsibilities:
    - Convert MCP tools to Python callables
    - Generate documentation for LLM prompts
    - Route tool calls to appropriate MCP server
    - Transform MCP results to SABRE Content
    """

    def __init__(self, client_manager: MCPClientManager, event_loop=None):
        """
        Initialize MCP helper adapter.

        Args:
            client_manager: MCP client manager instance
            event_loop: Optional event loop to use for async operations (for thread-safe execution)
        """
        self.client_manager = client_manager
        self._tools_cache: dict[str, MCPTool] = {}
        self._main_loop = event_loop  # Store reference to main event loop for thread-safe calls

    async def refresh_tools(self) -> None:
        """
        Refresh tools cache from all connected servers.

        This should be called after connecting to servers or when tools might have changed.
        """
        self._tools_cache.clear()

        all_tools = await self.client_manager.get_all_tools()

        for server_name, tools in all_tools.items():
            for tool in tools:
                # Create qualified tool name: ServerName.tool_name
                qualified_name = f"{server_name}.{tool.name}"
                self._tools_cache[qualified_name] = tool
                logger.debug(f"Registered MCP tool: {qualified_name}")

        logger.info(f"Refreshed MCP tools cache: {len(self._tools_cache)} tools available")

    def get_available_tools(self) -> dict[str, Callable]:
        """
        Get all MCP tools as Python callables.

        Returns:
            Dictionary mapping tool name to callable function

        Example:
            {
                "Postgres.query": <callable>,
                "GitHub.create_pr": <callable>,
            }
        """
        tools = {}

        for qualified_name, tool in self._tools_cache.items():
            # Create a callable that wraps the tool invocation
            callable_func = self._create_tool_callable(qualified_name, tool)
            tools[qualified_name] = callable_func

        return tools

    def _create_tool_callable(self, qualified_name: str, tool: MCPTool) -> Callable:
        """
        Create a Python callable for an MCP tool.

        The callable will invoke the tool asynchronously and return results.

        Args:
            qualified_name: Qualified tool name (ServerName.tool_name)
            tool: MCP tool definition

        Returns:
            Callable function
        """

        def tool_wrapper(*args, **kwargs) -> Any:
            """
            Wrapper function that invokes the MCP tool.

            Routes all tool calls to the MCP client's owning event loop to prevent
            "Task got Future attached to a different loop" errors.

            Accepts both positional and keyword arguments to be flexible with LLM calls.
            """
            try:
                # Convert positional args to keyword args based on tool schema
                if args:
                    # Get required parameters from tool schema
                    schema = tool.input_schema
                    if "properties" in schema:
                        prop_names = list(schema["properties"].keys())
                        # Map positional args to parameter names in order
                        for i, arg in enumerate(args):
                            if i < len(prop_names):
                                param_name = prop_names[i]
                                if param_name not in kwargs:  # Don't override explicit kwargs
                                    kwargs[param_name] = arg

                # Extract server name to get the client's owning loop
                server_name = qualified_name.split(".", 1)[0]
                client = self.client_manager.get_client_by_name(server_name)
                if client is None:
                    raise ValueError(f"MCP server '{server_name}' not found or not connected")
                target_loop = getattr(client, "loop", None)

                if target_loop is None:
                    # No loop stored - fall back to asyncio.run (should not happen in production)
                    logger.warning(f"No loop stored for client {server_name}, using asyncio.run()")
                    return asyncio.run(self.invoke_tool(qualified_name, **kwargs))

                logger.debug(f"[{qualified_name}] Routing to client loop: {hex(id(target_loop))}")

                # Build coroutine
                coro = self.invoke_tool(qualified_name, **kwargs)

                # Try to detect current running loop
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                import concurrent.futures

                # Helper function: schedule coro on target_loop and wait for result
                def rendezvous():
                    fut = asyncio.run_coroutine_threadsafe(coro, target_loop)
                    content_list = fut.result()  # This is list[Content]
                    logger.debug(f"[{qualified_name}] Got {len(content_list)} content items")

                    # DEBUG: Log what we actually got
                    for i, item in enumerate(content_list):
                        logger.debug(f"[{qualified_name}] Content {i}: type={type(item).__name__}, has_text={hasattr(item, 'text')}")
                        if hasattr(item, 'text') and item.text:
                            preview = item.text[:100] if len(item.text) > 100 else item.text
                            logger.debug(f"[{qualified_name}] Content {i} text preview: {preview}")

                    # Extract text from Content objects
                    # MCP tools typically return JSON in text format
                    result_value = None
                    if len(content_list) == 0:
                        result_value = None
                    elif len(content_list) == 1:
                        # Single content item - return the text directly
                        content_item = content_list[0]
                        if hasattr(content_item, 'text'):
                            text = content_item.text
                            if text:
                                # Strip whitespace
                                text = text.strip()
                                # Always try to parse as JSON first (handles strings, objects, arrays, etc.)
                                try:
                                    import json
                                    result_value = json.loads(text)
                                    logger.debug(f"[{qualified_name}] Parsed as JSON: {type(result_value).__name__}")
                                except json.JSONDecodeError:
                                    # Not valid JSON, use as-is
                                    logger.debug(f"[{qualified_name}] Not JSON, using text as-is")
                                    result_value = text
                            else:
                                result_value = text
                        else:
                            result_value = str(content_item)
                    else:
                        # Multiple content items - return list of texts
                        texts = []
                        for content_item in content_list:
                            if hasattr(content_item, 'text'):
                                text = content_item.text
                                if text:
                                    text = text.strip()
                                    # Try to parse each item as JSON
                                    try:
                                        import json
                                        texts.append(json.loads(text))
                                    except json.JSONDecodeError:
                                        # Not JSON, use text as-is
                                        texts.append(text)
                                else:
                                    texts.append(text)
                            else:
                                texts.append(str(content_item))
                        result_value = texts

                    # Log the final result type
                    logger.debug(f"[{qualified_name}] Final result type: {type(result_value).__name__}")

                    # Print the result so it's captured by the orchestrator
                    # This makes the tool output visible in <helpers_result>
                    if result_value is not None:
                        import json as json_mod
                        if isinstance(result_value, (dict, list)):
                            # Pretty print JSON for readability
                            print(json_mod.dumps(result_value, indent=2))
                        else:
                            print(result_value)

                    return result_value

                # Case 1: No running loop - we can safely call rendezvous directly
                if running_loop is None:
                    logger.debug(f"[{qualified_name}] No running loop, calling rendezvous directly")
                    return rendezvous()

                # Case 2: Running on same loop as target - offload to thread to avoid deadlock
                # (calling .result() blocks, which would prevent the loop from processing the task)
                if running_loop is target_loop:
                    logger.debug(
                        f"[{qualified_name}] Same loop detected ({hex(id(running_loop))}), "
                        "offloading to thread to prevent deadlock"
                    )
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(rendezvous).result()

                # Case 3: Running on different loop - can call rendezvous directly
                # (the .result() blocks on THIS loop, while target_loop processes the task)
                logger.debug(
                    f"[{qualified_name}] Different loops: running={hex(id(running_loop))}, "
                    f"target={hex(id(target_loop))}, calling rendezvous"
                )
                return rendezvous()

            except Exception as e:
                logger.error(f"Error invoking MCP tool {qualified_name}: {e}")
                raise

        # Set function metadata for introspection
        tool_wrapper.__name__ = qualified_name
        tool_wrapper.__doc__ = tool.description

        return tool_wrapper

    async def invoke_tool(self, qualified_name: str, **kwargs) -> list[Content]:
        """
        Invoke an MCP tool and return SABRE Content.

        This method is loop-aware: it ensures client.call_tool() runs on the client's owning loop.

        Args:
            qualified_name: Qualified tool name (ServerName.tool_name)
            **kwargs: Tool arguments

        Returns:
            List of SABRE Content objects

        Raises:
            MCPServerNotFoundError: If server not found
            MCPToolError: If tool invocation fails
        """
        # Parse server name and tool name from qualified name
        if "." not in qualified_name:
            raise ValueError(f"Tool name must be qualified with server name: {qualified_name}")

        server_name, tool_name = qualified_name.split(".", 1)

        # Get client for server
        client = self.client_manager.get_client_by_name(server_name)
        if client is None:
            raise MCPServerNotFoundError(server_name)
        target_loop = getattr(client, "_loop", None)

        # Detect current loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # If we're already on the client's loop (or no loop stored), just call directly
        if target_loop is None or current_loop is target_loop:
            logger.debug(f"Invoking MCP tool: {qualified_name} with args: {kwargs}")
            mcp_result = await client.call_tool(tool_name, kwargs)
            sabre_content = self._transform_result(mcp_result)
            logger.debug(f"MCP tool {qualified_name} returned {len(sabre_content)} content items")
            return sabre_content

        # We're on a different loop - route to the client's loop
        # This should not normally happen since invoke_tool is async and should be called
        # from the same context as the client, but we handle it for safety
        logger.debug(f"Routing MCP tool {qualified_name} to client's loop (current != target)")

        import concurrent.futures

        async def do_call():
            mcp_result = await client.call_tool(tool_name, kwargs)
            return self._transform_result(mcp_result)

        future = asyncio.run_coroutine_threadsafe(do_call(), target_loop)
        return future.result()

    def _transform_result(self, mcp_result: MCPToolResult) -> list[Content]:
        """
        Transform MCP tool result to SABRE Content.

        Args:
            mcp_result: MCP tool result

        Returns:
            List of SABRE Content objects
        """
        sabre_content = []

        for mcp_content in mcp_result.content:
            content = self._transform_content(mcp_content)
            if content:
                sabre_content.append(content)

        return sabre_content

    def _transform_content(self, mcp_content: MCPContent) -> Content | None:
        """
        Transform a single MCP content item to SABRE Content.

        Args:
            mcp_content: MCP content item

        Returns:
            SABRE Content object, or None if unsupported type
        """
        if mcp_content.type == "text":
            return TextContent(text=mcp_content.text or "")

        elif mcp_content.type == "image":
            # MCP images come as base64 data
            if mcp_content.data:
                return ImageContent(image_data=mcp_content.data, mime_type=mcp_content.mimeType or "image/png")

        elif mcp_content.type == "resource":
            # Resources are represented as text with URI reference
            resource_text = f"Resource: {mcp_content.uri}"
            if mcp_content.text:
                resource_text += f"\n{mcp_content.text}"
            return TextContent(text=resource_text)

        else:
            logger.warning(f"Unsupported MCP content type: {mcp_content.type}")
            return None

    def generate_documentation(self) -> str:
        """
        Generate documentation for all MCP tools.

        This is included in the system prompt to inform the LLM about available tools.

        Returns:
            Markdown-formatted documentation string
        """
        if not self._tools_cache:
            return ""

        # Group tools by server
        servers = {}
        for qualified_name, tool in self._tools_cache.items():
            server_name = tool.server_name
            if server_name not in servers:
                servers[server_name] = []
            servers[server_name].append((qualified_name, tool))

        # Generate markdown documentation
        doc_lines = ["## MCP Tools", ""]
        doc_lines.append("The following tools are available from connected MCP servers:")
        doc_lines.append("")

        for server_name, tools in sorted(servers.items()):
            doc_lines.append(f"### {server_name} Server")
            doc_lines.append("")

            for qualified_name, tool in tools:
                # Generate function signature
                signature = tool.get_signature()
                # Replace tool.name with qualified_name in signature
                signature = signature.replace(f"{tool.name}(", f"{qualified_name}(", 1)

                doc_lines.append(f"**{signature}**")
                doc_lines.append(f"{tool.description}")
                doc_lines.append("")

        return "\n".join(doc_lines)

    def get_tool_count(self) -> int:
        """Get number of available MCP tools"""
        return len(self._tools_cache)

    def get_server_names(self) -> list[str]:
        """Get list of servers that have tools registered"""
        servers = set()
        for tool in self._tools_cache.values():
            servers.add(tool.server_name)
        return sorted(servers)

    def has_tool(self, qualified_name: str) -> bool:
        """Check if a tool is available"""
        return qualified_name in self._tools_cache

    def get_tool(self, qualified_name: str) -> MCPTool | None:
        """Get tool definition by qualified name"""
        return self._tools_cache.get(qualified_name)
