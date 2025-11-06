"""
SABRE Agent adapter for tau2-bench.

This module provides a tau2-bench compatible agent that uses SABRE's
continuation-passing execution engine under the hood.
"""

import os
import asyncio
import logging
from typing import Optional, List
from pydantic import BaseModel

# tau2-bench imports
from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
    SystemMessage,
)
from tau2.environment.tool import Tool

# SABRE imports
from sabre.server.orchestrator import Orchestrator
from sabre.server.python_runtime import PythonRuntime
from sabre.server.tool_registry import ToolRegistry
from sabre.common.executors.response import ResponseExecutor
from sabre.common.models.execution_tree import ExecutionTree

logger = logging.getLogger(__name__)


class SabreAgentState(BaseModel):
    """State for SABRE agent matching tau2-bench expectations."""

    system_messages: list[SystemMessage]
    messages: list[Message]
    conversation_id: Optional[str]  # None for new conversations, populated after first turn
    turn_number: int = 0

    # NEW: For pause/resume orchestration
    pending_orchestration_result: Optional[object] = None  # OrchestrationResult when paused
    execution_tree: Optional[object] = None  # ExecutionTree for resumption

    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types


class SabreAgent(LocalAgent[SabreAgentState]):
    """
    SABRE-powered agent for tau2-bench evaluation.

    This agent uses SABRE's recursive execution engine to handle
    tau2-bench tasks while conforming to the LocalAgent interface.

    Usage:
        agent = SabreAgent(
            tools=tau2_tools,
            domain_policy="You are a retail customer service agent...",
            model="gpt-4o"
        )

        state = agent.get_init_state()
        message, state = agent.generate_next_message(user_message, state)
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize SABRE agent.

        Args:
            tools: tau2-bench tools (domain-specific functions)
            domain_policy: Domain-specific policy and guidelines
            llm: OpenAI model to use (default: from env or "gpt-4o")
            llm_args: Additional LLM arguments (ignored for now)
            **kwargs: Additional arguments (for compatibility)
        """
        logger.warning("=" * 80)
        logger.warning("🎯 SABRE AGENT INITIALIZING - tau2-bench is using SABRE!")
        logger.warning("=" * 80)

        super().__init__(tools=tools, domain_policy=domain_policy)

        # Determine model
        self.model = llm or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.llm_args = llm_args or {}

        # Initialize SABRE components
        self.runtime = PythonRuntime()
        self.executor = ResponseExecutor(
            api_key=os.getenv("OPENAI_API_KEY"),
            default_model=self.model
        )

        # Create tool registry
        self.tool_registry = ToolRegistry()

        # Register SABRE's default helpers as INTERNAL tools
        self._register_internal_tools()

        # Register tau2 tools as EXTERNAL tools and stub implementations
        self._register_tau2_tools(tools)

        # Create orchestrator with tool registry
        self.orchestrator = Orchestrator(
            executor=self.executor,
            python_runtime=self.runtime,
            max_iterations=10,
            tool_registry=self.tool_registry
        )

        logger.warning(f"✅ SABRE AGENT READY - {len(tools)} tools registered, model={self.model}")
        logger.info(f"Initialized SabreAgent with {len(tools)} tools, model={self.model}")
        logger.info(f"Tool Registry: {self.tool_registry}")

    def _register_internal_tools(self):
        """
        Register SABRE's default helpers as INTERNAL tools.

        These tools will be executed by SABRE's orchestrator internally.
        """
        # Get helpers from runtime namespace
        namespace = self.runtime.namespace

        # Register core SABRE helpers as internal
        internal_helpers = [
            'llm_call', 'llm_bind', 'llm_list_bind', 'pandas_bind',
            'result', 'download'
        ]

        for helper_name in internal_helpers:
            if helper_name in namespace:
                self.tool_registry.register_internal(
                    name=helper_name,
                    callable=namespace[helper_name],
                    description=f"SABRE internal helper: {helper_name}"
                )

        # Register helper classes (Bash, Search, Web)
        helper_classes = ['Bash', 'Search', 'Web']
        for class_name in helper_classes:
            if class_name in namespace:
                helper_class = namespace[class_name]
                # Register class methods as internal tools
                for method_name in dir(helper_class):
                    if not method_name.startswith('_'):
                        method = getattr(helper_class, method_name, None)
                        if callable(method):
                            full_name = f"{class_name}.{method_name}"
                            self.tool_registry.register_internal(
                                name=full_name,
                                callable=method,
                                description=f"SABRE internal helper: {full_name}"
                            )

        logger.info(f"Registered {len(self.tool_registry.get_internal_tools())} internal SABRE helpers")

    def _register_tau2_tools(self, tools: List[Tool]):
        """
        Register tau2-bench tools as EXTERNAL tools.

        These tools will be returned to tau2-bench for execution, not executed by SABRE.

        Args:
            tools: tau2-bench Tool objects
        """
        for tool in tools:
            # Extract tool metadata
            tool_name = tool.name
            tool_description = getattr(tool, 'short_desc', getattr(tool, 'description', ''))
            tool_schema = tool.openai_schema if hasattr(tool, 'openai_schema') else None

            # Register as EXTERNAL tool in registry
            self.tool_registry.register_external(
                name=tool_name,
                schema=tool_schema,
                description=tool_description
            )

        logger.info(f"Registered {len(tools)} tau2 tools as EXTERNAL in tool registry")

    @property
    def system_prompt(self) -> str:
        """
        Build system prompt combining domain policy with SABRE instructions.

        Returns:
            Complete system prompt string
        """
        # Build tool documentation
        tools_doc = self._build_tools_documentation()

        # Use SABRE-style prompt with customer service context
        return f"""You are a customer service agent. You help users by following the policy below and calling the appropriate tools/functions when needed.

# How to Use Tools

When you need to call a tool, emit Python code within <helpers></helpers> XML tags. The code will be executed for you and results returned in <helpers_result></helpers_result> tags.

Important rules:
- Keep code blocks very short (1-3 lines typically)
- Call only ONE tool per <helpers> block
- After getting <helpers_result>, you can either call another tool or respond to the user
- Tool functions are already imported and ready to use

# CRITICAL RULES for Tool Calls

- NEVER write variable assignments for tool calls. Write `find_user_id(...)` NOT `user_id = find_user_id(...)`.
- The <helpers> block is for DIRECT FUNCTION CALLS ONLY, not for storing intermediate results.
- You will receive tool results in <helpers_result> tags which you can reference directly in your response.
- MAXIMUM 1-2 TOOL CALLS per <helpers> block. NEVER call the same tool multiple times in one block.
- ONLY ONE <helpers> block per response. DO NOT generate multiple <helpers> blocks before seeing results.
- NEVER pre-emptively retry a tool call before seeing its result. Wait for <helpers_result> first.
- NEVER assume a tool call will fail before executing it. Trust the tool and wait for the actual result.
- If a tool returns an error, DO NOT retry the exact same call. Either try a different approach or inform the user.
- If a tool call succeeds (returns expected data without error), DO NOT call it again. Use the result you already have.
- DETECT SUCCESS: When an operation succeeds (e.g., "exchange_id" is returned, status shows success), STOP and inform the user. DO NOT continue calling the same operation "to verify" - trust the successful result.

# Examples

User: "Hi, I'm John Doe from zip 12345, I need help with my order"
Assistant:
<helpers>
find_user_id_by_name_zip(first_name="John", last_name="Doe", zip="12345")
</helpers>

User: "My order number is #W2378156"
Assistant:
<helpers>
get_order_details(order_id="#W2378156")
</helpers>

# Policy

{self.domain_policy}

# Available Tools

{tools_doc}

Remember: ALWAYS authenticate users first before accessing their data.
""".strip()

    def _build_tools_documentation(self) -> str:
        """
        Generate documentation for available tau2 tools.

        Returns:
            Formatted tool documentation string
        """
        if not self.tools:
            return "No tools available."

        docs = []
        for tool in self.tools:
            # Use the tool's built-in to_str() method if available
            if hasattr(tool, 'to_str'):
                docs.append(tool.to_str())
            else:
                # Fallback: use openai_schema
                schema = tool.openai_schema
                func_schema = schema.get('function', {})
                name = func_schema.get('name', tool.name)
                desc = func_schema.get('description', getattr(tool, 'short_desc', ''))
                docs.append(f"{name}: {desc}")

        return "\n\n".join(docs)

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None
    ) -> SabreAgentState:
        """
        Get initial agent state.

        Args:
            message_history: Previous conversation messages (optional)

        Returns:
            Initial SabreAgentState
        """
        if message_history is None:
            message_history = []

        # Generate unique conversation ID (OpenAI expects None for new conversations)
        # We'll let OpenAI generate the conversation ID
        conversation_id = None

        return SabreAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            conversation_id=conversation_id,
            turn_number=0
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: SabreAgentState
    ) -> tuple[AssistantMessage, SabreAgentState]:
        """
        Generate next assistant message using SABRE.

        This is the main method called by tau2-bench for each turn.

        Args:
            message: User or tool message(s) from tau2-bench
            state: Current agent state

        Returns:
            Tuple of (assistant_message, updated_state)
        """
        # Update state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        state.turn_number += 1

        logger.warning(f"🚀 SABRE AGENT CALLED - Turn {state.turn_number}: Processing message with SABRE")
        logger.info(f"Turn {state.turn_number}: Processing message with SABRE")

        # Log the incoming message
        if isinstance(message, UserMessage):
            logger.warning(f"💬 USER MESSAGE: {message.content[:200]}{'...' if len(message.content) > 200 else ''}")
        elif isinstance(message, ToolMessage):
            logger.warning(f"🔧 TOOL RESULT: {str(message.content)[:200]}{'...' if len(str(message.content)) > 200 else ''}")
        elif isinstance(message, MultiToolMessage):
            logger.warning(f"🔧 MULTIPLE TOOL RESULTS: {len(message.tool_messages)} results")
            for i, tm in enumerate(message.tool_messages[:3]):
                logger.warning(f"   Result {i+1}: {str(tm.content)[:100]}...")

        # Check if this is a tool result message and we have a pending orchestration
        is_tool_result = isinstance(message, (ToolMessage, MultiToolMessage))
        has_pending = state.pending_orchestration_result is not None

        # Run SABRE orchestration (synchronous wrapper for async)
        try:
            if is_tool_result and has_pending:
                # Resume orchestration with tool results
                logger.info("Resuming orchestration with tool results")
                result = asyncio.run(self._resume_sabre_orchestration(message, state))
            else:
                # Start new orchestration
                user_input = self._build_sabre_input(message, state)
                logger.warning(f"📝 SABRE INPUT:\n{user_input}")
                logger.debug(f"Input to SABRE: {user_input[:200]}...")
                result = asyncio.run(self._run_sabre_orchestration(
                    conversation_id=state.conversation_id,
                    input_text=user_input,
                    state=state
                ))

            # Update conversation_id from result if this was the first turn
            if state.conversation_id is None:
                state.conversation_id = result.conversation_id

        except Exception as e:
            logger.error(f"SABRE orchestration failed: {e}", exc_info=True)
            # Return error message
            assistant_message = AssistantMessage(
                role="assistant",
                content=f"I apologize, but I encountered an error: {str(e)}"
            )
            state.messages.append(assistant_message)
            # Clear pending state
            state.pending_orchestration_result = None
            state.execution_tree = None
            return assistant_message, state

        # Parse SABRE's response into tau2-bench format
        assistant_message = self._parse_sabre_response(result, state)

        # Update state
        state.messages.append(assistant_message)

        # Save orchestration state if paused
        if result.status == "awaiting_tool_results":
            state.pending_orchestration_result = result
            logger.info("Saved pending orchestration state")
        else:
            # Clear pending state on completion
            state.pending_orchestration_result = None
            state.execution_tree = None

        logger.info(f"Turn {state.turn_number}: Generated assistant message")
        logger.info(f"SABRE status: {result.status}")

        # Log the response details
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            logger.warning(f"🤖 ASSISTANT RESPONSE: [Tool calls - no text content]")
            logger.warning(f"✓ Assistant message has {len(assistant_message.tool_calls)} tool calls")
            for i, tc in enumerate(assistant_message.tool_calls[:5]):
                # ToolCall is an object, not a dict - access attributes
                tc_name = tc.name if hasattr(tc, 'name') else tc.get('name', 'unknown')
                tc_args = tc.arguments if hasattr(tc, 'arguments') else tc.get('arguments', {})
                arg_keys = list(tc_args.keys()) if isinstance(tc_args, dict) else []
                logger.warning(f"   Tool {i+1}: {tc_name}({arg_keys})")
            logger.info(f"✓ Assistant message has {len(assistant_message.tool_calls)} tool calls")
        else:
            logger.warning(f"🤖 ASSISTANT RESPONSE: {assistant_message.content[:500]}{'...' if len(assistant_message.content) > 500 else ''}")
            logger.info(f"Assistant message content length: {len(assistant_message.content)} chars")
            logger.info(f"No tool calls in this turn")

        return assistant_message, state

    def _build_sabre_input(
        self,
        message: ValidAgentInputMessage,
        state: SabreAgentState
    ) -> str:
        """
        Build input text for SABRE orchestrator.

        Args:
            message: Current message from tau2-bench
            state: Current agent state

        Returns:
            Formatted input string for SABRE
        """
        # BUGFIX: Limit tool result sizes to prevent context pollution
        MAX_TOOL_RESULT_CHARS = 500

        def truncate_content(content: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
            """Truncate long content but preserve important info"""
            if len(content) <= max_chars:
                return content
            # Truncate and add summary
            return f"{content[:max_chars]}... [truncated {len(content) - max_chars} chars]"

        parts = []

        # Add conversation history context (recent messages)
        if len(state.messages) > 1:
            parts.append("CONVERSATION HISTORY:")
            # Include last few exchanges for context
            recent_messages = state.messages[-6:]  # Last 3 exchanges
            for msg in recent_messages:
                if isinstance(msg, UserMessage):
                    parts.append(f"User: {truncate_content(msg.content)}")
                elif isinstance(msg, AssistantMessage):
                    parts.append(f"Assistant: {truncate_content(msg.content)}")
                elif isinstance(msg, ToolMessage):
                    parts.append(f"Tool result: {truncate_content(str(msg.content))}")
            parts.append("")

        # Add current message
        if isinstance(message, UserMessage):
            parts.append("CURRENT USER MESSAGE:")
            parts.append(message.content)
        elif isinstance(message, ToolMessage):
            parts.append("TOOL RESULT:")
            parts.append(truncate_content(str(message.content)))
        elif isinstance(message, MultiToolMessage):
            parts.append("TOOL RESULTS:")
            for tool_msg in message.tool_messages:
                # ToolMessage doesn't have 'name', only 'id'
                # For display purposes, use id or just show content
                parts.append(f"- Tool {tool_msg.id}: {truncate_content(str(tool_msg.content))}")

        return "\n".join(parts)

    async def _run_sabre_orchestration(
        self,
        conversation_id: str,
        input_text: str,
        state: SabreAgentState
    ):
        """
        Run SABRE orchestration using the full orchestrator.

        This uses the enhanced orchestrator which supports external tools.
        If the LLM calls external tau2 tools, orchestration will pause and
        return pending_tool_calls.

        Args:
            conversation_id: Unique conversation identifier (None for first turn)
            input_text: User input text
            state: Current agent state

        Returns:
            OrchestrationResult with response and possibly pending_tool_calls
        """
        try:
            # Create conversation if this is the first call
            if conversation_id is None:
                logger.info("Creating new OpenAI conversation")
                conversation = await self.executor.client.conversations.create(
                    metadata={"agent": "sabre_tau2", "domain": "tau2_bench"}
                )
                conversation_id = conversation.id
                logger.info(f"Created conversation: {conversation_id}")

            # Create execution tree for tracking (or reuse from state if resuming)
            tree = state.execution_tree if state.execution_tree else ExecutionTree()

            # Run full SABRE orchestration with tool registry
            result = await self.orchestrator.run(
                conversation_id=conversation_id,
                input_text=input_text,
                instructions=state.system_messages[0].content,
                tree=tree,
                model=self.model,
                max_tokens=4096,
                temperature=self.llm_args.get('temperature', 0.0)
            )

            # Save tree in state for potential resumption
            state.execution_tree = tree

            logger.info(f"Orchestration result: status={result.status}, success={result.success}")
            if result.pending_tool_calls:
                logger.info(f"Pending external tool calls: {len(result.pending_tool_calls)}")
                for tc in result.pending_tool_calls[:3]:
                    logger.info(f"  - {tc['name']}")

            return result

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            # Return error result
            from sabre.server.orchestrator import OrchestrationResult
            return OrchestrationResult(
                success=False,
                final_response="",
                conversation_id=conversation_id or "unknown",
                response_id="error",
                error=str(e),
                status="error"
            )

    async def _resume_sabre_orchestration(
        self,
        tool_message: ValidAgentInputMessage,
        state: SabreAgentState
    ):
        """
        Resume SABRE orchestration with external tool results.

        Called when tau2-bench has executed external tools and provides results.

        Args:
            tool_message: ToolMessage or MultiToolMessage with execution results
            state: Current agent state with pending_orchestration_result

        Returns:
            OrchestrationResult (may pause again or complete)
        """
        if not state.pending_orchestration_result:
            raise ValueError("No pending orchestration to resume")

        # Build mapping from tool call ID to tool name from pending tool calls
        tool_id_to_name = {}
        if state.pending_orchestration_result.pending_tool_calls:
            for tc in state.pending_orchestration_result.pending_tool_calls:
                # ToolCall has both id and name
                tc_id = tc.id if hasattr(tc, 'id') else tc.get('id', '')
                tc_name = tc.name if hasattr(tc, 'name') else tc.get('name', 'unknown')
                tool_id_to_name[tc_id] = tc_name

        # Extract tool results from message(s)
        # ToolMessage has 'id' (not 'name') - need to map to tool call name
        tool_results = []
        if isinstance(tool_message, ToolMessage):
            tool_name = tool_id_to_name.get(tool_message.id, 'unknown_tool')
            tool_results.append({
                "name": tool_name,
                "result": str(tool_message.content)
            })
        elif isinstance(tool_message, MultiToolMessage):
            for tm in tool_message.tool_messages:
                tool_name = tool_id_to_name.get(tm.id, 'unknown_tool')
                tool_results.append({
                    "name": tool_name,
                    "result": str(tm.content)
                })

        logger.warning(f"🔄 CONVERTING {len(tool_results)} tau2-bench TOOL RESULTS to SABRE format")
        for i, result in enumerate(tool_results):
            logger.warning(f"   Result {i+1}: {result['name']}: {str(result['result'])[:100]}...")
        logger.info(f"Resuming orchestration with {len(tool_results)} tool results")

        # Get execution tree from state
        tree = state.execution_tree if state.execution_tree else ExecutionTree()

        try:
            # Resume orchestration via continue_with_tool_results
            result = await self.orchestrator.continue_with_tool_results(
                conversation_id=state.conversation_id,
                tool_results=tool_results,
                tree=tree,
                instructions=state.system_messages[0].content,
                model=self.model,
                max_tokens=4096,
                temperature=self.llm_args.get('temperature', 0.0)
            )

            # Update tree in state
            state.execution_tree = tree

            logger.info(f"Resumed orchestration result: status={result.status}, success={result.success}")

            return result

        except Exception as e:
            logger.error(f"Failed to resume orchestration: {e}", exc_info=True)
            from sabre.server.orchestrator import OrchestrationResult
            return OrchestrationResult(
                success=False,
                final_response="",
                conversation_id=state.conversation_id or "unknown",
                response_id="error",
                error=str(e),
                status="error"
            )

    def _parse_sabre_response(
        self,
        sabre_result,
        state: SabreAgentState
    ) -> AssistantMessage:
        """
        Parse SABRE's OrchestrationResult into tau2-bench AssistantMessage.

        Handles both completed responses and paused orchestrations with pending tool calls.

        Args:
            sabre_result: OrchestrationResult from SABRE orchestration
            state: Current agent state

        Returns:
            AssistantMessage for tau2-bench
        """
        logger.warning(f"🔄 PARSING SABRE RESPONSE - Status: {sabre_result.status}")

        # Check for errors
        if sabre_result.status == "error" or (not sabre_result.success and not sabre_result.pending_tool_calls):
            content = f"I apologize, but I encountered an issue: {sabre_result.error}"
            logger.warning(f"❌ Error response: {content}")
            return AssistantMessage(role="assistant", content=content)

        # Check if orchestration paused with external tool calls
        if sabre_result.status == "awaiting_tool_results" and sabre_result.pending_tool_calls:
            logger.warning(f"🔧 CONVERTING {len(sabre_result.pending_tool_calls)} SABRE TOOL CALLS to tau2-bench format")
            for i, tc in enumerate(sabre_result.pending_tool_calls):
                tc_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                tc_args = tc.get('arguments', {}) if isinstance(tc, dict) else getattr(tc, 'arguments', {})
                logger.warning(f"   Tool {i+1}: {tc_name}({list(tc_args.keys()) if isinstance(tc_args, dict) else 'N/A'})")
            logger.info(f"Orchestration paused with {len(sabre_result.pending_tool_calls)} external tool calls")

            # Return AssistantMessage with tool_calls for tau2-bench to execute
            assistant_msg = AssistantMessage(
                role="assistant",
                content="",  # Empty content when making tool calls
                tool_calls=sabre_result.pending_tool_calls
            )
            logger.warning(f"✅ Created tau2-bench AssistantMessage with {len(sabre_result.pending_tool_calls)} tool_calls")
            return assistant_msg

        # Orchestration completed - extract response text
        logger.warning(f"✅ ORCHESTRATION COMPLETE - Returning final response")
        response_text = sabre_result.final_response

        # Clean up response (remove SABRE artifacts like <helpers_result> tags)
        clean_content = self._clean_response_for_user(response_text)

        # Ensure content is not empty
        if not clean_content or clean_content.strip() == "":
            clean_content = "I understand. How can I help you further?"

        return AssistantMessage(
            role="assistant",
            content=clean_content
        )

    def _extract_tool_calls_from_response(self, response: str) -> Optional[list]:
        """
        Extract tool calls from SABRE's response.

        Looks for <helpers> blocks and parses function calls.

        Args:
            response: SABRE's response text

        Returns:
            List of tool call dicts in tau2-bench format, or None
        """
        import re
        import ast
        import uuid
        import json

        # Find <helpers> blocks
        helpers_pattern = r'<helpers>(.*?)</helpers>'
        helpers_matches = re.findall(helpers_pattern, response, re.DOTALL)

        if not helpers_matches:
            return None

        tool_calls = []

        for helpers_code in helpers_matches:
            try:
                # Parse Python code
                tree = ast.parse(helpers_code)

                # Find function calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Get function name
                        if isinstance(node.func, ast.Name):
                            func_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            func_name = node.func.attr
                        else:
                            continue

                        # Skip internal functions
                        if func_name in ['result', 'print', 'len', 'str', 'int']:
                            continue

                        # Extract arguments
                        arguments = {}

                        for i, arg in enumerate(node.args):
                            arguments[f"arg{i}"] = self._extract_ast_value(arg)

                        for keyword in node.keywords:
                            arguments[keyword.arg] = self._extract_ast_value(keyword.value)

                        # Create tool call in tau2-bench format
                        tool_call = {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "name": func_name,
                            "arguments": arguments  # tau2 expects dict, not JSON string
                        }
                        tool_calls.append(tool_call)

            except SyntaxError:
                logger.warning(f"Failed to parse helpers block: {helpers_code[:100]}")
                continue

        return tool_calls if tool_calls else None

    def _extract_ast_value(self, node):
        """Extract Python value from AST node."""
        import ast

        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.List):
            return [self._extract_ast_value(elem) for elem in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._extract_ast_value(k): self._extract_ast_value(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Name):
            return f"${node.id}"  # Variable reference
        else:
            return str(node)

    def _clean_response_for_user(self, response: str) -> str:
        """
        Clean SABRE response for user display.

        Removes <helpers> blocks and other SABRE artifacts.

        Args:
            response: Raw SABRE response

        Returns:
            Cleaned response text
        """
        import re

        # Remove <helpers> blocks
        clean = re.sub(r'<helpers>.*?</helpers>', '', response, flags=re.DOTALL)

        # Remove <helpers_result> blocks
        clean = re.sub(r'<helpers_result>.*?</helpers_result>', '', clean, flags=re.DOTALL)

        # Remove excessive whitespace
        clean = re.sub(r'\n\s*\n', '\n\n', clean)

        return clean.strip()

    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        # SABRE/OpenAI doesn't support seeds directly, but we can try
        logger.info(f"Setting seed={seed} (note: OpenAI may not respect this)")
        # Store for potential future use
        self._seed = seed
