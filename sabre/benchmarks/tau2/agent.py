"""
SABRE agent adapter for tau2-bench evaluations.

This module provides an interface for tau2-bench to evaluate SABRE's
performance on multi-turn conversational tasks with tool use.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
import httpx
import time


class SabreTau2Agent:
    """
    Adapter that allows tau2-bench to evaluate SABRE.

    This implements an OpenAI-compatible API that tau2-bench can call,
    while internally using SABRE's recursive execution engine.

    Usage:
        agent = SabreTau2Agent(
            sabre_port=8011,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

        response = await agent.process_chat_completion(
            messages=[...],
            tools=[...]
        )
    """

    def __init__(
        self,
        sabre_port: int = 8011,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ):
        """
        Initialize SABRE tau2-bench adapter.

        Args:
            sabre_port: Port where SABRE server is running
            openai_api_key: OpenAI API key for SABRE
            model: Model name to report to tau2-bench
        """
        self.sabre_port = sabre_port
        self.sabre_url = f"http://localhost:{sabre_port}"
        self.model = model
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        # Track conversations for multi-turn dialogues
        self.conversations: Dict[str, List[Dict]] = {}

    async def process_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat completion request from tau2-bench.

        Args:
            messages: Conversation history in OpenAI format
                     [{"role": "system"|"user"|"assistant", "content": "..."}]
            tools: Available tools (tau2-bench API functions)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            OpenAI-format response with SABRE's output
        """
        # Extract system message (domain policy) and conversation
        system_message = ""
        conversation = []

        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                conversation.append(msg)

        # Format tools for SABRE
        tools_context = self._format_tools(tools) if tools else ""

        # Build the prompt for SABRE
        prompt = self._build_sabre_prompt(
            system_message,
            tools_context,
            conversation
        )

        # Generate valid conversation ID (OpenAI requires 'conv' prefix)
        # Use provided conv_id or generate one with timestamp
        conv_id = kwargs.get('conv_id')
        if not conv_id or not conv_id.startswith('conv'):
            conv_id = f"conv_tau2_{int(time.time() * 1000)}"

        # Call SABRE server (which returns SSE)
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # SABRE uses SSE, so we need to stream the response
                async with client.stream(
                    "POST",
                    f"{self.sabre_url}/message",
                    json={
                        "content": prompt,  # SABRE expects 'content' not 'message'
                        "conversation_id": conv_id
                    }
                ) as response:
                    if response.status_code != 200:
                        return self._error_response(
                            f"SABRE server error: {response.status_code}"
                        )

                    # Parse SSE stream to collect response
                    response_text = await self._parse_sse_stream(response)

        except httpx.ConnectError:
            return self._error_response(
                f"Cannot connect to SABRE server at {self.sabre_url}. "
                "Make sure SABRE server is running."
            )
        except Exception as e:
            return self._error_response(f"Error calling SABRE: {str(e)}")

        # Parse SABRE's response for tool calls
        # SABRE may return Python code that calls helpers
        tool_calls = self._extract_tool_calls(response_text)

        # Return in OpenAI format
        return self._format_openai_response(response_text, tool_calls)

    async def _parse_sse_stream(self, response) -> str:
        """
        Parse SABRE's SSE stream and extract the response text.

        SABRE returns Server-Sent Events with jsonpickle-encoded events:
        data: {"py/object": "sabre.common.models.events.ResponseTextEvent", "data": {"text": "..."}}
        data: {"py/object": "sabre.common.models.events.CompleteEvent", "data": {"final_message": "..."}}
        data: [DONE]

        Args:
            response: httpx streaming response

        Returns:
            Collected response text
        """
        import json
        import jsonpickle

        response_parts = []
        final_message = None

        async for line in response.aiter_lines():
            if not line or line.strip() == ":":
                continue

            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                # Skip __REQUEST_ID__ lines
                if data_str.startswith("__REQUEST_ID__"):
                    continue

                try:
                    # Try to decode jsonpickle first
                    try:
                        event = jsonpickle.decode(data_str)

                        # Check for response_text event
                        if hasattr(event, 'data') and isinstance(event.data, dict):
                            if 'text' in event.data:
                                text = event.data['text']
                                if text:
                                    response_parts.append(text)

                            # Check for final_message in CompleteEvent
                            if 'final_message' in event.data:
                                final_message = event.data['final_message']

                    except:
                        # Fall back to regular JSON parsing
                        data = json.loads(data_str)

                        # Try to extract text from various formats
                        if isinstance(data, dict):
                            if 'data' in data and isinstance(data['data'], dict):
                                if 'text' in data['data']:
                                    text = data['data']['text']
                                    if text:
                                        response_parts.append(text)
                                if 'final_message' in data['data']:
                                    final_message = data['data']['final_message']

                except (json.JSONDecodeError, Exception):
                    # Skip malformed lines
                    continue

        # Use final_message if available, otherwise join response_parts
        if final_message:
            return final_message
        return "".join(response_parts)

    def _format_tools(self, tools: List[Dict]) -> str:
        """
        Format tau2-bench tools for SABRE's context.

        Args:
            tools: List of tool definitions in OpenAI format

        Returns:
            Formatted string describing available tools
        """
        if not tools:
            return ""

        tools_text = "Available Tools (call these as Python functions):\n"
        for tool in tools:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            desc = func.get('description', '')
            params = func.get('parameters', {})

            # Format function signature
            signature = f"{name}({self._format_params(params)})"
            tools_text += f"\n{signature}\n"
            tools_text += f"  Description: {desc}\n"

            # Add parameter details
            if params and 'properties' in params:
                tools_text += "  Parameters:\n"
                for param_name, param_schema in params['properties'].items():
                    param_desc = param_schema.get('description', '')
                    param_type = param_schema.get('type', 'any')
                    required = param_name in params.get('required', [])
                    req_str = " (required)" if required else " (optional)"
                    tools_text += f"    - {param_name}: {param_type}{req_str} - {param_desc}\n"

        return tools_text

    def _format_params(self, params: Dict) -> str:
        """
        Format parameter schema into readable function signature.

        Args:
            params: Parameter schema from tool definition

        Returns:
            Formatted parameter list (e.g., "name: str, age: int = None")
        """
        if not params or 'properties' not in params:
            return ""

        props = params['properties']
        required = params.get('required', [])

        param_strs = []
        for name, schema in props.items():
            type_str = schema.get('type', 'any')
            if name in required:
                param_strs.append(f"{name}: {type_str}")
            else:
                param_strs.append(f"{name}: {type_str} = None")

        return ", ".join(param_strs)

    def _build_sabre_prompt(
        self,
        system_message: str,
        tools_context: str,
        conversation: List[Dict]
    ) -> str:
        """
        Build comprehensive prompt for SABRE.

        Args:
            system_message: Domain policy and instructions
            tools_context: Formatted tool descriptions
            conversation: Message history

        Returns:
            Complete prompt string for SABRE
        """
        prompt_parts = []

        # Add domain policy
        if system_message:
            prompt_parts.append("DOMAIN POLICY AND INSTRUCTIONS:")
            prompt_parts.append(system_message)
            prompt_parts.append("")

        # Add tools
        if tools_context:
            prompt_parts.append(tools_context)
            prompt_parts.append("")

        # Add conversation history (all but last message)
        if len(conversation) > 1:
            prompt_parts.append("CONVERSATION HISTORY:")
            for msg in conversation[:-1]:
                role = msg['role'].upper()
                content = msg['content']
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")

        # Add current user message
        if conversation:
            current_msg = conversation[-1]
            if current_msg['role'] == 'user':
                prompt_parts.append("CURRENT USER REQUEST:")
                prompt_parts.append(current_msg['content'])
            else:
                # If last message is assistant (shouldn't happen), just add it
                prompt_parts.append(current_msg['content'])

        return "\n".join(prompt_parts)

    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """
        Extract tool calls from SABRE's response.

        SABRE may return Python code that calls helpers.
        This parses it into OpenAI tool_calls format.

        Args:
            response: SABRE's response text

        Returns:
            List of tool calls in OpenAI format
        """
        # TODO: Implement parser for SABRE's helper calls
        # For now, return empty list - tau2-bench can work without this
        # as long as SABRE's text response correctly describes the actions
        return []

    def _format_openai_response(
        self,
        content: str,
        tool_calls: List[Dict]
    ) -> Dict[str, Any]:
        """
        Format SABRE response in OpenAI API format.

        Args:
            content: SABRE's response text
            tool_calls: Extracted tool calls

        Returns:
            OpenAI-compatible response dictionary
        """
        message = {
            "role": "assistant",
            "content": content
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-sabre-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {
                "prompt_tokens": len(content) // 4,  # Rough estimate
                "completion_tokens": len(content) // 4,
                "total_tokens": len(content) // 2
            }
        }

    def _error_response(self, error: str) -> Dict[str, Any]:
        """
        Return error in OpenAI format.

        Args:
            error: Error message

        Returns:
            OpenAI-compatible error response
        """
        return {
            "error": {
                "message": error,
                "type": "sabre_error",
                "code": "sabre_connection_error"
            }
        }
