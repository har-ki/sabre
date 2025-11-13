"""SABRE runner for tau2 tasks using DIALOGUE MODE with full user simulator.

This version uses tau2-mcp's dialogue mode (--dialogue-mode flag) to enable:
- Real multi-turn conversations with user simulator
- Complete tau2-bench evaluation (DB + COMMUNICATE + ACTION)
- Natural conversation flow

Differences from sabre_tau2_runner.py:
- Uses --dialogue-mode flag when connecting to tau2-mcp
- Uses tau2.get_next_message() and tau2.send_agent_message() for conversation
- Gets full evaluation including COMMUNICATE scores
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from sabre.server.mcp.models import MCPServerConfig, MCPTransportType
from sabre.server.mcp.client_manager import MCPClientManager
from sabre.server.mcp.helper_adapter import MCPHelperAdapter
from sabre.server.orchestrator import Orchestrator
from sabre.server.python_runtime import PythonRuntime
from sabre.common.executors.response import ResponseExecutor
from sabre.common import ExecutionTree

logger = logging.getLogger(__name__)


def extract_tool_calls_from_helpers_code(helpers_code: str, block_idx: int = 0) -> list[dict]:
    """Extract tool calls from a <helpers> code block.

    Args:
        helpers_code: Python code from a <helpers> block
        block_idx: Index of this helpers block (for generating unique IDs)

    Returns:
        List of tool call dicts in tau2 format
    """
    import re
    import ast
    import inspect

    tool_calls = []

    # Look for tau2_mcp.* calls
    # Pattern: tau2_mcp.function_name(...)
    # We need to match the full function call, handling nested parens
    call_pattern = r'tau2_mcp\.(\w+)\s*\('

    for match in re.finditer(call_pattern, helpers_code):
        tool_name = match.group(1)
        start_pos = match.end() - 1  # Position of opening '('

        # Find matching closing paren
        paren_count = 1
        pos = start_pos + 1
        in_string = False
        string_char = None

        while pos < len(helpers_code) and paren_count > 0:
            char = helpers_code[pos]

            if char in ['"', "'"] and (pos == 0 or helpers_code[pos-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif not in_string:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1

            pos += 1

        if paren_count == 0:
            # Extract arguments string
            args_str = helpers_code[start_pos + 1:pos - 1].strip()

            try:
                # Try to parse as Python function call using ast
                # Create a fake function call to parse
                fake_call = f"f({args_str})"
                try:
                    tree = ast.parse(fake_call, mode='eval')
                    call_node = tree.body

                    args_dict = {}

                    # Handle positional arguments - need to map to parameter names
                    # For tau2 tools, we'll use generic names or try to infer from tool schema
                    for idx, arg in enumerate(call_node.args):
                        # Try to evaluate the argument
                        try:
                            val = ast.literal_eval(ast.unparse(arg))
                        except:
                            val = ast.unparse(arg)

                        # For tau2 tools, common first parameter names:
                        # - order_id, user_id, email, etc.
                        # Use a generic name for now
                        param_names = {
                            'get_order_details': ['order_id'],
                            'get_user_details': ['user_id'],
                            'get_product_details': ['product_id'],
                            'find_user_id_by_email': ['email'],
                            'find_user_id_by_name_zip': ['first_name', 'last_name', 'zip'],
                            'cancel_pending_order': ['order_id'],
                            'modify_pending_order_address': ['order_id', 'address_line_1', 'address_line_2', 'city', 'state', 'country', 'zip'],
                            'exchange_delivered_order_items': ['order_id', 'item_idx', 'new_product_id'],
                            'return_delivered_order_items': ['order_id', 'item_idx'],
                        }

                        if tool_name in param_names and idx < len(param_names[tool_name]):
                            args_dict[param_names[tool_name][idx]] = val
                        else:
                            args_dict[f'arg_{idx}'] = val

                    # Handle keyword arguments
                    for keyword in call_node.keywords:
                        try:
                            val = ast.literal_eval(ast.unparse(keyword.value))
                        except:
                            val = ast.unparse(keyword.value)
                        args_dict[keyword.arg] = val

                    tool_calls.append({
                        "id": f"call_{block_idx}_{len(tool_calls)}_{tool_name}",
                        "name": tool_name,
                        "arguments": args_dict
                    })

                except Exception as e:
                    logger.debug(f"Could not parse arguments for {tool_name} using AST: {e}")
                    logger.debug(f"Arguments string: {args_str}")
                    # Fall back to empty arguments
                    tool_calls.append({
                        "id": f"call_{block_idx}_{len(tool_calls)}_{tool_name}",
                        "name": tool_name,
                        "arguments": {}
                    })

            except Exception as e:
                logger.debug(f"Could not extract arguments for {tool_name}: {e}")
                tool_calls.append({
                    "id": f"call_{block_idx}_{len(tool_calls)}_{tool_name}",
                    "name": tool_name,
                    "arguments": {}
                })

    return tool_calls


async def run_sabre_dialogue_mode(
    task_id: str,
    domain: str = "retail",
    tau2_data_dir: Optional[str] = None,
    model: str = "gpt-4o",
    max_iterations: int = 10,
    max_turns: int = 20,
) -> dict:
    """
    Run SABRE on a tau2 task using DIALOGUE MODE.

    This uses tau2-mcp's full dialogue mode with user simulator for complete
    multi-turn conversation evaluation.

    Args:
        task_id: Task ID to evaluate
        domain: Domain name (default: "retail")
        tau2_data_dir: Path to tau2 data directory (uses TAU2_DATA_DIR env if not provided)
        model: Model to use (default: "gpt-4o")
        max_iterations: Max orchestrator iterations per turn
        max_turns: Max conversation turns

    Returns:
        Dictionary with evaluation results including COMMUNICATE scores
    """
    # Get tau2 data directory
    tau2_data_dir = tau2_data_dir or os.getenv("TAU2_DATA_DIR")
    if not tau2_data_dir:
        raise ValueError("TAU2_DATA_DIR must be set")

    print(f"\n{'='*70}")
    print(f"SABRE x tau2-bench DIALOGUE MODE Evaluation")
    print(f"{'='*70}")
    print(f"Task ID: {task_id}")
    print(f"Domain: {domain}")
    print(f"Model: {model}")
    print(f"Mode: DIALOGUE (with user simulator)")
    print(f"{'='*70}\n")

    # Step 1: Configure tau2-mcp server in DIALOGUE MODE
    print("[1/7] Configuring tau2-mcp server (dialogue mode)...")

    # Find tau2-mcp executable
    tau2_mcp_path = os.getenv("TAU2_MCP_PATH")
    if not tau2_mcp_path:
        possible_paths = [
            "/Users/hnayak/Documents/workspace/tau2-bench/.venv/bin/tau2-mcp",
            os.path.expanduser("~/.local/bin/tau2-mcp"),
        ]
        for path in possible_paths:
            if Path(path).exists():
                tau2_mcp_path = path
                break

    if not tau2_mcp_path or not Path(tau2_mcp_path).exists():
        raise FileNotFoundError(
            "tau2-mcp not found. Set TAU2_MCP_PATH environment variable."
        )

    # CRITICAL: Add --dialogue-mode flag to enable full conversation mode
    tau2_server_config = MCPServerConfig(
        name="tau2_mcp",
        type=MCPTransportType.STDIO,
        command=tau2_mcp_path,
        args=[
            "--eval-mode",
            "--domain", domain,
            "--task-id", task_id,
            "--dialogue-mode",  # Enable dialogue mode!
            "--log-level", "INFO"  # Changed to INFO to see debug logs
        ],
        env={"TAU2_DATA_DIR": tau2_data_dir},
    )
    print(f"   ✓ Configured tau2-mcp in DIALOGUE MODE for task {task_id}")

    # Step 2: Create MCP client manager
    print("\n[2/7] Initializing SABRE MCP client manager...")
    client_manager = MCPClientManager()
    print("   ✓ MCP client manager created")

    try:
        # Step 3: Connect to tau2-mcp server
        print("\n[3/7] Connecting to tau2-mcp (dialogue mode)...")
        await client_manager.connect(tau2_server_config)
        print("   ✓ Connected to tau2-mcp with dialogue mode enabled")

        client = client_manager.get_client_by_name("tau2_mcp")
        if not client:
            raise RuntimeError("Failed to get tau2_mcp client")

        # Step 4: Get task information
        print("\n[4/7] Getting task information...")
        task_info_result = await client.call_tool("tau2.get_task_info", {})
        task_info = json.loads(task_info_result.content[0].text)

        if "error" in task_info:
            raise RuntimeError(f"tau2-mcp error: {task_info['error']}")

        task_goal = task_info["goal"]
        print(f"   ✓ Task: {task_goal[:100]}...")

        # Verify dialogue mode is enabled
        state_result = await client.call_tool("tau2.get_conversation_state", {})
        state = json.loads(state_result.content[0].text)
        if not state.get("dialogue_mode"):
            raise RuntimeError("Dialogue mode not enabled in tau2-mcp server!")
        print(f"   ✓ Dialogue mode active")

        # Step 5: Setup SABRE components
        print("\n[5/7] Setting up SABRE components...")
        mcp_adapter = MCPHelperAdapter(client_manager)
        await mcp_adapter.refresh_tools()

        executor = ResponseExecutor()
        runtime = PythonRuntime(mcp_adapter=mcp_adapter)
        orchestrator = Orchestrator(
            executor=executor,
            python_runtime=runtime,
            max_iterations=max_iterations,
        )
        runtime.set_orchestrator(orchestrator)

        print(f"   ✓ SABRE ready with {len(mcp_adapter.get_available_tools())} tools")

        # Step 6: Run dialogue conversation
        print("\n[6/7] Starting dialogue conversation...")
        conversation_history = []
        turn = 0

        while turn < max_turns:
            turn += 1
            print(f"\n   --- Turn {turn} ---")

            # Get next message from user simulator
            msg_result = await client.call_tool("tau2.get_next_message", {})
            msg_data = json.loads(msg_result.content[0].text)

            # Debug: print the full message data
            # Removed turn limit to see all messages
            print(f"   DEBUG: msg_data = {json.dumps(msg_data, indent=2)}")

            # Check for errors from tau2-mcp
            if msg_data.get("status") == "error":
                print(f"   ⚠️ Error from tau2-mcp: {msg_data.get('error')}")
                raise RuntimeError(f"tau2-mcp error: {msg_data.get('error')}")

            if msg_data.get("conversation_complete"):
                print(f"   ✓ Conversation complete")
                break

            message = msg_data.get("message")
            if not message:
                print(f"   ⚠️ No message in response: {json.dumps(msg_data, indent=2)}")
                raise RuntimeError(f"No message in tau2.get_next_message response")
            role = message["role"]
            content = message.get("content", "")

            # Display message (handle None content)
            content_preview = (content[:80] if content else "[no content]")
            print(f"   {role.upper()}: {content_preview}...")

            # Handle tool calls in user messages
            if "tool_calls" in message and message["tool_calls"]:
                print(f"   ⚠️ User message contains tool_calls - this indicates user simulator is trying to call tools")
                print(f"   This shouldn't happen in normal flow. Tau2 might be confused.")

            conversation_history.append(message)

            # Handle messages based on role
            # In tau2 dialogue mode, `get_next_message` returns messages in order.
            # When we see an "assistant" message, it means tau2's greeting has been sent.
            # We should just acknowledge it's been received and NOT send another message.
            # The next call to `get_next_message` will give us the user's response.

            if role == "assistant":
                # Assistant messages in dialogue mode can be:
                # 1. Initial greeting from tau2 (turn=0, no tool_calls)
                # 2. Echo of our own message if there was an error (has tool_calls or turn > 0)

                # Check if this is our own message being echoed back (error case)
                if "tool_calls" in message and message["tool_calls"]:
                    print(f"   ⚠️ Assistant message with tool_calls - likely an error echo, skipping")
                    continue

                if msg_data.get("turn", 0) > 0:
                    print(f"   ⚠️ Assistant message at turn {msg_data.get('turn')} - unexpected, skipping")
                    continue

                # This is the initial greeting at turn 0
                print(f"   (Received initial greeting from tau2)")

                # Send a proper greeting that indicates we're ready to help
                greeting_response = "Hello! I'm here to assist you. What can I help you with today?"
                send_result = await client.call_tool("tau2.send_agent_message", {
                    "content": greeting_response,
                    "tool_calls": []
                })
                send_data = json.loads(send_result.content[0].text)

                if send_data.get("status") != "success":
                    print(f"   ⚠️ Warning: greeting send failed: {send_data}")
                    break

                if not send_data.get("conversation_continues", True):
                    print(f"   ⚠️ Conversation ended after greeting")
                    break

                continue

            elif role == "user":
                # Agent's turn to respond to actual user message

                # Handle user messages with tool_calls (user-side tool calls)
                # In tau2, user simulator can call certain tools (e.g., find_user_id_by_name_zip)
                # These are user-side tools that SABRE doesn't need to handle
                # The tau2-mcp server should auto-execute these and return the next message
                # But if we see them here, skip to next message
                if "tool_calls" in message and message["tool_calls"]:
                    print(f"   ℹ️  User simulator calling user-side tools (tau2 feature) - getting next message...")
                    continue

                if not content:
                    print(f"   ⚠️ User message has no content and no tool_calls - unexpected state")
                    print(f"   Ending dialogue.")
                    break

                print(f"   SABRE processing user message...")

                # Create instructions for this turn
                instructions = _create_dialogue_instructions(
                    task_goal, content, conversation_history, mcp_adapter, orchestrator
                )

                # Run SABRE to generate response
                # Use event callback to capture helpers blocks
                captured_helpers = []

                async def capture_helpers_callback(event):
                    """Capture helpers blocks as they're extracted"""
                    from sabre.common.models.events import HelpersExecutionStartEvent
                    if isinstance(event, HelpersExecutionStartEvent):
                        # Extract code from event data
                        code = event.data.get("code", "")
                        if code:
                            captured_helpers.append(code)

                tree = ExecutionTree()
                result = await orchestrator.run(
                    conversation_id=None,
                    input_text=content,
                    tree=tree,
                    instructions=instructions,
                    model=model,
                    event_callback=capture_helpers_callback,
                )

                agent_response = result.final_response or "I understand. Let me help with that."
                print(f"   AGENT RESPONSE: {agent_response[:80]}...")

                # Ensure agent_response is a non-empty string
                if not agent_response or not agent_response.strip():
                    agent_response = "I understand. Let me help with that."
                    print(f"   ⚠️ Warning: Empty agent response, using default")

                # Extract tool calls from captured helpers blocks
                tool_calls_executed = []
                if captured_helpers and turn <= 3:
                    print(f"   DEBUG: Captured {len(captured_helpers)} helpers blocks:")
                    for idx, code in enumerate(captured_helpers):
                        print(f"   DEBUG: Block {idx}:")
                        print(f"   {code}")
                        print(f"   ---")

                for block_idx, helpers_code in enumerate(captured_helpers):
                    block_tools = extract_tool_calls_from_helpers_code(helpers_code, block_idx)
                    if turn <= 3:
                        print(f"   DEBUG: Block {block_idx} extracted {len(block_tools)} tool calls")
                    tool_calls_executed.extend(block_tools)

                if tool_calls_executed:
                    print(f"   ✓ {len(tool_calls_executed)} tool calls were executed during this turn:")
                    for tc in tool_calls_executed:
                        print(f"     - {tc['name']}({json.dumps(tc['arguments'])})")

                # Send response to tau2-mcp
                # NOTE: We don't send tool_calls because:
                # 1. SABRE already executed tools via MCP (results embedded in response)
                # 2. Sending them would cause double execution by orchestrator
                # 3. tau2 evaluation relies primarily on DB state changes
                # 4. COMMUNICATE evaluation uses message content
                # 5. ACTION evaluation may be limited but DB state is most accurate
                payload = {
                    "content": agent_response,
                    "tool_calls": []  # Don't send - tools already executed
                }

                send_result = await client.call_tool("tau2.send_agent_message", payload)
                send_data = json.loads(send_result.content[0].text)

                if send_data.get("status") != "success":
                    print(f"   ⚠️ Warning: send_agent_message returned: {send_data}")
                    # If there's an error, the dialogue may not be able to continue
                    break

                # Check if conversation should continue
                if not send_data.get("conversation_continues", True):
                    print(f"   ✓ Conversation ended by orchestrator")
                    break

        print(f"\n   ✓ Dialogue completed: {turn} turns")

        # Step 7: Get full evaluation
        print("\n[7/7] Getting evaluation (with COMMUNICATE scores)...")
        eval_result = await client.call_tool("tau2.get_result", {})
        eval_data = json.loads(eval_result.content[0].text)

        print(f"   ✓ Evaluation complete")

        # Display results
        print(f"\n{'='*70}")
        print(f"Dialogue Mode Evaluation Results")
        print(f"{'='*70}")
        print(f"tau2 Score:          {eval_data.get('score', 'N/A')}")
        print(f"tau2 Correct:        {eval_data.get('correct', 'N/A')}")
        print(f"DB Reward:           {eval_data.get('db_reward', 'N/A')}")
        print(f"COMMUNICATE Reward:  {eval_data.get('communicate_reward', 'N/A')}")
        print(f"ACTION Reward:       {eval_data.get('action_reward', 'N/A')}")
        print(f"Details:             {eval_data.get('details', 'N/A')}")
        print(f"Conversation Turns:  {turn}")
        print(f"{'='*70}\n")

        return {
            "task_id": task_id,
            "domain": domain,
            "mode": "dialogue",
            "tau2_score": eval_data["score"],
            "tau2_correct": eval_data["correct"],
            "db_reward": eval_data.get("db_reward"),
            "communicate_reward": eval_data.get("communicate_reward"),
            "action_reward": eval_data.get("action_reward"),
            "details": eval_data["details"],
            "conversation_turns": turn,
            "conversation_history": conversation_history,
        }

    finally:
        await client_manager.disconnect_all()
        print("✓ Disconnected from tau2-mcp")


def _create_dialogue_instructions(
    task_goal: str,
    current_user_message: str,
    conversation_history: list,
    mcp_adapter,
    orchestrator,
) -> str:
    """Create instructions for a single dialogue turn."""
    from sabre.common.utils.prompt_loader import PromptLoader
    from pathlib import Path

    # Get base orchestrator prompt
    default_instructions = orchestrator.load_default_instructions()

    # Load retail guidance
    base_prompt_path = Path(__file__).parent / "prompts" / "retail_agent_base.prompt"
    retail_prompt = PromptLoader.load(
        str(base_prompt_path),
        template={"context_window_tokens": "128000"}
    )
    retail_user_msg = retail_prompt.get("user_message", "")

    # Format conversation history
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_history[-5:]  # Last 5 messages for context
    ])

    instructions = f"""{default_instructions}

## DIALOGUE MODE: Multi-Turn Customer Service Conversation

You are in a REAL CONVERSATION with a customer (simulated user). This is NOT a one-shot task!

**Current Conversation Context:**
{history_text}

**Customer's Current Message:**
USER: {current_user_message}

**Your Role:**
- Respond naturally to the customer's current message
- Use tau2_mcp tools when you need information
- Execute actions when requested (exchanges, cancellations, etc.)
- Ask clarifying questions if needed
- Keep responses conversational and helpful

**Tool Usage:**
- Use tau2_mcp.* tools for all information and actions
- DON'T use web search or external tools
- Execute actions immediately when requested

**Original Task Goal (for context):**
{task_goal}

{retail_user_msg}

**CRITICAL for ACTION tasks:**
- If the customer is requesting an action (exchange, cancel, return), YOU MUST EXECUTE IT
- Don't just gather information - actually call the action tool
- Confirm completion only AFTER seeing the <helpers_result>

**Current Turn:**
Respond to the customer's message above. Use tools as needed.
"""
    return instructions


def main():
    """CLI entry point for dialogue mode runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SABRE on tau2-bench in DIALOGUE MODE (with user simulator)"
    )
    parser.add_argument("task_id", help="Task ID to evaluate")
    parser.add_argument("--domain", default="retail", help="Domain (default: retail)")
    parser.add_argument("--model", default="gpt-4o", help="Model (default: gpt-4o)")
    parser.add_argument("--max-turns", type=int, default=20, help="Max conversation turns")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Setup event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    import nest_asyncio
    nest_asyncio.apply(loop)

    # Run dialogue mode evaluation
    result = loop.run_until_complete(
        run_sabre_dialogue_mode(
            task_id=args.task_id,
            domain=args.domain,
            model=args.model,
            max_turns=args.max_turns,
        )
    )

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
