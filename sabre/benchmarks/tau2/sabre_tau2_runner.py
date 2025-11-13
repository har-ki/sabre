"""Direct runner for SABRE on tau2 tasks using SABRE's native MCP integration.

This uses SABRE's existing MCPClientManager and MCPHelperAdapter to connect to tau2-mcp.
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


async def run_sabre_on_tau2_task(
    task_id: str,
    domain: str = "retail",
    tau2_data_dir: Optional[str] = None,
    model: str = "gpt-4o",
    max_iterations: int = 10,
) -> dict:
    """
    Run SABRE on a tau2 task using tau2-mcp server.

    Args:
        task_id: Task ID to evaluate
        domain: Domain name (default: "retail")
        tau2_data_dir: Path to tau2 data directory (uses TAU2_DATA_DIR env if not provided)
        model: Model to use (default: "gpt-4o")
        max_iterations: Max orchestrator iterations

    Returns:
        Dictionary with evaluation results
    """
    # Get tau2 data directory
    tau2_data_dir = tau2_data_dir or os.getenv("TAU2_DATA_DIR")
    if not tau2_data_dir:
        raise ValueError("TAU2_DATA_DIR must be set")

    print(f"\n{'='*70}")
    print(f"SABRE x tau2-bench Evaluation")
    print(f"{'='*70}")
    print(f"Task ID: {task_id}")
    print(f"Domain: {domain}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    # Step 1: Configure tau2-mcp server as MCP server
    print("[1/6] Configuring tau2-mcp server...")

    # Find tau2-mcp executable
    # Check environment variable first for explicit override
    tau2_mcp_path = os.getenv("TAU2_MCP_PATH")

    if not tau2_mcp_path:
        # Try common locations
        possible_paths = [
            "/Users/hnayak/Documents/workspace/tau2-bench/.venv/bin/tau2-mcp",  # Development path
            os.path.expanduser("~/.local/bin/tau2-mcp"),  # User installation
        ]

        for path in possible_paths:
            if Path(path).exists():
                tau2_mcp_path = path
                break

    if not tau2_mcp_path or not Path(tau2_mcp_path).exists():
        raise FileNotFoundError(
            "tau2-mcp not found. Please set TAU2_MCP_PATH environment variable to point to tau2-mcp executable, "
            "or install it in one of the expected locations:\n"
            "  - /Users/hnayak/Documents/workspace/tau2-bench/.venv/bin/tau2-mcp\n"
            "  - ~/.local/bin/tau2-mcp\n"
            "Example: export TAU2_MCP_PATH=/path/to/tau2-bench/.venv/bin/tau2-mcp"
        )

    tau2_server_config = MCPServerConfig(
        name="tau2_mcp",  # Use underscore for valid Python identifier
        type=MCPTransportType.STDIO,
        command=tau2_mcp_path,
        args=["--eval-mode", "--domain", domain, "--task-id", task_id, "--log-level", "ERROR"],
        env={"TAU2_DATA_DIR": tau2_data_dir},
    )
    print(f"   ✓ Configured tau2-mcp server for task {task_id}")

    # Step 2: Create MCP client manager
    print("\n[2/6] Initializing SABRE MCP client manager...")
    client_manager = MCPClientManager()
    print("   ✓ MCP client manager created")

    try:
        # Step 3: Connect to tau2-mcp server
        print("\n[3/6] Connecting to tau2-mcp server...")
        await client_manager.connect(tau2_server_config)
        print("   ✓ Connected to tau2-mcp")

        # Step 4: Get task information from tau2
        print("\n[4/6] Getting task information from tau2...")

        # Debug: List all registered connectors
        connector_ids = client_manager.list_connector_ids()
        logger.debug(f"Registered connector IDs: {connector_ids}")
        logger.debug(f"name_to_id mapping: {client_manager.name_to_id}")
        logger.debug(f"clients keys: {list(client_manager.clients.keys())}")

        client = client_manager.get_client_by_name("tau2_mcp")
        if not client:
            # Try to get more info about what went wrong
            connector_id = client_manager.name_to_id.get("tau2_mcp")
            if connector_id:
                logger.error(f"Found connector_id {connector_id} for tau2_mcp, but no client!")
                logger.error(f"Available clients: {list(client_manager.clients.keys())}")
            else:
                logger.error(f"No connector_id found for 'tau2_mcp'")
                logger.error(f"Available names: {list(client_manager.name_to_id.keys())}")
            raise RuntimeError(f"Failed to get tau2_mcp client after connection. name_to_id={client_manager.name_to_id}, clients={list(client_manager.clients.keys())}")
        task_info_result = await client.call_tool("tau2.get_task_info", {})
        task_info = json.loads(task_info_result.content[0].text)

        # Check if tau2-mcp returned an error
        if "error" in task_info:
            raise RuntimeError(f"tau2-mcp error: {task_info['error']}")

        task_goal = task_info["goal"]

        print(f"   ✓ Task ID: {task_info['task_id']}")
        print(f"   ✓ Domain: {task_info['domain']}")
        print(f"   ✓ Goal: {task_goal[:100]}...")

        # List available tools
        all_tools = await client_manager.get_all_tools()
        tau2_tools = all_tools.get("tau2-mcp", [])
        print(f"   ✓ Available tools: {len(tau2_tools)}")

        # Step 5: Run SABRE with MCP tools
        print("\n[5/6] Running SABRE agent...")
        print("   Creating SABRE components...")

        # Create MCP helper adapter
        mcp_adapter = MCPHelperAdapter(client_manager)
        await mcp_adapter.refresh_tools()
        print(f"   ✓ MCP tools registered: {len(mcp_adapter.get_available_tools())} tools")

        # Create SABRE components with MCP adapter
        executor = ResponseExecutor()
        runtime = PythonRuntime(mcp_adapter=mcp_adapter)
        orchestrator = Orchestrator(
            executor=executor,
            python_runtime=runtime,
            max_iterations=max_iterations,
        )

        # Set orchestrator reference in runtime (needed for recursive calls)
        runtime.set_orchestrator(orchestrator)

        # MCP tools are automatically added to namespace by PythonRuntime
        mcp_tools = mcp_adapter.get_available_tools()
        tool_count = len([n for n in mcp_tools.keys() if not n.startswith("tau2.")])
        print(f"   ✓ {len(mcp_tools)} MCP tools available in SABRE runtime ({tool_count} callable tools)")

        # Log available tools for debugging
        for tool_name in mcp_tools.keys():
            if not tool_name.startswith("tau2."):
                logger.debug(f"   - {tool_name}")

        # Create system instructions (using default orchestrator prompt + tau2 guidance)
        instructions = _create_system_instructions(task_goal, mcp_adapter, orchestrator)

        # Create execution tree
        tree = ExecutionTree()

        # Run SABRE
        print("   Running SABRE orchestrator...")
        result = await orchestrator.run(
            conversation_id=None,
            input_text=task_goal,
            tree=tree,
            instructions=instructions,
            model=model,
        )

        print(f"   ✓ SABRE completed: {result.success}")

        # Step 6: Get evaluation from tau2
        print("\n[6/6] Getting evaluation from tau2...")
        eval_result = await client.call_tool("tau2.get_result", {})
        eval_data = json.loads(eval_result.content[0].text)

        print(f"   ✓ Evaluation complete")

        # Display results
        print(f"\n{'='*70}")
        print(f"Evaluation Results")
        print(f"{'='*70}")
        print(f"tau2 Score:      {eval_data.get('score', 'N/A')}")
        print(f"tau2 Correct:    {eval_data.get('correct', 'N/A')}")
        print(f"tau2 Details:    {eval_data.get('details', 'N/A')}")
        print(f"SABRE Success:   {result.success}")
        print(f"Tool Calls:      {eval_data.get('tool_calls', 0)}")
        print(f"Eval Type:       {eval_data.get('evaluation_type', 'N/A')}")
        print(f"{'='*70}\n")

        return {
            "task_id": task_id,
            "domain": domain,
            "tau2_score": eval_data["score"],
            "tau2_correct": eval_data["correct"],
            "tau2_details": eval_data["details"],
            "sabre_success": result.success,
            "sabre_response": result.final_response,
            "tool_calls": eval_data.get("tool_calls", 0),
        }

    finally:
        # Cleanup
        await client_manager.disconnect_all()
        print("✓ Disconnected from tau2-mcp")


def _create_system_instructions(task_goal: str, mcp_adapter, orchestrator) -> str:
    """Create system instructions for SABRE with retail customer service expertise.

    Uses the default orchestrator prompt (which includes the full <helpers> execution model
    and all available tools) and appends tau2-specific retail guidance.
    """
    from sabre.common.utils.prompt_loader import PromptLoader
    from pathlib import Path

    # CRITICAL: Use the default orchestrator prompt as base
    # This includes the full <helpers> execution model, all available tools, and examples
    default_instructions = orchestrator.load_default_instructions()

    # Load the retail-specific guidance (for context, not to replace default)
    base_prompt_path = Path(__file__).parent / "prompts" / "retail_agent_base.prompt"
    retail_prompt = PromptLoader.load(
        str(base_prompt_path),
        template={"context_window_tokens": "128000"}  # gpt-4o/gpt-4o-mini context window
    )
    retail_system_msg = retail_prompt.get("system_message", "")
    retail_user_msg = retail_prompt.get("user_message", "")

    # Get full tool documentation from MCP adapter
    mcp_docs = mcp_adapter.generate_documentation()

    # Filter to only show tau2_mcp tools (exclude tau2.* evaluation tools)
    # Parse the documentation and rebuild without tau2.* tools
    doc_lines = []
    skip_next = False
    for line in mcp_docs.split('\n'):
        # Skip lines with tau2.get_task_info and tau2.get_result
        if 'tau2.get_task_info' in line or 'tau2.get_result' in line:
            skip_next = True
            continue
        if skip_next and (line.startswith('**') or line.strip() == ''):
            skip_next = False
            continue
        if not skip_next:
            doc_lines.append(line)

    filtered_docs = '\n'.join(doc_lines)

    # Create TAU2-specific tool usage examples based on discovered tools
    tau2_examples = _generate_tau2_examples(mcp_adapter)

    # Combine default instructions with retail-specific tau2 guidance
    # The default instructions include the full <helpers> execution model and ALL available tools
    # We're just adding domain-specific context and best practices
    instructions = f"""{default_instructions}

## DOMAIN-SPECIFIC GUIDANCE: Retail Customer Service for TAU2

🚨 CRITICAL: This is a CLOSED-DOMAIN retail system evaluation 🚨

**You have access to a complete retail database via MCP tools (tau2_mcp.*)**

**IMPORTANT TOOL USAGE RULES**:
- ✅ ONLY use tau2_mcp.* tools for this task (get_order_details, get_product_details, etc.)
- ❌ DO NOT use Search.web_search() - all data is in the retail database
- ❌ DO NOT use Web.get() - you don't need external websites
- ❌ DO NOT use download() - all product info is in MCP tools
- ✅ ALL product information is available via tau2_mcp.list_all_product_types() and tau2_mcp.get_product_details()
- ✅ ALL user information is available via tau2_mcp.find_user_id_* and tau2_mcp.get_user_details()
- ✅ ALL order information is available via tau2_mcp.get_order_details()

**Why web search won't work**:
- This is a simulated retail environment for evaluation
- Products only exist in the tau2_mcp database, not on the real web
- Web search will return irrelevant results from real e-commerce sites
- You must use the MCP tools to access the simulated retail data

{retail_user_msg}

## TAU2 Task-Specific Examples

{tau2_examples}

## Current Customer Service Task

{task_goal}

## TAU2 Evaluation Requirements

🚨🚨🚨 THIS IS AN ACTION TASK - YOU MUST EXECUTE THE ACTION! 🚨🚨🚨

**CRITICAL RULE**: You CANNOT say "The exchange has been completed" or "The exchange was successful" UNLESS you have:
1. Written a <helpers> block with tau2_mcp.exchange_delivered_order_items(...)
2. Received a <helpers_result> showing the exchange succeeded

**DO NOT HALLUCINATE COMPLETION!** If you claim success without executing the tool, you FAIL!

Your job is to EXECUTE the requested action (exchange, cancellation, etc.), NOT just find and present options!

**Required workflow - MUST follow this order**:
1. ✓ Gather information: user_id, order details, product details
2. ✓ Find replacement items: Use tau2_mcp.get_product_details() to find suitable variants
3. ✓ Get payment method: From user details or order payment_history
4. ✓✓✓ **EXECUTE THE ACTION** - Write <helpers> block calling:
   ```python
   <helpers>
   result = tau2_mcp.exchange_delivered_order_items(
       order_id="#W...",
       item_ids=["old_item_1", "old_item_2"],
       new_item_ids=["new_item_1", "new_item_2"],
       payment_method_id="credit_card_..."
   )
   print(result)
   </helpers>
   ```
5. ✓ Wait for <helpers_result> - You will receive the exchange result
6. ✓ ONLY THEN can you confirm success to the customer

**Examples of WRONG behavior** (DO NOT DO THIS):
❌ "I will now process the exchange..." then claim it's done without calling the tool
❌ "The exchange has been successfully completed" without seeing <helpers_result>
❌ Presenting replacement options and stopping without executing exchange
❌ Saying you'll execute it "next" and then never doing it

**Examples of CORRECT behavior**:
✅ Find items → Call exchange tool in <helpers> → Wait for result → Confirm
✅ "Let me execute the exchange now..." <helpers>exchange_delivered_order_items(...)</helpers> → See result → "Exchange completed!"

**VERIFICATION**: Before you claim completion, ask yourself:
- Did I write a <helpers> block with exchange_delivered_order_items?
- Did I receive a <helpers_result> showing success?
- If NO to either: YOU MUST EXECUTE THE EXCHANGE FIRST!

DO NOT just present options - you must ACTUALLY EXECUTE the exchange/cancel/return using the MCP tools!
"""
    return instructions


def _generate_tau2_examples(mcp_adapter) -> str:
    """Generate TAU2-specific examples based on available tools."""
    # Get available tools
    tools = mcp_adapter.get_available_tools()

    # Check which tools are available
    has_list_products = any('list_all_product_types' in name for name in tools.keys())
    has_find_user = any('find_user_id' in name for name in tools.keys())
    has_exchange = any('exchange_delivered_order_items' in name for name in tools.keys())
    has_get_order = any('get_order_details' in name for name in tools.keys())
    has_get_user = any('get_user_details' in name for name in tools.keys())
    has_get_product = any('get_product_details' in name for name in tools.keys())

    examples = "## TAU2-Specific Tool Usage Examples\n\n"

    # Example 1: Listing product types
    if has_list_products:
        examples += """### Example 1: Listing Product Types

`list_all_product_types()` returns a **dict** mapping product names to product IDs:

```python
# Get product types - returns dict like:
# {"Mechanical Keyboard": "1656367028", "Smart Thermostat": "2345678901", ...}
product_types = tau2_mcp.list_all_product_types()

# Access as dict - keys are product names, values are product IDs
print("Available products:", list(product_types.keys()))
keyboard_product_id = product_types.get("Mechanical Keyboard")
thermostat_product_id = product_types.get("Smart Thermostat")

# IMPORTANT: This is NOT a list! Don't try to iterate it as a list of products.
# It's a dict where keys=names and values=IDs.
```

"""

    # Example 2: Finding users
    if has_find_user:
        examples += """### Example 2: Finding User IDs

User lookup tools return the **user_id string directly**, not a dict:

```python
# Returns string directly (e.g., "yusuf_rossi_9620")
user_id = tau2_mcp.find_user_id_by_name_zip(
    first_name="Yusuf",
    last_name="Rossi",
    zip="33139"
)
# user_id is already a string - use it directly!
print(f"Found user: {user_id}")

# Now get user details (returns dict)
user = tau2_mcp.get_user_details(user_id=user_id)
```

"""

    # Example 3: Exchange flow
    if has_exchange and has_get_order and has_get_user and has_get_product:
        examples += """### Example 3: Complete Exchange Flow

```python
# Step 1: Get order and user
order = tau2_mcp.get_order_details(order_id="#W2378156")
user_id = tau2_mcp.find_user_id_by_name_zip(first_name="Yusuf", last_name="Rossi", zip="19122")
user = tau2_mcp.get_user_details(user_id=user_id)

# Step 2: Find item to exchange (check 'items' list in order)
old_item_id = None
old_product_id = None
for item in order.get('items', []):
    if 'keyboard' in item.get('name', '').lower():
        old_item_id = item['item_id']  # This is the variant ID (e.g., "9690244451")
        old_product_id = item['product_id']  # Product type ID (e.g., "1656367028")
        break

# Step 3: Find replacement variant
# Get product details to see all available variants
product_details = tau2_mcp.get_product_details(product_id=old_product_id)

# product_details has structure:
# {
#   "name": "Mechanical Keyboard",
#   "product_id": "1656367028",
#   "variants": {
#     "9690244451": {"item_id": "9690244451", "options": {...}, "available": true, "price": 236.51},
#     "7706410293": {"item_id": "7706410293", "options": {...}, "available": true, "price": 269.16},
#     ...
#   }
# }
variants = product_details.get('variants', {})

# Find suitable variant (use .get() for optional fields!)
new_item_id = None
for variant_id, variant_info in variants.items():
    opts = variant_info.get('options', {})
    if (variant_info.get('available', False) and
        opts.get('switch type') == 'clicky' and
        opts.get('backlight') == 'RGB'):
        new_item_id = variant_id
        break

# Step 4: Get payment method
payment_methods = user.get('payment_methods', {})
payment_method_id = None
for pm_id in payment_methods.keys():
    if 'credit_card' in pm_id:
        payment_method_id = pm_id
        break
if not payment_method_id and payment_methods:
    payment_method_id = list(payment_methods.keys())[0]

# Step 5: EXECUTE THE EXCHANGE
exchange_result = tau2_mcp.exchange_delivered_order_items(
    order_id="#W2378156",
    item_ids=[old_item_id],          # List of old variant IDs
    new_item_ids=[new_item_id],      # List of new variant IDs (same length!)
    payment_method_id=payment_method_id  # REQUIRED!
)

print(f"Exchange completed: {exchange_result}")
```

**CRITICAL - This example shows you MUST call the exchange tool!**
- Notice line: `exchange_result = tau2_mcp.exchange_delivered_order_items(...)`
- This is NOT optional - you MUST execute this call
- Without this call, the exchange doesn't happen (you just gathered information)
- The task FAILS if you don't execute the action tool

**Key Points:**
- `item_ids` and `new_item_ids` are LISTS of strings (variant IDs)
- They must be the same length (1-to-1 mapping)
- `payment_method_id` is REQUIRED - get from user's payment_methods
- Always use `.get()` for optional fields to avoid KeyErrors
- **MOST IMPORTANT**: Actually call exchange_delivered_order_items() - don't just present options!

"""

    return examples


async def run_batch_evaluation(
    task_ids: list[str],
    domain: str = "retail",
    model: str = "gpt-4o",
    output_file: Optional[str] = None,
) -> list[dict]:
    """Run SABRE on multiple tau2 tasks."""

    print(f"\n{'='*70}")
    print(f"SABRE x tau2-bench Batch Evaluation")
    print(f"{'='*70}")
    print(f"Tasks: {len(task_ids)}")
    print(f"Domain: {domain}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    results = []

    for i, task_id in enumerate(task_ids, 1):
        print(f"\n[{i}/{len(task_ids)}] Evaluating task {task_id}...")

        try:
            result = await run_sabre_on_tau2_task(task_id, domain=domain, model=model)
            results.append(result)
        except Exception as e:
            logger.error(f"Error on task {task_id}: {e}")
            results.append({
                "task_id": task_id,
                "domain": domain,
                "tau2_score": 0.0,
                "tau2_correct": False,
                "error": str(e),
            })

        # Small delay between tasks
        await asyncio.sleep(1)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.get("tau2_correct", False))
    avg_score = sum(r.get("tau2_score", 0.0) for r in results) / total if total > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"Batch Evaluation Summary")
    print(f"{'='*70}")
    print(f"Total:          {total}")
    print(f"Passed:         {passed} ({passed/total*100:.1f}%)")
    print(f"Failed:         {total - passed}")
    print(f"Average Score:  {avg_score:.3f}")
    print(f"{'='*70}\n")

    # Per-task results
    print("Per-Task Results:")
    for r in results:
        status = "✓" if r.get("tau2_correct") else "✗"
        print(f"  {status} Task {r['task_id']}: {r.get('tau2_score', 0.0):.2f}")

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SABRE on tau2-bench tasks using native MCP integration"
    )
    parser.add_argument("task_ids", nargs="+", help="Task IDs to evaluate")
    parser.add_argument("--domain", default="retail", help="Domain (default: retail)")
    parser.add_argument("--model", default="gpt-4o", help="Model (default: gpt-4o)")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Get or create event loop and apply nest_asyncio
    # This is needed because SABRE's PythonRuntime calls async MCP tools synchronously
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Apply nest_asyncio to allow nested event loops
    import nest_asyncio
    nest_asyncio.apply(loop)

    # Run evaluation
    if len(args.task_ids) == 1:
        # Single task
        result = loop.run_until_complete(
            run_sabre_on_tau2_task(
                task_id=args.task_ids[0],
                domain=args.domain,
                model=args.model,
            )
        )
    else:
        # Batch
        results = loop.run_until_complete(
            run_batch_evaluation(
                task_ids=args.task_ids,
                domain=args.domain,
                output_file=args.output,
            )
        )


if __name__ == "__main__":
    main()
