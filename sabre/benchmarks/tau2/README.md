# SABRE Agent for tau2-bench

This module provides a tau2-bench compatible agent implementation using SABRE's continuation-passing execution engine.

## Overview

**Problem**: tau2-bench uses a simple LLMAgent that calls OpenAI's API directly. We want to test SABRE's more sophisticated execution model against tau2-bench tasks.

**Solution**: `SabreAgent` - A drop-in replacement that implements tau2-bench's `LocalAgent` interface while using SABRE's orchestration engine under the hood.

## Architecture

```
tau2-bench ←→ SabreAgent ←→ SABRE Orchestrator ←→ OpenAI API
                  ↑
            LocalAgent interface
            (tau2-bench compatible)
```

### Key Components

1. **SabreAgent** (`sabre_agent.py`)
   - Implements tau2-bench's `LocalAgent[SabreAgentState]` interface
   - Manages SABRE orchestrator and runtime
   - Registers tau2 tools as SABRE custom tools
   - Translates between tau2 and SABRE formats

2. **Tool Registration**
   - tau2 tools → SABRE runtime stubs
   - Stubs log calls for tau2 scoring
   - Tools callable in `<helpers>` blocks

3. **Message Flow**
   - tau2-bench sends UserMessage/ToolMessage
   - SABRE processes with orchestrator
   - Response parsed back to AssistantMessage

## Installation

### Prerequisites

```bash
# 1. Install SABRE (if not already installed)
cd /path/to/sabre
uv sync

# 2. Install tau2-bench
cd /path/to/tau2-bench
uv sync

# 3. Set environment variables
export OPENAI_API_KEY="your-api-key"
export TAU2_DATA_DIR="/tmp/tau2-bench/data"  # Optional: where to save results
```

### Required tau2-bench Modifications

**IMPORTANT**: You must modify two files in your tau2-bench installation to register the SABRE agent. Make the following changes:

#### 1. Modify `src/tau2/registry.py`

Add SABRE agent import at the top (after existing imports):

```python
from tau2.agent.base import BaseAgent
from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent

# Add these lines:
# Import SABRE agent (requires sabre package installed)
try:
    from sabre.benchmarks.tau2.sabre_agent import SabreAgent
    SABRE_AVAILABLE = True
except ImportError:
    SABRE_AVAILABLE = False
    SabreAgent = None

from tau2.data_model.tasks import Task
```

Then register the agent in the `try` block where other agents are registered:

```python
try:
    registry.register_agent(LLMAgent, "llm_agent")
    registry.register_agent(LLMGTAgent, "llm_agent_gt")
    registry.register_agent(LLMSoloAgent, "llm_agent_solo")

    # Add these lines:
    # Register SABRE agent if available
    if SABRE_AVAILABLE:
        registry.register_agent(SabreAgent, "sabre_agent")
        logger.debug("SABRE agent registered successfully")
    else:
        logger.debug("SABRE agent not available (sabre package not installed)")

    registry.register_domain(mock_domain_get_environment, "mock")
    # ... rest of registrations
```

#### 2. Modify `src/tau2/run.py`

Add SABRE agent import at the top (after existing imports):

```python
from loguru import logger

from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent

# Add these lines:
# Import SABRE agent if available
try:
    from sabre.benchmarks.tau2.sabre_agent import SabreAgent
    SABRE_AVAILABLE = True
except ImportError:
    SABRE_AVAILABLE = False
    SabreAgent = None

from tau2.data_model.simulation import (
    AgentInfo,
    # ... rest of imports
```

Then add SABRE agent constructor in the `run_task()` function (around line 451):

```python
    elif isinstance(AgentConstructor, type) and issubclass(AgentConstructor, LLMSoloAgent):
        agent = AgentConstructor(
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    # Add these lines:
    elif SABRE_AVAILABLE and issubclass(AgentConstructor, SabreAgent):
        agent = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
        )
    else:
        raise ValueError(
            f"Unknown agent type: {AgentConstructor}. Should be LLMAgent, LLMSoloAgent, or SabreAgent"
        )
```

### Verify Setup

```bash
# From SABRE directory
uv run python -c "from sabre.benchmarks.tau2.sabre_agent import SabreAgent; print('✓ SabreAgent loaded')"

# Check tau2-bench is accessible
uv run python -c "from tau2.agent.base import LocalAgent; print('✓ tau2-bench loaded')"
```

## Usage

Use tau2's CLI with the `sabre_agent` module:

```bash
# Quick test - 1 task
TAU2_DATA_DIR=/tmp/tau2-bench/data OPENAI_API_KEY=$OPENAI_API_KEY \
uv run python -m tau2.cli run \
    --agent sabre_agent \
    --domain retail \
    --num-tasks 1 \
    --agent-llm gpt-4o-mini

# Small test - 5 tasks
TAU2_DATA_DIR=/tmp/tau2-bench/data OPENAI_API_KEY=$OPENAI_API_KEY \
uv run python -m tau2.cli run \
    --agent sabre_agent \
    --domain retail \
    --num-tasks 5 \
    --agent-llm gpt-4o-mini

# Full domain evaluation
TAU2_DATA_DIR=/tmp/tau2-bench/data OPENAI_API_KEY=$OPENAI_API_KEY \
uv run python -m tau2.cli run \
    --agent sabre_agent \
    --domain retail \
    --agent-llm gpt-4o

# With reproducible seed
TAU2_DATA_DIR=/tmp/tau2-bench/data OPENAI_API_KEY=$OPENAI_API_KEY \
uv run python -m tau2.cli run \
    --agent sabre_agent \
    --domain retail \
    --num-tasks 1 \
    --agent-llm gpt-4o-mini \
    --seed 42

# Save output to log file
TAU2_DATA_DIR=/tmp/tau2-bench/data OPENAI_API_KEY=$OPENAI_API_KEY \
uv run python -m tau2.cli run \
    --agent sabre_agent \
    --domain retail \
    --num-tasks 5 \
    --agent-llm gpt-4o-mini \
    2>&1 | tee tau2_retail_test.log
```

**Available Domains:**
- `retail` - Customer service for retail/ecommerce
- `airline` - Airline booking and support
- `banking` - Banking operations
- (Run `uv run python -m tau2.cli run --help` for full list)

**Key Parameters:**
- `--agent sabre_agent` - Use SABRE agent (required)
- `--agent-llm MODEL` - Model to use (`gpt-4o`, `gpt-4o-mini`)
- `--domain DOMAIN` - Specific domain to test
- `--num-tasks N` - Number of tasks to run (default: all)
- `--seed N` - Random seed for reproducibility


## How It Works

### 1. Initialization

```python
agent = SabreAgent(tools=tau2_tools, domain_policy="...")
```

- Creates SABRE Orchestrator and PythonRuntime
- Registers tau2 tools as stubs in runtime namespace
- Tools become callable in `<helpers>` blocks

### 2. Tool Registration

tau2 tools are converted to SABRE stubs:

```python
# tau2 tool definition
Tool(name="find_user_id_by_name_zip", description="...", parameters={...})

# SABRE stub (registered in runtime)
def find_user_id_by_name_zip(**kwargs):
    logger.info(f"[STUB] find_user_id_by_name_zip({kwargs})")
    return "mock_user_id"
```

### 3. Turn Processing

For each turn:

1. tau2-bench sends message (user or tool result)
2. SabreAgent builds SABRE input with context
3. SABRE orchestrator runs with system prompt + tools
4. LLM generates response with `<helpers>` blocks
5. SABRE executes helper code (tool calls)
6. SabreAgent extracts tool calls from response
7. Formats as AssistantMessage for tau2-bench

### 4. Tool Call Extraction

```python
# SABRE response with <helpers> block
"""
I'll look up your order.
<helpers>
user_id = find_user_id_by_name_zip(first_name="Sarah", last_name="Johnson", zip="90210")
orders = get_order_details(user_id=user_id)
result(orders)
</helpers>
"""

# Parsed into tau2 format
AssistantMessage(
    content="",
    tool_calls=[
        {"id": "call_123", "function": {"name": "find_user_id_by_name_zip", "arguments": {...}}},
        {"id": "call_456", "function": {"name": "get_order_details", "arguments": {...}}}
    ]
)
```

## Output and Results

### Console Output

```
2025-11-04 10:30:45 | INFO | Running tau2-bench with SABRE agent
2025-11-04 10:30:45 | INFO | Model: gpt-4o
2025-11-04 10:30:45 | INFO | Quick mode: True
2025-11-04 10:30:45 | INFO | Max turns: 10

============================================================
Evaluating domain: retail
============================================================

Loaded 3 tasks for domain retail

Task 1/3: retail_order_tracking_001
Description: Customer wants to know where their order is...
  Reward: 0.80
  Actions: 4/5
  Turns: 6

Task 2/3: retail_return_request_002
Description: Customer wants to return an item...
  Reward: 1.00
  Actions: 3/3
  Turns: 4

============================================================
EVALUATION COMPLETE
============================================================
Total tasks: 3
Average reward: 0.87
Actions completed: 12/13 (92.3%)
Results saved to: /tmp/tau2-bench/data/simulations
```

### Results Files

Results saved to `$TAU2_DATA_DIR/simulations/`:

```
2025-11-04T10:30:45.123456_retail_sabre_gpt-4o.json
```

Contains full conversation traces, tool calls, and scoring metrics.

### Analyzing Results

```python
import json
from pathlib import Path

# Load simulation
sim_file = Path("/tmp/tau2-bench/data/simulations").glob("*_sabre_*.json")
latest = max(sim_file, key=lambda p: p.stat().st_mtime)

with open(latest) as f:
    simulation = json.load(f)

# Analyze
print(f"Task: {simulation['task_id']}")
print(f"Reward: {simulation['reward']}")
print(f"Turns: {len(simulation['turns'])}")

# Check tool usage
for turn in simulation['turns']:
    if 'tool_calls' in turn:
        print(f"Turn {turn['turn_num']}: {len(turn['tool_calls'])} tools called")
```

## Comparison: SABRE vs Default tau2 Agent

| Feature | tau2 LLMAgent | SABRE Agent |
|---------|---------------|-------------|
| **Execution Model** | Direct OpenAI API | Continuation-passing with helpers |
| **Tool Calls** | OpenAI function calling | Python code in `<helpers>` blocks |
| **Multi-step Reasoning** | Single LLM call | Recursive orchestration loop |
| **Context Management** | Message history only | Execution tree + state |
| **Custom Logic** | Limited | Full Python in helpers |
| **Debugging** | API logs only | Full execution traces |

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("sabre").setLevel(logging.DEBUG)
logging.getLogger("tau2").setLevel(logging.INFO)
```

### Common Issues

**Issue: "Tools not being called"**

Check if LLM generates `<helpers>` blocks:
```bash
# Look for helpers in logs
grep -E "(<helpers>|tool_calls)" /tmp/tau2_test.log
```

**Issue: "NameError: tool not found"**

Verify tools registered:
```python
agent = SabreAgent(...)
print("Registered tools:", list(agent.runtime.namespace.keys()))
```

**Issue: "0.0 reward score"**

- Check simulation logs for tool call extraction
- Verify system prompt includes tool documentation
- Look for SABRE execution errors

### Test Individual Components

```python
# Test tool registration
from sabre.benchmarks.tau2.sabre_agent import SabreAgent
from tau2.data import load_tasks_from_domain

task = load_tasks_from_domain("retail", num_tasks=1)[0]
agent = SabreAgent(tools=task.tools, domain_policy=task.domain_policy)

# Check tools
print("Tools registered:", [tool.name for tool in task.tools])
print("Runtime namespace:", list(agent.runtime.namespace.keys()))

# Test single turn
state = agent.get_init_state()
from tau2.data_model.message import UserMessage

msg = UserMessage(role="user", content="Test message")
response, new_state = agent.generate_next_message(msg, state)
print("Response:", response.content)
```

## Performance Tips

1. **Use Quick Mode First**: Test with `--quick` before full evaluation
2. **Start with One Domain**: `--domain retail` to isolate issues
3. **Monitor Token Usage**: SABRE uses more tokens (orchestration loops)
4. **Adjust Max Turns**: Increase `--max-turns` for complex tasks
5. **Cache Results**: Reuse simulation outputs for analysis

## Support

For issues or questions:
- Check logs in `~/.local/state/sabre/logs/`
- Review simulation outputs in `$TAU2_DATA_DIR/simulations/`
- Open an issue on GitHub with logs attached
