# TAU2 Benchmark Integration

This directory contains SABRE's integration with the [tau2-bench](https://github.com/sierra-research/tau-bench) benchmark for evaluating customer service agents.

## Directory Structure

```
sabre/benchmarks/tau2/
├── README.md                      # This file
├── README_API.md                  # API runner documentation (recommended!)
├── __init__.py                    # Package exports
├── sabre_tau2_runner.py           # Internal runner (direct component usage)
├── sabre_tau2_api_runner.py       # API runner (uses HTTP API) ⭐ RECOMMENDED
├── run_tau2_eval.sh               # Shell script for internal runner
├── run_tau2_eval_api.sh           # Shell script for API runner ⭐ RECOMMENDED
├── example_api_usage.py           # Example code for API runner
└── prompts/
    └── retail_agent_base.prompt   # Generic retail agent (reusable!)
```

## Quick Start

### ⭐ Recommended: API Runner (Clean & Production-Ready)

```bash
# 1. Set environment variables
export OPENAI_API_KEY=your-api-key
export TAU2_DATA_DIR=/path/to/tau2-bench/data

# 2. Start SABRE server
uv run sabre-server &

# 3. Run evaluation
./sabre/benchmarks/tau2/run_tau2_eval_api.sh task_0

# Or use Python directly
uv run python -m sabre.benchmarks.tau2.sabre_tau2_api_runner task_0 --model gpt-4o-mini

# Run batch evaluation
./sabre/benchmarks/tau2/run_tau2_eval_api.sh --batch 5 --output results.json
```

**See [README_API.md](README_API.md) for complete API runner documentation.**

### Alternative: Internal Runner (Direct Component Usage)

```bash
# Set environment variables
export OPENAI_API_KEY=your-api-key
export TAU2_DATA_DIR=/path/to/tau2-bench/data

# Run single task
./sabre/benchmarks/tau2/run_tau2_eval.sh 0 --model gpt-4o-mini --domain retail

# Or use Python directly
uv run python -m sabre.benchmarks.tau2.sabre_tau2_runner task_0 --model gpt-4o-mini --domain retail

# Run batch evaluation
uv run python -m sabre.benchmarks.tau2.sabre_tau2_runner task_0 task_1 task_2 --model gpt-4o-mini --domain retail
```

## Architecture

### Two Runner Implementations

#### 1. API Runner (Recommended) ⭐

**File**: `sabre_tau2_api_runner.py`

Uses SABRE's HTTP API (`/v1/message`, `/v1/connectors`) as a clean client.

**Benefits**:
- ✅ Clean separation (tau2 runner is just an HTTP client)
- ✅ Reuses server logic (MCP, session management, logging)
- ✅ Easier to maintain (decoupled from internal implementation)
- ✅ Better testing (can test against remote servers)
- ✅ Production-ready (same path as real SABRE users)

**Flow**:
1. Create tau2-mcp connector via `/v1/connectors`
2. Get task info via `/v1/message` (with helpers)
3. Execute task via `/v1/message` (continues conversation)
4. Get evaluation via `/v1/message` (continues conversation)
5. Remove connector via `/v1/connectors/{id}`

**See [README_API.md](README_API.md) for details.**

#### 2. Internal Runner (Legacy)

**File**: `sabre_tau2_runner.py`

Directly instantiates SABRE components (`Orchestrator`, `PythonRuntime`, `MCPClientManager`).

**Use when**: You need to test internal component changes or debug low-level issues.

**Flow**:
1. Create `MCPClientManager`
2. Connect to tau2-mcp server
3. Create `PythonRuntime` with MCP adapter
4. Create `Orchestrator`
5. Run orchestrator with task
6. Clean up connections

### Generic + Runtime Injection Pattern

This implementation uses a **generic retail agent prompt** with **runtime-injected tool documentation**:

1. **`retail_agent_base.prompt`**: Generic retail customer service expertise
   - Core skills (order management, exchanges, returns)
   - Defensive programming patterns
   - Error handling
   - **Reusable across all retail benchmarks**

2. **`sabre_tau2_runner.py`**: Runtime injection
   - Loads generic base prompt
   - Discovers available MCP tools
   - Generates tau2-specific examples dynamically via `_generate_tau2_examples()`
   - Injects tool documentation and evaluation requirements

### Benefits

✅ **Reusability**: Base prompt works with any retail benchmark  
✅ **Maintainability**: Tool-specific details injected at runtime  
✅ **Flexibility**: Easy to add new benchmarks  
✅ **Discoverability**: Examples auto-generate from available tools

## Adding New Benchmarks

To create a new retail benchmark integration:

```python
# sabre/benchmarks/your_bench/runner.py
from sabre.common.utils.prompt_loader import PromptLoader
from pathlib import Path

def _create_instructions(mcp_adapter):
    # Load the same generic retail prompt
    base_prompt = PromptLoader.load(
        Path(__file__).parent / "../tau2/prompts/retail_agent_base.prompt"
    )
    
    # Generate benchmark-specific examples
    examples = _generate_examples(mcp_adapter)
    
    # Combine and return
    return f"{base_prompt}\n\n{tool_docs}\n\n{examples}"
```

## Implementation Details

**MCP Integration**: Uses SABRE's native `MCPClientManager` and `MCPHelperAdapter` for seamless tool integration.

**Tool Discovery**: Dynamically discovers tools from tau2-mcp server and generates appropriate examples.

**Evaluation**: Gets task goals and evaluations directly from tau2's built-in evaluator via MCP.

## Requirements

- `tau2-bench` installed and data downloaded
- `TAU2_DATA_DIR` environment variable set
- OpenAI API key set via `OPENAI_API_KEY`
- tau2-mcp server available in PATH or at expected location
