# SABRE x tau2-bench Dialogue Mode Integration

## Overview

SABRE now supports **two modes** for tau2-bench evaluation:

1. **Simple Mode** (`sabre_tau2_runner.py`) - DB-only evaluation, autonomous agent
2. **Dialogue Mode** (`sabre_tau2_dialogue_runner.py`) - Full conversation with user simulator

## Simple Mode vs. Dialogue Mode

### Simple Mode (Original)

**File**: `sabre_tau2_runner.py`

**How it works**:
- Agent gets task goal and works autonomously
- No user simulator interaction
- Evaluation: DB-state only
- Single orchestrator run

**When to use**:
- Quick testing
- DB-only tasks
- Autonomous agent evaluation

**Run**:
```bash
python -m sabre.benchmarks.tau2.sabre_tau2_runner 0
```

### Dialogue Mode (NEW!)

**File**: `sabre_tau2_dialogue_runner.py`

**How it works**:
- Real multi-turn conversation with user simulator
- Agent responds to user messages naturally
- Uses `tau2.get_next_message()` and `tau2.send_agent_message()`
- Evaluation: DB + COMMUNICATE + ACTION

**When to use**:
- Complete tau2-bench evaluation
- Testing conversation flow
- COMMUNICATE policy validation
- Real-world agent behavior

**Run**:
```bash
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0
```

## Quick Start

### Prerequisites

```bash
# 1. Set tau2 data directory
export TAU2_DATA_DIR=/path/to/tau2-bench/data

# 2. Ensure tau2-mcp is installed
pip install -e /path/to/tau2-bench-mcp

# 3. Set tau2-mcp path (if not in standard location)
export TAU2_MCP_PATH=/path/to/tau2-bench/.venv/bin/tau2-mcp
```

### Run Simple Mode

```bash
cd /path/to/sabre
python -m sabre.benchmarks.tau2.sabre_tau2_runner 0 --domain retail
```

Output:
```
======================================================================
SABRE x tau2-bench Evaluation
======================================================================
Task ID: 0
Domain: retail
Model: gpt-4o
======================================================================

[1/6] Configuring tau2-mcp server...
[2/6] Initializing SABRE MCP client manager...
[3/6] Connecting to tau2-mcp server...
[4/6] Getting task information from tau2...
[5/6] Running SABRE agent...
[6/6] Getting evaluation from tau2...

======================================================================
Evaluation Results
======================================================================
tau2 Score:      0.75
tau2 Correct:    True
tau2 Details:    DB State: PASS
SABRE Success:   True
======================================================================
```

### Run Dialogue Mode

```bash
cd /path/to/sabre
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0 --domain retail
```

Output:
```
======================================================================
SABRE x tau2-bench DIALOGUE MODE Evaluation
======================================================================
Task ID: 0
Domain: retail
Model: gpt-4o
Mode: DIALOGUE (with user simulator)
======================================================================

[1/7] Configuring tau2-mcp server (dialogue mode)...
[2/7] Initializing SABRE MCP client manager...
[3/7] Connecting to tau2-mcp (dialogue mode)...
[4/7] Getting task information...
[5/7] Setting up SABRE components...
[6/7] Starting dialogue conversation...

   --- Turn 1 ---
   USER: I need to exchange my keyboard from order #W2378156...
   SABRE processing...
   AGENT: I can help with that! Let me look up your order...

   --- Turn 2 ---
   USER: The order number is #W2378156...
   SABRE processing...
   AGENT: I found your order. I can see you have a mechanical keyboard...

   ... (more turns)

   ✓ Dialogue completed: 8 turns

[7/7] Getting evaluation (with COMMUNICATE scores)...

======================================================================
Dialogue Mode Evaluation Results
======================================================================
tau2 Score:          0.95
tau2 Correct:        True
DB Reward:           1.0
COMMUNICATE Reward:  0.9
ACTION Reward:       1.0
Details:             DB State: PASS; COMMUNICATE: PASS
Conversation Turns:  8
======================================================================
```

## Key Differences

| Feature | Simple Mode | Dialogue Mode |
|---------|-------------|---------------|
| User Simulator | ❌ No | ✅ Yes |
| Multi-turn Conversation | ❌ No | ✅ Yes |
| DB Evaluation | ✅ Yes | ✅ Yes |
| COMMUNICATE Evaluation | ❌ No | ✅ **Yes** |
| ACTION Evaluation | ⚠️ Partial | ✅ **Full** |
| Conversation History | ❌ No | ✅ Yes |
| Natural Interaction | ❌ No | ✅ Yes |

## Implementation Details

### Dialogue Mode Flow

1. **Connect with --dialogue-mode flag**
   ```python
   args=["--eval-mode", "--domain", domain, "--task-id", task_id, "--dialogue-mode"]
   ```

2. **Conversation Loop**
   ```python
   while turn < max_turns:
       # Get next message from user simulator
       msg = await client.call_tool("tau2.get_next_message", {})

       # SABRE processes user message
       result = await orchestrator.run(...)

       # Send agent response back
       await client.call_tool("tau2.send_agent_message", {
           "content": agent_response,
           "tool_calls": []
       })

       # Check if conversation complete
       if conversation_complete:
           break
   ```

3. **Get Full Evaluation**
   ```python
   eval_result = await client.call_tool("tau2.get_result", {})
   # Returns DB + COMMUNICATE + ACTION scores
   ```

### Code Changes Needed

**None required for existing code!** The dialogue mode runner is a separate file.

- `sabre_tau2_runner.py` - Keep as-is (simple mode)
- `sabre_tau2_dialogue_runner.py` - NEW file (dialogue mode)

Both can coexist and be used based on your needs.

## Command Line Options

### Simple Mode

```bash
python -m sabre.benchmarks.tau2.sabre_tau2_runner \
    <task_ids...> \
    [--domain DOMAIN] \
    [--model MODEL] \
    [--output FILE] \
    [--verbose]

# Examples
python -m sabre.benchmarks.tau2.sabre_tau2_runner 0
python -m sabre.benchmarks.tau2.sabre_tau2_runner 0 1 2 --output results.json
python -m sabre.benchmarks.tau2.sabre_tau2_runner 0 --domain airline --verbose
```

### Dialogue Mode

```bash
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner \
    <task_id> \
    [--domain DOMAIN] \
    [--model MODEL] \
    [--max-turns N] \
    [--output FILE] \
    [--verbose]

# Examples
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0 --max-turns 30
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0 --output result.json
```

## Troubleshooting

### tau2-mcp not found

```
FileNotFoundError: tau2-mcp not found
```

**Solution**:
```bash
export TAU2_MCP_PATH=/path/to/tau2-bench/.venv/bin/tau2-mcp
```

### Dialogue mode not enabled

```
RuntimeError: Dialogue mode not enabled in tau2-mcp server!
```

**Solution**: Make sure the `--dialogue-mode` flag is in the args:
```python
args=["--eval-mode", "--domain", domain, "--task-id", task_id, "--dialogue-mode"]
```

### COMMUNICATE reward is None

This is expected in simple mode. Use dialogue mode for COMMUNICATE evaluation:
```bash
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0
```

## Testing

### Test Simple Mode

```bash
cd /path/to/sabre
export TAU2_DATA_DIR=/path/to/tau2-bench/data
python -m sabre.benchmarks.tau2.sabre_tau2_runner 0
```

Expected: Quick run, DB-only evaluation

### Test Dialogue Mode

```bash
cd /path/to/sabre
export TAU2_DATA_DIR=/path/to/tau2-bench/data
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0
```

Expected: Multi-turn conversation, full evaluation

## Advanced Usage

### Custom Max Turns

```bash
# Allow longer conversations
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0 --max-turns 50
```

### Different Model

```bash
# Use GPT-4o-mini for faster/cheaper evaluation
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0 --model gpt-4o-mini
```

### Save Conversation History

```bash
# Output includes full conversation history
python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner 0 --output result.json

# View conversation
jq '.conversation_history' result.json
```

## Future Enhancements

### Planned Improvements

1. **Tool Call Extraction**: Extract tool calls from SABRE execution tree
2. **Conversation Metrics**: Track turn efficiency, tool usage patterns
3. **Batch Dialogue Mode**: Run multiple tasks in dialogue mode
4. **Resume Capability**: Resume interrupted conversations

### Contributing

To improve the dialogue mode integration:

1. Extract tool calls from `ExecutionTree` after each turn
2. Add conversation metrics (turns to completion, tool calls per turn)
3. Improve conversation history formatting
4. Add support for user-initiated tool calls

## References

- tau2-mcp dialogue mode: `/path/to/tau2-bench-mcp/DIALOGUE_MODE.md`
- tau2-mcp testing: `/path/to/tau2-bench-mcp/TESTING.md`
- SABRE MCP integration: `/path/to/sabre/docs/mcp.md`
- tau2-bench: https://github.com/ServiceNow/tau2-bench

## Summary

✅ **Simple Mode** - Use for quick testing, DB-only evaluation
✅ **Dialogue Mode** - Use for complete evaluation with COMMUNICATE scores

Both modes work with the same SABRE codebase - just use the appropriate runner!
