# External Tool Execution Plan

## Overview

Enable SABRE to delegate tool execution to the calling environment rather than executing all tools internally. This allows SABRE to integrate with external systems, benchmarks, IDEs, and applications that provide their own tool implementations.

## Problem Statement

### Current Architecture

SABRE currently executes all tools internally within its Python runtime:

1. LLM generates `<helpers>` code blocks
2. Orchestrator executes code in isolated Python namespace
3. All tools (Search, Web, Bash, llm_call) run within SABRE
4. Results injected back for continuation

**Limitation**: This tight coupling prevents SABRE from:
- Working with external benchmark frameworks (TAU-bench, SWE-bench)
- Integrating with IDEs that have their own tool implementations
- Using caller-provided APIs and databases
- Executing tools in sandboxed/privileged environments

### Desired Architecture

SABRE should support **dual execution modes**:

1. **Internal execution** (current): SABRE executes tools in its runtime
2. **External execution** (new): SABRE pauses and delegates tools to caller

This enables SABRE to act as an **orchestration engine** that coordinates tool usage without requiring all tools to run within its process.

## Design Goals

1. **Minimal disruption**: Existing internal tools continue to work unchanged
2. **Clean separation**: Clear distinction between internal vs external tools
3. **Flexible integration**: Support various caller environments (benchmarks, apps, CLIs)
4. **Stateful resumption**: Pause/resume orchestration cleanly across tool calls
5. **Type safety**: Maintain proper tool schemas and validation

## Architecture

### High-Level Flow

```
Caller
  ↓ (1) Request with input
SABRE Orchestrator
  ↓ (2) LLM generates <helpers>
Tool Classifier
  ├─ Internal tools → Execute in runtime
  └─ External tools → Pause and return to caller
       ↓ (3) Return tool calls
Caller
  ↓ (4) Execute tools in external environment
  ↓ (5) Submit results back
SABRE Orchestrator
  ↓ (6) Resume with results
  ↓ (7) Continue or complete
```

### Key Components

#### 1. Tool Registry

**Purpose**: Track which tools are internal vs external

**Responsibilities**:
- Register tool definitions with execution modes
- Classify tools as INTERNAL or EXTERNAL
- Store tool schemas for validation
- Provide lookup during orchestration

**Interface**:
- `register_internal(name, callable, description)`
- `register_external(name, schema, description)`
- `is_external(name) -> bool`
- `get_external_tools() -> list[ToolDefinition]`

#### 2. Orchestrator Enhancements

**Purpose**: Support pause/resume for external tool execution

**Responsibilities**:
- Accept tool registry during initialization
- Partition helpers into internal vs external
- Pause orchestration when external tools detected
- Return external tool calls to caller
- Resume orchestration when results provided

**New Methods**:
- `_partition_helpers(helpers) -> (internal, external)`
- `_parse_external_helpers(helpers) -> list[ToolCall]`
- `_extract_function_name(code) -> str`
- `resume_with_tool_results(conversation_id, results) -> OrchestrationResult`

**State Changes**:
- New `OrchestrationResult` fields: `status`, `pending_tool_calls`
- Support "awaiting_tool_results" status
- Maintain continuation state for resumption

#### 3. Tool Call Representation

**Purpose**: Standardize external tool call format

**Structure**:
```
ToolCall:
  - id: Unique identifier for correlation
  - name: Tool function name
  - arguments: Dict of parameter name -> value
  - raw_code: Original helper code (for debugging)
```

**Parsing Strategy**:
- Use AST to extract function calls from helper code
- Parse arguments (literals, variables, expressions)
- Generate unique IDs for correlation
- Preserve original code for debugging

#### 4. Result Injection

**Purpose**: Feed external tool results back into orchestration

**Approach**:
- Caller executes tools and collects results
- Results submitted back to SABRE with tool call IDs
- SABRE constructs `<helpers_result>` tags
- Continuation proceeds with results in context

**Result Format**:
```
ToolResult:
  - tool_call_id: Correlates to original call
  - result: Tool execution result (any JSON-serializable value)
  - error: Error message if tool failed
```

## Implementation Strategy

### Phase 1: Foundation

#### 1.1 Tool Registry System

Create infrastructure to classify and track tools:

**Components**:
- `ToolExecutionMode` enum (INTERNAL, EXTERNAL)
- `ToolDefinition` dataclass
- `ToolRegistry` class

**Functionality**:
- Register tools with execution modes
- Query tool classification
- Store tool metadata

**Integration Point**: Pass registry to orchestrator during initialization

#### 1.2 Helper Classification

Add logic to distinguish internal vs external helpers:

**Approach**:
- Parse helper code blocks with AST
- Extract function calls
- Query tool registry for each function
- Partition into internal/external lists

**Edge Cases**:
- Multiple calls in one block
- Mixed internal/external calls
- Unknown tool names

### Phase 2: Orchestrator Changes

#### 2.1 Pause/Resume Support

Modify orchestration loop to support pausing:

**When External Tools Detected**:
1. Stop iteration loop
2. Extract tool calls from external helpers
3. Create `OrchestrationResult` with status="awaiting_tool_results"
4. Return to caller with pending tool calls
5. Preserve continuation state

**State to Preserve**:
- Current conversation_id
- Last response_id
- Accumulated response text
- Execution tree
- Iteration count

#### 2.2 Result Resumption

Add method to resume with external tool results:

**Approach**:
1. Accept tool results from caller
2. Construct `<helpers_result>` tags
3. Format results as execution output
4. Continue orchestration loop
5. Return final result or pause again

**Continuation Mechanism**:
- Use existing response continuation
- Inject results as user message or continuation input
- Maintain conversation context

### Phase 3: API Changes

#### 3.1 Extended OrchestrationResult

Add fields to communicate external tool needs:

**New Fields**:
- `status`: "completed" | "awaiting_tool_results" | "error"
- `pending_tool_calls`: List of external tools to execute
- `continuation_state`: Opaque state for resumption (internal)

**Backward Compatibility**:
- Default status="completed" for existing code
- pending_tool_calls=None when no external tools

#### 3.2 Resume API

Add method for callers to submit tool results:

**Signature**:
```python
async def resume_with_tool_results(
    conversation_id: str,
    tool_results: list[ToolResult],
    event_callback: Callable | None = None
) -> OrchestrationResult
```

**Behavior**:
- Look up conversation state
- Validate tool result IDs
- Format results for continuation
- Resume orchestration
- Return next result (may pause again)

### Phase 4: Integration Support

#### 4.1 Helper Stub Generation

For external tools, create stubs in runtime namespace:

**Purpose**: Allow LLM to reference tools in helper code without import errors

**Implementation**:
- Generate stub functions for all external tools
- Stubs log calls (for debugging) but don't execute
- Return placeholder values
- Real execution happens externally

**Example**:
```python
def external_tool_stub(**kwargs):
    logger.debug(f"[EXTERNAL STUB] tool called with {kwargs}")
    return "EXTERNAL_TOOL_PLACEHOLDER"
```

#### 4.2 Tool Schema Documentation

Generate system prompt documentation for external tools:

**Approach**:
- Convert tool schemas to Python signatures
- Add to system prompt as available helpers
- Document parameters and return types
- Include usage guidance

**Format**:
```
Available external tools:
  - find_user(name: str, zip: str) -> str
    Description: Find user ID by name and zip code
    Returns: User ID as string
```

## Technical Challenges

### Challenge 1: Argument Parsing

**Problem**: Extract structured arguments from Python code AST

**Complexity**:
- Literal values (strings, numbers, bools)
- Variable references
- Expressions and computations
- Nested data structures

**Solution**:
- Parse AST nodes into JSON-serializable values
- Resolve simple variable references from runtime namespace
- Serialize complex expressions as strings
- Document limitations in external tool usage

### Challenge 2: Multi-Step Dependencies

**Problem**: What if external tool result is used in subsequent internal helpers?

**Example**:
```python
# External tool
user_id = find_user(name="Sarah")
# Internal tool - needs user_id value
search_results = Search.web(f"orders for {user_id}")
```

**Solution Options**:
1. **Block-level execution**: Execute entire helper block atomically
2. **Statement-level execution**: Parse dependencies and execute in order
3. **Require separation**: Force external and internal calls into separate turns

**Recommendation**: Start with option 3 (separate turns) for simplicity

### Challenge 3: State Serialization

**Problem**: How to preserve orchestration state between pause/resume?

**State Components**:
- Conversation ID (persisted by OpenAI)
- Response ID (persisted by OpenAI)
- Execution tree (in-memory)
- Runtime namespace (in-memory)

**Solution**:
- Store conversation_id and response_id (lightweight)
- Keep execution tree in server memory (session-based)
- For stateless scenarios, require full message history on resume

### Challenge 4: Error Handling

**Problem**: What if external tool execution fails?

**Scenarios**:
- Tool returns error
- Tool times out
- Tool not found in caller environment
- Invalid arguments

**Solution**:
- Return error in ToolResult
- Format error as execution failure in `<helpers_result>`
- Let LLM see error and retry or adjust approach
- Support partial success (some tools succeed, some fail)

## Integration Patterns

### Pattern 1: Benchmark Integration

**Use Case**: TAU-bench, SWE-bench, other agent benchmarks

**Flow**:
1. Benchmark provides tools and tasks
2. Register benchmark tools as external
3. Run SABRE orchestration
4. Return tool calls to benchmark
5. Benchmark executes and scores
6. Resume SABRE with results

**Benefits**:
- SABRE works with any benchmark
- No modification to SABRE's helpers
- Benchmark controls tool execution

### Pattern 2: Application Integration

**Use Case**: Web apps, chatbots, assistants with custom APIs

**Flow**:
1. App defines domain-specific tools (database, APIs)
2. Register tools as external in SABRE
3. SABRE orchestrates conversation
4. App executes tools in its environment
5. Results fed back to SABRE

**Benefits**:
- Keep business logic in app
- SABRE provides orchestration only
- Secure tool execution in app context

### Pattern 3: IDE Integration

**Use Case**: Code editors with language servers and file access

**Flow**:
1. IDE provides tools (read_file, write_file, run_tests)
2. Register as external tools
3. SABRE generates code and suggests actions
4. IDE executes in user's workspace
5. Results shown to user and SABRE

**Benefits**:
- Respect file permissions
- Use IDE's language servers
- Proper undo/redo support

## Testing Strategy

### Unit Tests

1. **Tool Registry**
   - Register internal/external tools
   - Query execution modes
   - Handle unknown tools

2. **Helper Partitioning**
   - Classify helpers correctly
   - Handle mixed helpers
   - Parse tool calls from code

3. **AST Parsing**
   - Extract function names
   - Parse literal arguments
   - Handle variable references

### Integration Tests

1. **Pause/Resume Flow**
   - Pause on external tools
   - Resume with results
   - Multi-turn conversations

2. **Tool Call Format**
   - Validate JSON structure
   - Verify ID generation
   - Check argument serialization

3. **Error Handling**
   - Tool execution failures
   - Invalid results
   - Missing tools

### End-to-End Tests

1. **Mock Benchmark**
   - Simulate benchmark framework
   - Provide external tools
   - Verify scoring

2. **Multi-Turn Scenarios**
   - Complex task decomposition
   - Mixed internal/external usage
   - Error recovery

## Migration Path

### For Existing SABRE Users

**No Breaking Changes**:
- Default tool registry is empty
- All tools treated as internal by default
- Current workflows unchanged

**Opt-In External Tools**:
- Pass tool_registry to Orchestrator
- Register specific tools as external
- Handle pause/resume in caller

### For New Integrations

**Recommended Pattern**:
1. Create ToolRegistry
2. Register SABRE's helpers as internal
3. Register domain tools as external
4. Initialize Orchestrator with registry
5. Handle OrchestrationResult status in caller

## Success Metrics

### Functional Requirements

- [ ] Tool registry correctly classifies tools
- [ ] Orchestrator pauses on external tools
- [ ] Tool calls extracted with valid structure
- [ ] Resumption works with tool results
- [ ] Multi-turn conversations complete successfully

### Performance Requirements

- [ ] Minimal overhead for internal-only workflows
- [ ] Efficient AST parsing
- [ ] Reasonable token usage with external tools

### Integration Requirements

- [ ] Works with benchmark frameworks
- [ ] Supports application integration
- [ ] Maintains SABRE's orchestration quality

## Future Enhancements

### Advanced Features

1. **Parallel Tool Execution**
   - Identify independent tool calls
   - Return for parallel execution
   - Accept results in any order

2. **Tool Call Validation**
   - Validate arguments against schemas
   - Type checking before execution
   - Better error messages

3. **Streaming Tool Results**
   - Stream results as they become available
   - Progressive continuation
   - Real-time updates

### Developer Experience

1. **Better Debugging**
   - Visualize tool call graph
   - Trace execution flow
   - Debug AST parsing

2. **Tool Development Kit**
   - Helper utilities for tool registration
   - Schema generation from Python functions
   - Testing utilities

## Conclusion

External tool execution transforms SABRE from a self-contained agent into a flexible orchestration engine. By cleanly separating tool classification from execution, SABRE can coordinate complex workflows while delegating actual tool execution to the most appropriate environment.

This architecture enables:
- **Benchmark participation** without sacrificing SABRE's execution model
- **Application integration** while maintaining security boundaries
- **Hybrid execution** combining SABRE's internal helpers with external capabilities

The design maintains backward compatibility while opening new integration possibilities, positioning SABRE as a universal agent orchestration layer.
