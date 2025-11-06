# External Tools Implementation

**Status**: ✅ Implemented (2025-01-05)
**Related**: `plans/PERSONA_PLAN.md`, `tests/test_external_tools.py`

## Overview

SABRE's external tools system enables **pause/resume orchestration** for domain-specific tool calls that need to be executed by the caller (e.g., benchmarks, external systems) rather than by SABRE itself.

This is critical for integrations like tau2-bench, where:
- Benchmark tools must be executed by tau2-bench's environment
- Tool calls need to be tracked and scored
- Results must come from real implementations, not mocks

## Architecture

### Core Components

#### 1. ToolRegistry (`sabre/server/tool_registry.py`)

Central registry that tracks which tools are **internal** (executed by SABRE) vs **external** (executed by caller).

```python
from sabre.server.tool_registry import ToolRegistry, ToolMode

# Create registry
registry = ToolRegistry()

# Register external tool (for benchmarks)
registry.register_external(
    name="find_user_id",
    description="Find user ID by name and zip",
    parameters={
        "first_name": {"type": "str", "required": True},
        "last_name": {"type": "str", "required": True},
        "zip": {"type": "str", "required": True}
    }
)

# Register internal tool (executed by SABRE)
def custom_search(query: str) -> dict:
    return {"results": [...]}

registry.register_internal(
    name="custom_search",
    callable=custom_search,
    description="Custom search implementation"
)

# Check tool mode
mode = registry.get_mode("find_user_id")  # Returns ToolMode.EXTERNAL
```

**Key Methods**:
- `register_external(name, description, parameters)` - Register tool that caller will execute
- `register_internal(name, callable, description)` - Register tool SABRE will execute
- `get_mode(name)` - Get execution mode for a tool
- `is_external(name)` - Check if tool is external
- `get_callable(name)` - Get callable for internal tools

#### 2. Enhanced Orchestrator (`sabre/server/orchestrator.py`)

The orchestrator now supports **pause/resume** workflow for external tools.

**Key Changes**:
```python
class Orchestrator:
    def __init__(
        self,
        executor: ResponseExecutor,
        runtime: PythonRuntime,
        event_callback: Optional[Callable] = None,
        tool_registry: Optional[ToolRegistry] = None  # NEW!
    ):
        self.tool_registry = tool_registry or ToolRegistry()
```

**Partition Logic**:
```python
# In orchestrator.run() - after parsing helpers
helpers = parser.extract_helpers()

if self.tool_registry:
    # Partition into internal vs external
    internal_helpers = []
    external_helpers = []

    for helper_code in helpers:
        tool_name = extract_tool_name(helper_code)
        if self.tool_registry.is_external(tool_name):
            external_helpers.append(helper_code)
        else:
            internal_helpers.append(helper_code)

    # If external tools detected, PAUSE
    if external_helpers:
        return OrchestrationResult(
            status=OrchestrationStatus.AWAITING_TOOL_RESULTS,
            pending_tool_calls=external_helpers,
            conversation_id=conversation_id
        )

    # Execute internal tools normally
    helpers = internal_helpers
```

**Resume Method**:
```python
async def continue_with_tool_results(
    self,
    conversation_id: str,
    tool_results: List[ToolResult]
) -> OrchestrationResult:
    """Resume orchestration after external tools executed"""

    # Format results as <helpers_result> tags
    result_text = format_tool_results(tool_results)

    # Continue orchestration with results
    return await self.run(
        conversation_id=conversation_id,
        input_text=result_text,
        instructions=self.last_instructions
    )
```

#### 3. OrchestrationResult Enhancement (`sabre/common/models/execution_tree.py`)

Added fields to support pause/resume:

```python
@dataclass
class OrchestrationResult:
    status: OrchestrationStatus  # NEW: completed, awaiting_tool_results, error
    content: str
    conversation_id: str
    tree: Optional[ExecutionTree] = None
    pending_tool_calls: Optional[List[str]] = None  # NEW: For external tools
    metadata: Optional[Dict] = None
```

**Status Enum**:
```python
class OrchestrationStatus(str, Enum):
    COMPLETED = "completed"
    AWAITING_TOOL_RESULTS = "awaiting_tool_results"
    ERROR = "error"
```

## Usage Patterns

### Pattern 1: Benchmark Integration (External Tools)

**Use Case**: tau2-bench needs to execute and score retail domain tools

```python
from sabre.server.tool_registry import ToolRegistry
from sabre.server.orchestrator import Orchestrator

# 1. Create registry with benchmark tools
registry = ToolRegistry()

# Register all tau2-bench tools as external
registry.register_external(
    name="find_user_id_by_name_zip",
    description="Find user ID by first name, last name, and zip",
    parameters={
        "first_name": {"type": "str", "required": True},
        "last_name": {"type": "str", "required": True},
        "zip": {"type": "str", "required": True}
    }
)

registry.register_external(
    name="get_order_details",
    description="Get order details for a user",
    parameters={
        "user_id": {"type": "str", "required": True}
    }
)

# 2. Create orchestrator with registry
orchestrator = Orchestrator(
    executor=executor,
    runtime=runtime,
    tool_registry=registry
)

# 3. Run orchestration
result = await orchestrator.run(
    conversation_id="conv_123",
    input_text="Help me find my order. I'm John Doe from 90210",
    instructions=system_prompt
)

# 4. Check if paused for external tools
if result.status == OrchestrationStatus.AWAITING_TOOL_RESULTS:
    # Extract tool calls
    tool_calls = result.pending_tool_calls
    # ['find_user_id_by_name_zip("John", "Doe", "90210")']

    # 5. Execute in benchmark environment
    tool_results = []
    for call in tool_calls:
        # Parse and execute
        result = tau2_env.execute_tool(call)
        tool_results.append(ToolResult(
            tool_name="find_user_id_by_name_zip",
            result=result
        ))

    # 6. Resume orchestration with results
    result = await orchestrator.continue_with_tool_results(
        conversation_id="conv_123",
        tool_results=tool_results
    )

# 7. Final result
if result.status == OrchestrationStatus.COMPLETED:
    print(result.content)
```

### Pattern 2: Mixed Internal/External Tools

**Use Case**: Some tools executed by SABRE, others by caller

```python
registry = ToolRegistry()

# Internal tools (SABRE executes)
def custom_analysis(data: str) -> dict:
    return {"insights": [...]}

registry.register_internal(
    name="analyze_data",
    callable=custom_analysis,
    description="Analyze data using custom logic"
)

# External tools (caller executes)
registry.register_external(
    name="query_database",
    description="Execute SQL query",
    parameters={"query": {"type": "str", "required": True}}
)

# LLM generates both types
# <helpers>
# data = query_database("SELECT * FROM users")
# insights = analyze_data(data)
# </helpers>

# Orchestrator will:
# 1. Detect query_database is external → PAUSE
# 2. Return pending_tool_calls=["query_database(...)"]
# 3. Caller executes query_database
# 4. Caller resumes with results
# 5. Orchestrator executes analyze_data internally
# 6. Returns final result
```

### Pattern 3: Complete Workflow Example

See `tests/test_external_tools.py` for complete examples:

```python
def test_external_tool_workflow():
    """Test complete pause/resume workflow with external tools"""

    # Setup
    registry = ToolRegistry()
    registry.register_external(
        name="find_user",
        description="Find user by name",
        parameters={"name": {"type": "str", "required": True}}
    )

    orchestrator = Orchestrator(
        executor=mock_executor,
        runtime=runtime,
        tool_registry=registry
    )

    # Mock LLM to return external tool call
    mock_executor.set_response('<helpers>find_user("John")</helpers>')

    # Step 1: Run orchestration
    result = await orchestrator.run(
        conversation_id="test",
        input_text="Find John",
        instructions="You are a helpful assistant"
    )

    # Verify paused
    assert result.status == OrchestrationStatus.AWAITING_TOOL_RESULTS
    assert "find_user" in result.pending_tool_calls[0]

    # Step 2: Execute external tool
    user_data = {"user_id": "123", "name": "John"}

    # Step 3: Resume with results
    result = await orchestrator.continue_with_tool_results(
        conversation_id="test",
        tool_results=[
            ToolResult(tool_name="find_user", result=user_data)
        ]
    )

    # Verify completed
    assert result.status == OrchestrationStatus.COMPLETED
    assert "123" in result.content  # LLM used the result
```

## Tool Call Format

External tools are returned as Python code strings:

```python
# Single tool call
pending_tool_calls = [
    'find_user_id_by_name_zip("John", "Doe", "90210")'
]

# Multiple tool calls
pending_tool_calls = [
    'find_user_id_by_name_zip("John", "Doe", "90210")',
    'get_order_details(user_id="user_123")',
    'check_inventory(product_id="prod_456")'
]

# With variable assignments (LLM style)
pending_tool_calls = [
    'user_id = find_user_id_by_name_zip("John", "Doe", "90210")'
]
```

**Parsing**: Caller is responsible for:
1. Extracting function name
2. Parsing arguments
3. Executing in their environment
4. Returning results

## Result Format

Tool results are returned as `ToolResult` objects:

```python
from sabre.common.models.execution_tree import ToolResult

results = [
    ToolResult(
        tool_name="find_user_id_by_name_zip",
        result="user_123",  # Can be any serializable type
        error=None  # Optional error message
    ),
    ToolResult(
        tool_name="get_order_details",
        result={
            "order_id": "ord_456",
            "status": "shipped",
            "items": [...]
        }
    )
]
```

Results are formatted as `<helpers_result>` tags for LLM:

```xml
<helpers_result>
find_user_id_by_name_zip: "user_123"
get_order_details: {"order_id": "ord_456", "status": "shipped", ...}
</helpers_result>
```

## Integration with tau2-bench

Complete integration example in `sabre/benchmarks/tau2/sabre_agent.py`:

```python
class SabreAgent(Agent):
    def __init__(self):
        # Create tool registry from tau2-bench tool schemas
        self.tool_registry = self._create_tool_registry()

        # Create orchestrator with registry
        self.orchestrator = Orchestrator(
            executor=self.executor,
            runtime=self.runtime,
            tool_registry=self.tool_registry
        )

    def _create_tool_registry(self) -> ToolRegistry:
        """Register all tau2-bench tools as external"""
        registry = ToolRegistry()

        # Get tools from tau2-bench environment
        for tool_name, tool_schema in self.env.get_tools().items():
            registry.register_external(
                name=tool_name,
                description=tool_schema.description,
                parameters=tool_schema.parameters
            )

        return registry

    async def get_asst_message(
        self,
        message: ValidAgentInputMessage,
        state: SabreAgentState
    ) -> AgentResponseMessage:
        """Main agent loop with pause/resume"""

        # Build input from conversation history
        input_text = self._build_input(message, state)

        # Run orchestration
        result = await self.orchestrator.run(
            conversation_id=state.conversation_id,
            input_text=input_text,
            instructions=self.system_prompt
        )

        # Check if paused for external tools
        if result.status == OrchestrationStatus.AWAITING_TOOL_RESULTS:
            # Convert SABRE tool calls to tau2-bench format
            tool_calls = self._convert_to_tau2_format(
                result.pending_tool_calls
            )

            # Return to tau2-bench for execution
            return AssistantMessage(tool_calls=tool_calls)

        # Return final response
        return AssistantMessage(content=result.content)

    async def handle_tool_results(
        self,
        results: MultiToolMessage,
        state: SabreAgentState
    ) -> AgentResponseMessage:
        """Resume after tau2-bench executes tools"""

        # Convert tau2-bench results to SABRE format
        tool_results = self._convert_results(results)

        # Resume orchestration
        result = await self.orchestrator.continue_with_tool_results(
            conversation_id=state.conversation_id,
            tool_results=tool_results
        )

        # Check if paused again (chained tool calls)
        if result.status == OrchestrationStatus.AWAITING_TOOL_RESULTS:
            tool_calls = self._convert_to_tau2_format(
                result.pending_tool_calls
            )
            return AssistantMessage(tool_calls=tool_calls)

        # Return final response
        return AssistantMessage(content=result.content)
```

## Design Decisions

### 1. Why Pause/Resume Instead of Callbacks?

**Chosen**: Pause orchestration, return control to caller

**Alternative**: Callback system where caller provides functions

**Rationale**:
- ✅ Caller has full control over execution
- ✅ Enables benchmarking and scoring
- ✅ Clean separation of concerns
- ✅ No need to pass domain logic into SABRE
- ✅ Supports synchronous and asynchronous callers

### 2. Why ToolRegistry Instead of Runtime Namespace?

**Chosen**: Separate ToolRegistry for tracking tool modes

**Alternative**: Add external tools to runtime namespace as stubs

**Rationale**:
- ✅ Clean separation: registry tracks modes, runtime executes
- ✅ No stub pollution in runtime namespace
- ✅ Easier to query "what tools are external?"
- ✅ Supports future enhancements (tool schemas, validation)
- ✅ Registry can be shared across orchestrator instances

### 3. Why Return Python Code Strings?

**Chosen**: Return tool calls as Python code strings

**Alternative**: Parse into structured format (function name + args dict)

**Rationale**:
- ✅ Preserves exactly what LLM generated
- ✅ Caller can parse in their preferred way
- ✅ Simpler implementation (no parsing ambiguity)
- ✅ Supports complex expressions if needed
- ❌ Caller must implement parser (but they know their tools best)

### 4. Why OrchestrationStatus Enum?

**Chosen**: Status enum on OrchestrationResult

**Alternative**: Different result types (CompletedResult, PausedResult, ErrorResult)

**Rationale**:
- ✅ Simpler type system (one result type)
- ✅ Easy to check: `if result.status == AWAITING_TOOL_RESULTS`
- ✅ Optional fields (pending_tool_calls only when paused)
- ✅ Future-proof (can add new statuses)

## Testing

Comprehensive test suite in `tests/test_external_tools.py`:

### Test Coverage

```python
# Basic functionality
test_tool_registry_external_registration()
test_tool_registry_internal_registration()
test_tool_registry_mode_detection()

# Orchestrator integration
test_orchestrator_with_external_tools()
test_orchestrator_partitions_tools_correctly()
test_orchestrator_continues_with_results()

# Edge cases
test_multiple_external_tools_in_one_turn()
test_mixed_internal_external_tools()
test_chained_external_tool_calls()
test_external_tool_with_error()

# Integration
test_complete_external_tool_workflow()
test_orchestrator_resumes_conversation_state()
```

### Running Tests

```bash
# Run all external tools tests
uv run pytest tests/test_external_tools.py -v

# Run specific test
uv run pytest tests/test_external_tools.py::test_complete_external_tool_workflow -v

# Run with coverage
uv run pytest tests/test_external_tools.py --cov=sabre.server.tool_registry --cov=sabre.server.orchestrator
```

### Test Results (as of 2025-01-05)

```
tests/test_external_tools.py::test_tool_registry_external_registration PASSED
tests/test_external_tools.py::test_tool_registry_internal_registration PASSED
tests/test_external_tools.py::test_orchestrator_with_external_tools PASSED
tests/test_external_tools.py::test_orchestrator_partitions_tools PASSED
tests/test_external_tools.py::test_complete_workflow PASSED
... (13 tests total, all passing)
```

## Future Enhancements

### 1. Tool Schema Validation

Validate tool calls against registered schemas:

```python
registry.register_external(
    name="find_user",
    parameters={
        "name": {"type": "str", "required": True},
        "age": {"type": "int", "required": False, "min": 0, "max": 150}
    },
    validate=True  # Enable validation
)

# Orchestrator would validate before pausing
result = orchestrator.run(...)  # Validates tool calls match schema
```

### 2. Tool Result Validation

Validate results match expected return types:

```python
registry.register_external(
    name="find_user",
    returns={"type": "object", "properties": {"user_id": "str", "name": "str"}},
    validate_results=True
)

# Would validate results before resuming
orchestrator.continue_with_tool_results(...)  # Validates result format
```

### 3. Streaming External Tool Calls

Support streaming tool calls as LLM generates them:

```python
async for event in orchestrator.run_streaming(...):
    if event.type == "external_tool_detected":
        # Execute immediately without waiting for full response
        result = execute_tool(event.tool_call)
        await orchestrator.provide_tool_result(event.tool_id, result)
```

### 4. Tool Call Batching

Batch multiple external tool calls for parallel execution:

```python
result = orchestrator.run(...)
if result.status == AWAITING_TOOL_RESULTS:
    # Execute all tools in parallel
    results = await asyncio.gather(*[
        execute_tool(call) for call in result.pending_tool_calls
    ])
    orchestrator.continue_with_tool_results(results)
```

### 5. Tool Execution Timeout

Add timeout support for external tools:

```python
registry.register_external(
    name="slow_api_call",
    timeout=30.0  # seconds
)

# Orchestrator tracks timeout, returns error if exceeded
```

## Troubleshooting

### Issue: External tools not detected

**Symptom**: Orchestrator tries to execute external tool, gets NameError

**Solution**: Ensure tool is registered in ToolRegistry before orchestration:
```python
# Check if registered
assert registry.is_external("tool_name")

# Check mode
mode = registry.get_mode("tool_name")
assert mode == ToolMode.EXTERNAL
```

### Issue: Results not formatted correctly

**Symptom**: LLM doesn't understand tool results in continuation

**Solution**: Ensure ToolResult objects are properly formatted:
```python
# Correct format
ToolResult(
    tool_name="exact_function_name",  # Must match tool call
    result={"key": "value"},  # Serializable data
    error=None  # No error
)

# Incorrect - mismatched name
ToolResult(
    tool_name="find_user",  # Tool was called as find_user_by_name
    result=...
)
```

### Issue: Orchestration doesn't pause

**Symptom**: Orchestrator executes external tool instead of pausing

**Solution**: Check tool_registry is passed to orchestrator:
```python
# Wrong - no registry
orchestrator = Orchestrator(executor, runtime)

# Correct - with registry
orchestrator = Orchestrator(executor, runtime, tool_registry=registry)
```

### Issue: Multiple pause/resume cycles

**Symptom**: Orchestration pauses multiple times for same tools

**Solution**: This is expected for chained tool calls:
```python
# Turn 1: LLM calls find_user → PAUSE
result1 = orchestrator.run(...)

# Turn 2: Resume with user data, LLM calls get_orders → PAUSE
result2 = orchestrator.continue_with_tool_results(...)

# Turn 3: Resume with order data, LLM generates final response → COMPLETE
result3 = orchestrator.continue_with_tool_results(...)
```

## API Reference

### ToolRegistry

```python
class ToolRegistry:
    """Registry for tracking internal vs external tools"""

    def register_external(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict] = None
    ) -> None:
        """Register tool that caller will execute"""

    def register_internal(
        self,
        name: str,
        callable: Callable,
        description: str
    ) -> None:
        """Register tool that SABRE will execute"""

    def get_mode(self, name: str) -> Optional[ToolMode]:
        """Get execution mode for a tool"""

    def is_external(self, name: str) -> bool:
        """Check if tool is external"""

    def is_internal(self, name: str) -> bool:
        """Check if tool is internal"""

    def get_callable(self, name: str) -> Optional[Callable]:
        """Get callable for internal tool"""
```

### Orchestrator

```python
class Orchestrator:
    def __init__(
        self,
        executor: ResponseExecutor,
        runtime: PythonRuntime,
        event_callback: Optional[Callable] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        """Create orchestrator with optional tool registry"""

    async def run(
        self,
        conversation_id: str,
        input_text: str,
        instructions: str,
        **kwargs
    ) -> OrchestrationResult:
        """
        Run orchestration. May pause for external tools.

        Returns:
            OrchestrationResult with status:
            - COMPLETED: Orchestration finished
            - AWAITING_TOOL_RESULTS: Paused for external tools
            - ERROR: Error occurred
        """

    async def continue_with_tool_results(
        self,
        conversation_id: str,
        tool_results: List[ToolResult]
    ) -> OrchestrationResult:
        """Resume orchestration after external tools executed"""
```

### OrchestrationResult

```python
@dataclass
class OrchestrationResult:
    status: OrchestrationStatus  # completed, awaiting_tool_results, error
    content: str  # Final response or partial response
    conversation_id: str
    tree: Optional[ExecutionTree] = None
    pending_tool_calls: Optional[List[str]] = None  # When status=AWAITING_TOOL_RESULTS
    metadata: Optional[Dict] = None
```

### ToolResult

```python
@dataclass
class ToolResult:
    tool_name: str  # Name of tool that was called
    result: Any  # Result from tool execution
    error: Optional[str] = None  # Error message if tool failed
```

## Related Documentation

- **Persona System**: `plans/PERSONA_PLAN.md` - How personas use custom tools
- **tau2-bench Integration**: `sabre/benchmarks/tau2/README.md` - Complete integration guide
- **Testing**: `tests/test_external_tools.py` - Comprehensive test examples
- **Architecture**: `plans/PERSONA_PLAN.md#custom-tools` - Design rationale and future plans

## Changelog

### 2025-01-05: Initial Implementation
- Created ToolRegistry for tool mode tracking
- Enhanced Orchestrator with pause/resume workflow
- Added OrchestrationStatus and pending_tool_calls
- Implemented tool partitioning (internal vs external)
- Created comprehensive test suite (13 tests)
- Integrated with tau2-bench SABRE agent
- Updated PERSONA_PLAN with Phase 0 completion

### Future Versions
- Schema validation for tool calls
- Streaming external tool execution
- Tool call batching and parallel execution
- Timeout support for external tools
- Enhanced error handling and recovery
