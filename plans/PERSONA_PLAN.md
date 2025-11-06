# SABRE - Example-Driven Persona System

> **✅ MAJOR UPDATE (2025-01-05)**: External tools implementation complete! Phase 2 (Custom Tools) is now 70% done. See [Implementation Status](#implementation-status) below.

## Implementation Status

### ✅ Completed (Phase 0 & Partial Phase 2)

- **ToolRegistry** - Internal/external tool classification ✅
- **Orchestrator Enhancements** - Tool partitioning, pause/resume ✅
- **External Tool Pattern** - Full pause/resume orchestration ✅
- **SABRE Agent Update** - Demonstrates tool registry usage ✅
- **Comprehensive Tests** - 13 tests passing ✅
- **Documentation** - `docs/EXTERNAL_TOOLS_IMPLEMENTATION.md` ✅

### 🚧 In Progress

- **PersonaLoader** - Load persona configs from YAML
- **PromptBuilder** - Build prompts with persona variables
- **PythonRuntime.register_tools()** - Formal API for tool registration
- **ToolImplementationLoader** - Load Python implementations for internal tools

### ⏸️ Deferred

- **MCP Integration** - Not needed for tau2-bench, add later

**Key Insight**: The hard part (tool execution architecture) is done. Remaining work is configuration, prompting, and user experience.

## Problem Statement

**Current State:**
- SABRE uses a generic "helpful assistant" identity for all tasks
- Every prompt includes full documentation for ALL ~15 helpers
- No domain-specific guidance or workflow examples
- Result: Wasted tokens, no specialization, generic responses

**What We Want:**
- Domain-focused identities (web researcher, coder, data analyst)
- Example workflows showing how each persona approaches tasks
- Relevant helpers featured prominently, others accessible via `helpers()`
- Teach by example, not by restriction

## Solution: Example-Driven Personas with Custom Tools

Instead of **filtering** helpers (rigid, limits flexibility), we **teach by example**:

1. **Persona identity** - "You are an expert web researcher..."
2. **Example workflows** - 2-3 concrete code examples showing approach
3. **Featured helpers** - Tools used in examples, shown with full docs
4. **Custom tools** - Domain-specific tools registered in runtime (NEW!)
5. **Generic access** - `helpers()` still available as escape hatch

### Key Insight

LLMs learn better from examples than from API documentation. By showing working code patterns, the model learns:
- **What** tools to use
- **How** to combine them
- **When** to use each approach
- **Why** this workflow is effective

The model can still access any helper via `helpers()` when needed, but it naturally gravitates toward the patterns shown in examples.

## Three-Layer Prompt Architecture

### Layer 1: System Execution Flow (Constant)

Technical mechanics that never change:
- How `<helpers>` blocks execute
- How results return in `<helpers_result>` tags
- Variable persistence across blocks
- When to use `</complete>`

### Layer 2: Meta Execution Patterns (Constant)

Strategic patterns that apply universally:
- When to use llm_call vs direct answers
- How to use sabre_call for task delegation
- Data binding with llm_bind/pandas_bind
- Result verification and validation

### Layer 3: Persona Examples (Variable)

Domain-specific examples showing this persona's approach:
- **Web Researcher**: Search → Download → Extract → Cross-reference → Synthesize
- **Coder**: Read → Execute → Analyze → Fix → Verify
- **Data Analyst**: Load → Clean → Visualize → Analyze → Report

## SABRE Personas

### 1. Default (General Purpose)

**Identity:**
```
You are a helpful AI assistant. You solve problems by breaking them down into
smaller tasks and using the available Python helpers to execute those tasks.
```

**Examples:** None (or minimal generic example)

**Featured Helpers:** All helpers shown equally

**Use case:** General purpose, exploratory tasks, when domain is unclear

### 2. Web Researcher

**Identity:**
```
You are an expert web researcher skilled at finding accurate information,
analyzing multiple sources, and synthesizing comprehensive, well-cited answers.
You excel at fact-checking and source evaluation.
```

**Example Workflows:**
```
## Example 1: Find recent information on a topic
<helpers>
# 1. Search for recent sources
results = Search.search("quantum computing 2024 breakthroughs", num_results=5)

# 2. Download top results as screenshots
content = download(results[:3])

# 3. Extract key information using llm_call
findings = llm_call(content, "Extract main points about quantum computing breakthroughs in 2024")

result(findings)
</helpers>

## Example 2: Verify a claim with multiple sources
<helpers>
# 1. Search for authoritative sources from different perspectives
scientific = Search.search("climate change scientific evidence 2024")
fact_check = Search.search("climate change fact check reuters")

# 2. Download both sets
all_content = download(scientific[:2] + fact_check[:2])

# 3. Cross-reference using llm_call
verification = llm_call(
    all_content,
    "Verify the claim. Cite sources. Note contradictions and consensus."
)

result(verification)
</helpers>

## Example 3: Deep research with sub-questions
<helpers>
# 1. Break down the question
sub_questions = llm_call(
    "Who invented the internet?",
    "Break this into 3-4 sub-questions covering different aspects"
)

# 2. Research each sub-question
answers = []
for question in sub_questions:
    results = Search.search(question)
    content = download(results[:2])
    answer = llm_call(content, f"Answer: {question}")
    answers.append(answer)

# 3. Synthesize final answer with citations
final = llm_call(answers, "Synthesize a comprehensive answer with sources")
result(final)
</helpers>
```

**Featured Helpers:**
- `Search.search(query, num_results=10)` - DuckDuckGo search
- `download(urls)` - Download pages as screenshots/files
- `llm_call(expr_list, instructions)` - Analyze and extract information
- `result(value)` - Return final answer

**Use case:** Research tasks, fact-checking, information gathering, source analysis

### 3. Coder

**Identity:**
```
You are an expert programmer who helps with coding tasks, debugging, and
software development. You write clean, well-tested code and follow best practices.
```

**Example Workflows:**
```
## Example 1: Debug a Python script
<helpers>
# 1. Read the file
code = FS.read_file("script.py")

# 2. Identify the issue using llm_call
analysis = llm_call(code, "What's causing the error? Explain the bug.")

# 3. Generate fix
fixed_code = llm_call([code, analysis], "Generate the corrected version")

# 4. Write back
FS.write_file("script.py", fixed_code)

result("Fixed the bug. See script.py")
</helpers>

## Example 2: Run tests and fix failures
<helpers>
# 1. Run test suite
test_output = Bash.execute("pytest tests/ -v")

# 2. If failures, analyze them
if "FAILED" in test_output:
    failures = llm_call(test_output, "Extract which tests failed and why")

    # 3. Read relevant code
    code = FS.read_file("src/module.py")

    # 4. Generate fixes
    fixes = llm_call([code, failures], "Generate fixes for the failing tests")

    result(fixes)
else:
    result("All tests passed! ✓")
</helpers>

## Example 3: Refactor code
<helpers>
# 1. Read the code to refactor
code = FS.read_file("legacy_module.py")

# 2. Analyze structure
analysis = llm_call(code, "Identify code smells and refactoring opportunities")

# 3. Generate refactored version
refactored = llm_call(
    [code, analysis],
    "Refactor this code: extract functions, improve naming, add type hints"
)

# 4. Write to new file
FS.write_file("module_refactored.py", refactored)

result("Refactored code written to module_refactored.py")
</helpers>
```

**Featured Helpers:**
- `FS.read_file(path)` - Read file contents
- `FS.write_file(path, content)` - Write files
- `FS.list_files(directory, pattern)` - List files
- `Bash.execute(command)` - Run shell commands
- `llm_call(expr_list, instructions)` - Analyze code, generate fixes

**Use case:** Programming tasks, debugging, refactoring, test fixing, code review

### 4. Data Analyst

**Identity:**
```
You are an expert data analyst skilled at working with datasets, creating
visualizations, and extracting insights from data. You write clear, reproducible
analyses.
```

**Example Workflows:**
```
## Example 1: Analyze CSV data
<helpers>
# 1. Download the data
csv_path = Web.download_csv("https://example.com/data.csv")

# 2. Load with pandas
import pandas as pd
df = pd.read_csv(csv_path)

# 3. Create smart DataFrame
smart_df = pandas_bind(df)

# 4. Ask questions
insights = smart_df.ask("What are the key trends? Any outliers?")

result(insights)
</helpers>

## Example 2: Create visualization
<helpers>
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sales_data.csv")

# Create visualization
plt.figure(figsize=(12, 6))
df.groupby('month')['revenue'].sum().plot(kind='bar')
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)

# Figure is auto-captured and displayed
result("See revenue trend visualization above")
</helpers>

## Example 3: Compare datasets
<helpers>
import pandas as pd

# Load two datasets
df_2023 = pd.read_csv("2023_data.csv")
df_2024 = pd.read_csv("2024_data.csv")

# Get summary statistics
summary_2023 = df_2023.describe().to_string()
summary_2024 = df_2024.describe().to_string()

# Use llm_call to analyze differences
comparison = llm_call(
    [summary_2023, summary_2024],
    "Compare these datasets. What changed? What are the key trends?"
)

result(comparison)
</helpers>
```

**Featured Helpers:**
- `Web.download_csv(url)` - Download CSV files
- `pandas_bind(df)` - Create smart DataFrame with `.ask()` method
- `llm_call(expr_list, instructions)` - Analyze data, generate insights
- `matplotlib` - Create visualizations (auto-captured)

**Use case:** Data analysis, visualization, statistics, dataset comparison

## Custom Tools: Domain-Specific Functions

### Problem: Benchmarks and Specialized Domains

SABRE's built-in helpers (Search, Bash, llm_call, etc.) are general-purpose. But some use cases need **domain-specific tools**:

- **tau2-bench**: Retail functions like `find_user_id_by_name_zip()`, `get_order_details()`
- **Healthcare**: `lookup_patient_record()`, `check_insurance_eligibility()`
- **Banking**: `get_account_balance()`, `transfer_funds()`
- **DevOps**: `query_database()`, `search_logs()`, `send_slack_message()`

These tools don't exist in SABRE's runtime by default, causing `NameError` when the LLM tries to call them.

### Solution: Custom Tools with Internal/External Execution

> **✅ IMPLEMENTATION STATUS**: The foundation is complete! ToolRegistry and external tool support implemented (2025-01-05). See `docs/EXTERNAL_TOOLS_IMPLEMENTATION.md` for complete documentation and `tests/test_external_tools.py` for usage examples.

Personas can define **custom tools** that are either:
1. **Executed internally by SABRE** (internal mode)
2. **Returned to caller for execution** (external mode)

This enables two key use cases:
- **Benchmarks** (external): Tools returned to benchmark for execution and scoring
- **Interactive** (internal): Tools executed by SABRE with Python/MCP implementations

### Two Execution Modes

#### 1. External Mode (For Benchmarks)

> **✅ IMPLEMENTED**: External tools fully working with ToolRegistry and pause/resume orchestration.

Tools are **documented but not executed** by SABRE. When LLM calls them, orchestration pauses and returns tool calls to caller for execution.

```yaml
custom_tools:
  find_user_id_by_name_zip:
    description: "Find user ID by first name, last name, and zip code"
    execution_mode: external  # Tool returned to caller
    parameters:
      first_name:
        type: str
        required: true
        description: "Customer's first name"
      last_name:
        type: str
        required: true
        description: "Customer's last name"
      zip:
        type: str
        required: true
        description: "Customer's 5-digit zip code"
    returns:
      type: str
      description: "Unique user ID"
```

**How it works:**
1. Tool registered in ToolRegistry as EXTERNAL
2. LLM generates: `<helpers>find_user_id_by_name_zip(...)</helpers>`
3. Orchestrator detects external tool, **pauses orchestration**
4. Returns `OrchestrationResult(status="awaiting_tool_results", pending_tool_calls=[...])`
5. Caller (tau2-bench) executes tool in their environment
6. Caller provides results back via `orchestrator.continue_with_tool_results()`
7. Orchestration **resumes** with real results

**Benefits:**
- ✅ No stub pollution in runtime
- ✅ Real tool execution by domain owner
- ✅ Accurate benchmark scoring (real calls tracked)
- ✅ Caller has full control

#### 2. Internal Mode (For Interactive Use)

> **🚧 PARTIALLY IMPLEMENTED**: ToolRegistry supports internal tools. Need formal `PythonRuntime.register_tools()` API.

Tools **executed by SABRE** using Python implementations or MCP server references.

**Python file:**
```yaml
custom_tools:
  calculate_risk_score:
    description: "Calculate customer risk score"
    execution_mode: internal  # SABRE executes
    implementation: "~/.config/sabre/tools/risk_scoring.py::calculate_risk"
    parameters:
      user_id:
        type: str
        required: true
    returns:
      type: float
```

**MCP server reference (future):**
```yaml
custom_tools:
  query_database:
    description: "Execute SQL query on production database"
    execution_mode: internal
    implementation: "mcp://postgres-mcp/query"  # References MCP server
    parameters:
      query:
        type: str
        required: true
        description: "SQL query to execute"
    returns:
      type: list
      description: "Query results as list of dicts"
```

**How it works:**
1. Tool registered in ToolRegistry as INTERNAL with callable
2. Tool added to PythonRuntime namespace
3. LLM generates: `<helpers>calculate_risk_score(user_id="123")</helpers>`
4. Orchestrator executes tool in runtime (like Search, Bash, etc.)
5. Results returned in `<helpers_result>` tags
6. Orchestration continues automatically

### Custom Tools Workflow

> **✅ UPDATED**: Workflow revised based on implementation. External tools use ToolRegistry, not runtime namespace.

#### For External Tools (Benchmarks)
1. **Declaration**: Persona YAML declares custom tools with `execution_mode: external`
2. **Registration**: ToolRegistry.register_external(name, schema, description)
3. **Documentation**: Featured helpers section includes custom tools
4. **Execution**: Orchestrator detects external tools, pauses, returns to caller
5. **Resumption**: Caller provides results, orchestration resumes

#### For Internal Tools (Interactive)
1. **Declaration**: Persona YAML declares custom tools with `execution_mode: internal`
2. **Loading**: PersonaLoader loads Python implementations or MCP references
3. **Registration**: runtime.register_tools() adds callables to namespace
4. **Availability**: Tools appear in namespace, usable in `<helpers>` blocks
5. **Documentation**: Featured helpers section includes custom tools
6. **Execution**: Orchestrator executes normally like Search, Bash, etc.

### Example: tau2-retail Persona

Complete persona for tau2-bench evaluation:

```yaml
personas:
  tau2-retail:
    name: "Tau2 Retail Customer Service"
    description: "E-commerce customer service agent for tau2-bench evaluation"

    identity: |
      You are a helpful e-commerce customer service agent for an online retail store.
      Your responsibilities include helping customers find orders, process returns,
      and answer product questions. Always verify customer identity (name + zip code)
      before accessing account information.

    examples: |
      ## Example 1: Help customer track order

      User: "Hi, I'm Sarah Johnson from zip 90210. Where's my order?"

      <helpers>
      # 1. Find customer by name and zip
      user_id = find_user_id_by_name_zip(
          first_name="Sarah",
          last_name="Johnson",
          zip="90210"
      )

      # 2. Get order details
      orders = get_order_details(user_id=user_id)

      # 3. Check most recent order status
      if orders:
          latest = orders[0]
          result(f"Your order #{latest['id']} is {latest['status']}")
      else:
          result("I don't see any orders for your account")
      </helpers>

      ## Example 2: Process return

      User: "I want to return my recent order. Michael Chen, 02139"

      <helpers>
      # 1. Look up customer
      user_id = find_user_id_by_name_zip(
          first_name="Michael",
          last_name="Chen",
          zip="02139"
      )

      # 2. Get recent order
      orders = get_order_details(user_id=user_id)
      latest_order_id = orders[0]['id']

      # 3. Initiate return
      updated = update_order_status(
          order_id=latest_order_id,
          status="return_initiated"
      )

      result("Return initiated! You'll receive a return label via email.")
      </helpers>

    featured_helpers:
      - "find_user_id_by_name_zip"
      - "get_order_details"
      - "update_order_status"
      - "process_refund"
      - "result"

    custom_tools:
      find_user_id_by_name_zip:
        description: "Find user ID by customer name and zip code"
        execution_mode: external  # tau2-bench executes this
        parameters:
          first_name:
            type: str
            required: true
            description: "Customer's first name"
          last_name:
            type: str
            required: true
            description: "Customer's last name"
          zip:
            type: str
            required: true
            description: "Customer's 5-digit zip code"
        returns:
          type: str
          description: "Unique user ID"

      get_order_details:
        description: "Retrieve all orders for a user"
        execution_mode: external  # tau2-bench executes this
        parameters:
          user_id:
            type: str
            required: true
            description: "User ID from find_user_id_by_name_zip"
        returns:
          type: list
          description: "List of order objects with id, status, items, etc."

      update_order_status:
        description: "Update the status of an order"
        execution_mode: external  # tau2-bench executes this
        parameters:
          order_id:
            type: str
            required: true
            description: "Order ID to update"
          status:
            type: str
            required: true
            description: "New status: shipped, delivered, return_initiated, refunded"
        returns:
          type: dict
          description: "Updated order object"

      process_refund:
        description: "Process a refund for an order"
        execution_mode: external  # tau2-bench executes this
        parameters:
          order_id:
            type: str
            required: true
            description: "Order ID to refund"
          amount:
            type: float
            required: false
            description: "Refund amount (defaults to full order amount)"
          reason:
            type: str
            required: false
            description: "Reason for refund"
        returns:
          type: dict
          description: "Refund confirmation with transaction ID"
```

### MCP Servers (Separate Infrastructure)

MCP servers are configured **separately** from personas, at the orchestrator/server level:

```yaml
# ~/.config/sabre/mcp_servers.yaml
servers:
  postgres-mcp:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "${POSTGRES_URL}"

  filesystem-mcp:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
    args_extra: ["/data", "/var/log"]

  slack-mcp:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-slack"]
    env:
      SLACK_BOT_TOKEN: "${SLACK_BOT_TOKEN}"
```

Personas reference MCP servers via `implementation: "mcp://server/tool"` syntax in interactive mode.

**Benefits of Separation:**
- ✅ One MCP server serves multiple personas
- ✅ Clean config (infrastructure vs domain logic)
- ✅ Personas don't manage server lifecycle
- ✅ MCP is just one way to implement interactive tools

## Persona Configuration Format

### personas.yaml

```yaml
personas:
  default:
    name: "SABRE Default"
    description: "General purpose AI assistant"

    identity: |
      You are a helpful AI assistant. You solve problems by breaking them down
      into smaller tasks and using the available Python helpers to execute those tasks.

    examples: ""  # No specific examples

    featured_helpers:
      - "*"  # All helpers shown equally

  web-researcher:
    name: "Web Research Specialist"
    description: "Expert at finding and analyzing information from multiple sources"

    identity: |
      You are an expert web researcher skilled at finding accurate information,
      analyzing multiple sources, and synthesizing comprehensive, well-cited answers.
      You excel at fact-checking and source evaluation.

    examples: |
      ## Example Workflows

      Here are examples of how you approach common research tasks:

      ### Example 1: Find recent information on a topic

      <helpers>
      # 1. Search for recent sources
      results = Search.search("quantum computing 2024 breakthroughs", num_results=5)

      # 2. Download top results as screenshots
      content = download(results[:3])

      # 3. Extract key information using llm_call
      findings = llm_call(content, "Extract main points about quantum computing breakthroughs in 2024")

      result(findings)
      </helpers>

      ### Example 2: Verify a claim with multiple sources

      <helpers>
      # 1. Search for authoritative sources
      scientific = Search.search("climate change scientific evidence 2024")
      fact_check = Search.search("climate change fact check reuters")

      # 2. Download both sets
      all_content = download(scientific[:2] + fact_check[:2])

      # 3. Cross-reference using llm_call
      verification = llm_call(
          all_content,
          "Verify the claim. Cite sources. Note contradictions and consensus."
      )

      result(verification)
      </helpers>

    featured_helpers:
      - "Search.search"
      - "download"
      - "Web.download_csv"
      - "Browser.screenshot"
      - "llm_call"
      - "llm_bind"
      - "llm_list_bind"
      - "result"

  coder:
    name: "Programming Assistant"
    description: "Expert at coding, debugging, and software development"

    identity: |
      You are an expert programmer who helps with coding tasks, debugging, and
      software development. You write clean, well-tested code and follow best practices.

    examples: |
      ## Example Workflows

      Here are examples of how you approach common coding tasks:

      ### Example 1: Debug a Python script

      <helpers>
      # 1. Read the file
      code = FS.read_file("script.py")

      # 2. Identify the issue
      analysis = llm_call(code, "What's causing the error? Explain the bug.")

      # 3. Generate fix
      fixed_code = llm_call([code, analysis], "Generate the corrected version")

      # 4. Write back
      FS.write_file("script.py", fixed_code)

      result("Fixed the bug. See script.py")
      </helpers>

      ### Example 2: Run tests and fix failures

      <helpers>
      # 1. Run test suite
      test_output = Bash.execute("pytest tests/ -v")

      # 2. If failures, fix them
      if "FAILED" in test_output:
          failures = llm_call(test_output, "Extract which tests failed and why")
          code = FS.read_file("src/module.py")
          fixes = llm_call([code, failures], "Generate fixes")
          result(fixes)
      else:
          result("All tests passed! ✓")
      </helpers>

    featured_helpers:
      - "FS.read_file"
      - "FS.write_file"
      - "FS.list_files"
      - "Bash.execute"
      - "llm_call"
      - "result"

  data-analyst:
    name: "Data Analysis Specialist"
    description: "Expert at analyzing datasets and creating visualizations"

    identity: |
      You are an expert data analyst skilled at working with datasets, creating
      visualizations, and extracting insights from data. You write clear, reproducible
      analyses.

    examples: |
      ## Example Workflows

      Here are examples of how you approach data analysis tasks:

      ### Example 1: Analyze CSV data

      <helpers>
      # 1. Download the data
      csv_path = Web.download_csv("https://example.com/data.csv")

      # 2. Load and analyze
      import pandas as pd
      df = pd.read_csv(csv_path)
      smart_df = pandas_bind(df)

      # 3. Ask questions
      insights = smart_df.ask("What are the key trends? Any outliers?")

      result(insights)
      </helpers>

      ### Example 2: Create visualization

      <helpers>
      import pandas as pd
      import matplotlib.pyplot as plt

      df = pd.read_csv("sales_data.csv")

      plt.figure(figsize=(12, 6))
      df.groupby('month')['revenue'].sum().plot(kind='bar')
      plt.title("Monthly Revenue")

      result("See revenue trend visualization above")
      </helpers>

    featured_helpers:
      - "Web.download_csv"
      - "pandas_bind"
      - "llm_call"
      - "coerce"
      - "result"
      - "matplotlib"
```

## Prompt Template Structure

```
[system_message]
{{persona_identity}}

[Context: date, timezone, working directory, etc.]

[user_message]

## System Execution Flow

How the continuation system works:
* Emit Python code in <helpers></helpers> blocks
* Results returned in <helpers_result> tags
* Variables persist across blocks
* Finish with </complete>

## Meta Execution Patterns

**When to use direct answers vs helpers:**
* Simple questions → direct answer + </complete>
* Needs tools/data → use <helpers> blocks

**Text analysis:**
* Use llm_call(expr_list, instructions) for analysis/extraction

**Task delegation:**
* Use sabre_call(description, expr_list) for sub-tasks with fresh context

**Data binding:**
* Use llm_bind(data, "func_signature") for structured extraction
* Use pandas_bind(df) for smart DataFrames

{{persona_examples}}

## Featured Helpers

{{featured_helpers_docs}}

## Generic Helper Access

If you need a helper not shown above, call helpers() to see all available functions.
Use helpers("search_term") to find specific functionality.
```

## Implementation Architecture

### Components

> **✅ IMPLEMENTATION STATUS**: ToolRegistry and Orchestrator enhancements complete. Persona infrastructure still needed.

1. **ToolRegistry** (`sabre/server/tool_registry.py`) - **✅ IMPLEMENTED**
   - Track internal vs external tools
   - Register tools with execution mode and metadata
   - Lookup by name to determine execution strategy
   - Support both internal callables and external schemas

2. **Orchestrator Updates** (`sabre/server/orchestrator.py`) - **✅ IMPLEMENTED**
   - Accept `tool_registry` parameter in `__init__`
   - Partition helpers into internal vs external
   - Execute internal tools normally
   - Pause orchestration for external tools
   - Return pending_tool_calls to caller
   - `continue_with_tool_results()` method for resumption

3. **PersonaLoader** (`sabre/config/persona_loader.py`) - **🚧 TODO**
   - Load persona configs from YAML
   - Support user overrides in `~/.config/sabre/personas.yaml`
   - Validate persona structure
   - Process `custom_tools` section
   - Create ToolRegistry from persona's custom_tools

4. **PromptBuilder** (`sabre/server/prompt_builder.py`) - **🚧 TODO**
   - Build prompts with persona template variables
   - Inject identity, examples, featured helpers
   - Generate helper documentation for featured helpers only
   - Include custom tools in documentation

5. **PythonRuntime Updates** (`sabre/server/python_runtime.py`) - **🚧 TODO**
   - Add formal `register_tools(tools)` method
   - Add internal custom tools to namespace
   - Handle async tools (wrap for sync execution in `<helpers>` blocks)

6. **ToolImplementationLoader** (`sabre/config/tool_implementation_loader.py`) - **🚧 TODO** (replaces ToolStubGenerator)
   - Load Python implementations for internal tools
   - Support file references: `~/.config/sabre/tools/file.py::function`
   - Future: Load MCP server references: `mcp://server/tool`

7. **MCPClientManager** (`sabre/config/mcp_client_manager.py`) - **⏸️ DEFERRED**
   - Manage MCP server connections (separate from personas)
   - Load from `~/.config/sabre/mcp_servers.yaml`
   - Route tool calls to appropriate MCP servers
   - **Status**: Not needed for tau2-bench, add later as enhancement

### File Structure

```
sabre/
├── config/
│   ├── personas.yaml                    # Default persona definitions (TODO)
│   ├── persona_loader.py                # Persona loading logic (TODO)
│   ├── tool_implementation_loader.py    # Load Python/MCP implementations (TODO)
│   └── mcp_servers.yaml                 # MCP server configuration (DEFERRED)
├── server/
│   ├── tool_registry.py                 # ✅ IMPLEMENTED: Internal/external tool tracking
│   ├── orchestrator.py                  # ✅ IMPLEMENTED: Tool partitioning, pause/resume
│   ├── python_runtime.py                # TODO: register_tools() method
│   ├── prompt_builder.py                # TODO: Build prompts with personas
│   └── prompts/
│       └── continuation.prompt          # Base template with {{variables}} (TODO)
├── benchmarks/
│   └── tau2/
│       └── sabre_agent.py               # ✅ UPDATED: Uses ToolRegistry, demonstrates pattern
```

## Migration Path

> **✅ UPDATE (2025-01-05)**: Phase 2 significantly advanced by external tools implementation. Timeline revised.

### Phase 0: External Tools Foundation ✅ COMPLETE

**Goal:** ✅ Build tool execution infrastructure for internal/external tools

**Completed:**
- ✅ Created `ToolRegistry` class for internal/external tool tracking
- ✅ Enhanced `Orchestrator` with tool partitioning logic
- ✅ Implemented pause/resume orchestration for external tools
- ✅ Added `continue_with_tool_results()` method
- ✅ Updated SABRE agent to demonstrate pattern
- ✅ Comprehensive tests (13 tests passing)

**Files Completed:**
- ✅ NEW: `sabre/server/tool_registry.py`
- ✅ MODIFIED: `sabre/server/orchestrator.py`
- ✅ MODIFIED: `sabre/common/models/execution_tree.py`
- ✅ MODIFIED: `sabre/benchmarks/tau2/sabre_agent.py`
- ✅ NEW: `tests/test_external_tools.py`
- ✅ NEW: `docs/EXTERNAL_TOOLS_IMPLEMENTATION.md`

### Phase 1: Persona Infrastructure (Week 1)

**Goal:** Get persona system working with default persona only

**Tasks:**
1. Create `personas.yaml` with just `default` persona
2. Create `PersonaLoader` class
3. Create `PromptBuilder` class
4. Update `Orchestrator` to accept `persona` parameter (already has `tool_registry`)
5. Test that default persona works identically to current behavior

**Files:**
- NEW: `sabre/config/personas.yaml`
- NEW: `sabre/config/persona_loader.py`
- NEW: `sabre/server/prompt_builder.py`
- MODIFIED: `sabre/server/orchestrator.py` (add persona support)

**Reduced Scope:** Focus on configuration and prompting. Tool execution already done.

### Phase 2: Custom Tools Integration (Week 2) - **70% COMPLETE**

**Goal:** Integrate custom tools with persona system

**Remaining Tasks:**
1. Add formal `register_tools()` method to `PythonRuntime`
2. Update `PersonaLoader` to process `custom_tools` section and create ToolRegistry
3. Create `ToolImplementationLoader` for loading Python implementations
4. Create `tau2-retail` persona configuration
5. Test external tools via persona system

**Files:**
- MODIFIED: `sabre/server/python_runtime.py` (add formal `register_tools()` API)
- MODIFIED: `sabre/config/persona_loader.py` (create ToolRegistry from persona)
- NEW: `sabre/config/tool_implementation_loader.py` (load Python files)
- NEW: `sabre/config/personas/tau2-retail.yaml`

**Already Complete:**
- ✅ ToolRegistry infrastructure
- ✅ Orchestrator tool partitioning
- ✅ External tool execution pattern
- ✅ Demonstration in SABRE agent

### Phase 3: Example Personas (Week 3)

**Goal:** Add web-researcher, coder, data-analyst personas

**Tasks:**
1. Write example workflows for each persona
2. Define featured helpers for each
3. Test each persona independently
4. Validate examples are syntactically correct

**Files:**
- MODIFIED: `sabre/config/personas.yaml`

### Phase 4: MCP Integration (Week 4) - **DEFERRED**

**Goal:** Support MCP servers for interactive tools

**Status:** ⏸️ Not critical path. Can be added later as enhancement.

**Tasks (when needed):**
1. Create `MCPClientManager` class
2. Add MCP server configuration (`mcp_servers.yaml`)
3. Update `ToolImplementationLoader` to handle `mcp://` references
4. Test with postgres-mcp and filesystem-mcp
5. Create DevOps persona with MCP tools

**Files:**
- NEW: `sabre/config/mcp_client_manager.py`
- NEW: `sabre/config/mcp_servers.yaml`
- MODIFIED: `sabre/config/tool_implementation_loader.py`

**Rationale for Deferral:**
- Not needed for tau2-bench integration
- External tool pattern works without MCP
- Can add when interactive internal tools are needed
- Reduces scope and complexity for initial release

### Phase 5: Server Integration (Week 5)

**Goal:** Allow selecting persona at startup

**Tasks:**
1. Add `--persona` CLI arg to server
2. Pass persona through to orchestrator
3. Add `/persona` slash command to show current persona
4. Test persona switching

**Files:**
- MODIFIED: `sabre/server/__main__.py`
- MODIFIED: `sabre/client/slash_commands/` (new persona command)

### Phase 6: Documentation (Week 6)

**Goal:** Document persona system

**Tasks:**
1. Update CLAUDE.md with persona info
2. Add persona examples to README
3. Create persona authoring guide
4. Document custom tools specification
5. Document MCP integration

**Files:**
- MODIFIED: `CLAUDE.md`
- MODIFIED: `README.md`
- NEW: `docs/PERSONA_AUTHORING.md`
- NEW: `docs/CUSTOM_TOOLS.md`

## Benefits

### ✅ Learn by Example
- Models learn patterns better than API docs
- Working code is clearer than descriptions
- Shows not just WHAT but HOW and WHY

### ✅ Token Efficient
- Featured helpers get full docs (~500 tokens)
- Other helpers accessible but not in prompt (~5000 tokens saved)
- Examples teach patterns without verbose documentation

### ✅ Flexible
- Not locked into filtered helper set
- Can use `helpers()` to discover any function
- Examples guide but don't restrict

### ✅ Easy to Author
- Just write working code examples
- YAML configuration, no code changes
- User overrides in `~/.config/sabre/`

### ✅ Composable
- Personas define identity + examples
- Same base execution model
- Can mix and match approaches

### ✅ Domain-Specific Tools (NEW)
- Benchmark stubs for evaluation (tau2-bench)
- Python implementations for custom logic
- MCP integration for databases, filesystems, APIs
- API wrappers for REST endpoints
- No code changes needed for new domains

### ✅ Complete Domain Packages
- Identity, examples, helpers, AND custom tools in one config
- Tools are declared, demonstrated, documented together
- MCP servers separate (infrastructure vs domain logic)
- Reusable across personas

## Testing Checklist

### Core Persona System
- [ ] PersonaLoader loads default persona correctly
- [ ] PersonaLoader loads custom personas from YAML
- [ ] PersonaLoader supports user config overrides
- [ ] PromptBuilder injects identity correctly
- [ ] PromptBuilder injects examples correctly
- [ ] PromptBuilder generates featured helper docs only
- [ ] Orchestrator passes persona to prompt builder
- [ ] Default persona works identically to current behavior
- [ ] Web researcher persona uses Search + download pattern
- [ ] Coder persona uses FS + Bash pattern
- [ ] Data analyst persona uses pandas + matplotlib
- [ ] helpers() works in all personas
- [ ] --persona CLI flag works
- [ ] /persona slash command shows current persona

### Custom Tools
- [ ] ToolStubGenerator creates benchmark stubs correctly
- [ ] Benchmark stubs log calls and return mock data
- [ ] PythonRuntime register_tools() adds tools to namespace
- [ ] Custom tools callable in <helpers> blocks
- [ ] tau2-retail persona registers all custom tools
- [ ] tau2-bench integration works (>0.0 reward score)
- [ ] ToolStubGenerator loads Python implementations (interactive mode)
- [ ] ToolStubGenerator generates API wrappers (api mode)
- [ ] Custom tools appear in helpers() output
- [ ] Custom tools documented in featured helpers section

### MCP Integration
- [ ] MCPClientManager connects to MCP servers
- [ ] MCPClientManager loads from mcp_servers.yaml
- [ ] ToolStubGenerator handles mcp:// references
- [ ] MCP tool wrappers work in <helpers> blocks
- [ ] Async MCP tools wrapped for sync execution
- [ ] MCP servers shared across multiple personas
- [ ] DevOps persona works with postgres-mcp + filesystem-mcp
- [ ] MCP server errors handled gracefully

## Open Questions

1. **Should examples be validated at load time?**
   - Pro: Catch syntax errors early
   - Con: Adds complexity, examples are just strings
   - **Recommendation**: Basic syntax check (parse as Python), not execution

2. **Should we support persona inheritance?**
   - e.g., `web-researcher` extends `default` with additional examples
   - **Recommendation**: Not in v1, add later if needed

3. **How many examples per persona?**
   - **Recommendation**: 2-4 examples, ~300 tokens total

4. **Should featured helpers be auto-detected from examples?**
   - Pro: DRY, no redundancy
   - Con: Less explicit, harder to understand config
   - **Recommendation**: Explicit list, easier to understand and override
