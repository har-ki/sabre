# Evaluating SABRE with tau2-bench

tau2-bench is a benchmark for evaluating conversational agents on multi-turn dialogues with tool use and policy adherence across realistic customer service scenarios.

## Overview

tau2-bench provides three domains:

- **retail**: E-commerce customer service (115 tasks)
  - Order management, returns, exchanges
  - Product inquiries and recommendations
  - Policy adherence (return windows, payment methods)

- **airline**: Flight booking and changes (50 tasks)
  - Flight search, booking, modifications
  - Seat selection, baggage handling
  - Complex policies (change fees, cabin restrictions)

- **telecom**: Technical support (100+ tasks, dual-control)
  - Device troubleshooting
  - Plan changes, billing issues
  - User actively participates with tools

## Quick Start

```bash
# Install tau2-bench
pip install git+https://github.com/sierra-research/tau2-bench

# Quick evaluation (5 tasks, 1 trial)
OPENAI_API_KEY=your_key sabre benchmark tau2 --quick

# Full evaluation on retail domain
OPENAI_API_KEY=your_key sabre benchmark tau2 --domain retail --trials 5

# All domains
OPENAI_API_KEY=your_key sabre benchmark tau2 --domain retail --trials 5
OPENAI_API_KEY=your_key sabre benchmark tau2 --domain airline --trials 5
OPENAI_API_KEY=your_key sabre benchmark tau2 --domain telecom --trials 5
```

## Manual Setup

For more control over the evaluation process:

### Step 1: Start SABRE Server

```bash
OPENAI_API_KEY=your_key uv run sabre-server
```

### Step 2: Start API Bridge

In a new terminal:

```bash
python -m sabre.benchmarks.tau2.api_bridge
```

This starts an OpenAI-compatible API server on port 8765.

### Step 3: Run Evaluation

In another terminal:

```bash
tau2 run \
  --domain retail \
  --agent-llm sabre \
  --agent-api-base http://localhost:8765/v1 \
  --user-llm gpt-4o \
  --num-trials 5
```

## Advanced Options

### Run Specific Tasks

```bash
# Run only tasks 1, 5, and 10
sabre benchmark tau2 --domain retail --task-ids 1,5,10

# Run first 20 tasks
sabre benchmark tau2 --domain airline --tasks 20
```

### Different Ports

```bash
# Custom SABRE and bridge ports
sabre benchmark tau2 \
  --sabre-port 8012 \
  --bridge-port 8766 \
  --domain retail
```

### Analyze Results

```bash
# Results are saved in data/tau2/simulations/
ls data/tau2/simulations/

# Analyze specific result file
python -m sabre.benchmarks.tau2.runner analyze \
  data/tau2/simulations/sabre_retail_20250103.json
```

## Evaluation Metrics

tau2-bench provides several metrics:

### Pass^k Success Rate
Measures consistency by running each task k times:
- **Pass^1**: Success rate on first attempt
- **Pass^2**: Success rate in at least 1 of 2 attempts
- **Pass^4**: Success rate in at least 1 of 4 attempts

Higher k values test consistency and reliability.

### Tool Use Accuracy
- Correct API function calls
- Proper parameter values
- Appropriate call sequencing

### Policy Adherence
- Following domain-specific rules
- Handling edge cases correctly
- Proper information gathering

## Results Directory

Results are saved in JSON format:

```
data/tau2/simulations/
├── sabre_retail_20250103_143025.json
├── sabre_airline_20250103_150132.json
└── sabre_telecom_20250103_153045.json
```

Each file contains:
- Task descriptions
- Full conversation trajectories
- Tool calls made
- Database state changes
- Success/failure reasons

## Troubleshooting

### SABRE Server Not Running

```
✗ SABRE server not running on port 8011
```

**Solution**: Start the server:
```bash
OPENAI_API_KEY=your_key uv run sabre-server
```

### tau2-bench Not Installed

```
tau2-bench not found. Install with:
  pip install git+https://github.com/sierra-research/tau2-bench
```

**Solution**: Install tau2-bench:
```bash
pip install git+https://github.com/sierra-research/tau2-bench
```

### Connection Refused

```
Cannot connect to SABRE server at http://localhost:8011
```

**Solution**: Check that SABRE server is running and accessible:
```bash
curl http://localhost:8011/health
```

## Performance Tips

### Optimize for tau2-bench

1. **Use specific prompts**: tau2-bench requires precise tool use
2. **Test incrementally**: Start with `--quick`, then scale up
3. **Monitor token usage**: Long conversations can be expensive
4. **Run overnight**: Full evaluations take 1-2 hours per domain

### Compare with Baselines

Track your improvements:

```bash
# Run evaluation and save results
sabre benchmark tau2 --domain retail --trials 5 > results_v1.txt

# Make improvements to SABRE
# ...

# Run again
sabre benchmark tau2 --domain retail --trials 5 > results_v2.txt

# Compare
diff results_v1.txt results_v2.txt
```

## References

- [tau2-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [tau2-bench Paper](https://arxiv.org/abs/2506.07982)
- [Original tau-bench Paper](https://arxiv.org/abs/2406.12045)
- [Leaderboard](https://taubench.com)

## Contributing

Improvements to the integration are welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

Areas for contribution:
- Better tool call parsing
- Result visualization
- Domain-specific optimizations
- Additional metrics and analysis
