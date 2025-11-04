# tau2-bench Integration for SABRE

This directory contains the integration between SABRE and tau2-bench, a benchmark for evaluating conversational agents on multi-turn dialogues with tool use and policy adherence.

## Overview

tau2-bench tests agents on realistic customer service scenarios across three domains:
- **retail**: E-commerce (115 tasks)
- **airline**: Flight booking (50 tasks)
- **telecom**: Technical support with dual-control (100+ tasks)

## Quick Start

```bash
# Make sure SABRE server is running
OPENAI_API_KEY=your_key uv run sabre-server

# In another terminal, run evaluation
OPENAI_API_KEY=your_key sabre benchmark tau2 --quick
```

## Manual Setup

If you need more control over the evaluation:

```bash
# Terminal 1: Start SABRE server
OPENAI_API_KEY=your_key uv run sabre-server

# Terminal 2: Start API bridge
python -m sabre.benchmarks.tau2.api_bridge

# Terminal 3: Run tau2-bench
tau2 run \
  --domain retail \
  --agent-llm sabre \
  --agent-api-base http://localhost:8765/v1 \
  --user-llm gpt-4o \
  --num-trials 5
```

## Architecture

```
tau2-bench ──HTTP──> API Bridge ──Internal──> SABRE Server
(evaluator)    (OpenAI API)   (FastAPI)         (agent)
```

## Components

### agent.py
The main adapter that translates between tau2-bench's OpenAI-compatible API and SABRE's internal format.

### api_bridge.py
FastAPI server that exposes an OpenAI-compatible API endpoint for tau2-bench to call.

### runner.py
CLI utilities and Python API for running evaluations and analyzing results.

## Evaluation Metrics

tau2-bench provides:
- **Pass^k**: Success rate over k repeated trials
- **Tool use accuracy**: Correct API function calls
- **Policy adherence**: Following domain-specific rules

## Results

Results are saved in `data/tau2/simulations/` with detailed trajectories showing:
- Full conversation history
- Tool calls made
- Database state changes
- Success/failure reasons

## Development

To improve the integration:

1. Enhance tool call parsing in `agent.py`
2. Add result visualization in `runner.py`
3. Optimize prompt formatting for better performance
4. Add domain-specific optimizations

## References

- [tau2-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [tau2-bench Paper](https://arxiv.org/abs/2506.07982)
- [SABRE Documentation](../../README.md)
