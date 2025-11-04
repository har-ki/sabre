# SABRE Benchmarks

This directory contains integrations with various agent benchmarks to evaluate SABRE's performance.

## Available Benchmarks

### tau2-bench

Multi-turn conversational agent benchmark with tool use and policy adherence testing.

- **Domains**: retail, airline, telecom
- **Tasks**: 165+ real-world customer service scenarios
- **Metrics**: Pass^k success rates, tool use accuracy, policy adherence

**Quick Start:**
```bash
OPENAI_API_KEY=your_key sabre benchmark tau2 --quick
```

See `tau2/README.md` for detailed documentation.

## Adding New Benchmarks

To add a new benchmark integration:

1. Create a new directory: `sabre/benchmarks/your_benchmark/`
2. Implement the agent adapter
3. Add CLI commands in `sabre/cli.py`
4. Add tests in `tests/benchmarks/`
5. Document in `docs/benchmarks/`
