#!/bin/bash
# Wrapper script to run SABRE x tau2-bench evaluation
# Usage: ./run_tau2_eval.sh TASK_ID [--model MODEL] [--domain DOMAIN] [additional args...]

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY=your-key-here"
    exit 1
fi

# Set default TAU2_DATA_DIR if not already set
export TAU2_DATA_DIR=${TAU2_DATA_DIR:-/Users/hnayak/Documents/workspace/tau2-bench/data}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to SABRE root directory (3 levels up: tau2 -> benchmarks -> sabre -> root)
SABRE_ROOT="$SCRIPT_DIR/../../.."
cd "$SABRE_ROOT"

# Run with uv to ensure correct environment
# OPENAI_API_KEY is automatically passed through to child processes
uv run python -m sabre.benchmarks.tau2.sabre_tau2_dialogue_runner "$@"
