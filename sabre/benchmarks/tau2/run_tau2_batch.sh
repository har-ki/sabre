#!/bin/bash
# Wrapper script for running batch tau2-bench evaluation
# Usage examples:
#   ./run_tau2_batch.sh --num-tasks 10                  # Run first 10 tasks
#   ./run_tau2_batch.sh --start 0 --end 20              # Run tasks 0-19
#   ./run_tau2_batch.sh --all                           # Run ALL retail tasks
#   ./run_tau2_batch.sh --num-tasks 5 --model gpt-4o-mini  # Use different model

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
uv run python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner "$@"
