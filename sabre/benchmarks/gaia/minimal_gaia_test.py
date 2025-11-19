#!/usr/bin/env python3
"""
Minimal GAIA test

NOTE: This requires inspect-evals >= 0.3.103 (unreleased as of Nov 2025)
The fix for metadata.jsonl -> metadata.parquet was merged on Nov 8, 2025 (PR #660)

To install the fixed version from GitHub:
    uv pip install git+https://github.com/UKGovernmentBEIS/inspect_evals.git

After installing, clear caches:
    rm -rf ~/.cache/inspect_evals/gaia_dataset ~/.cache/huggingface/datasets/gaia*
"""
from inspect_ai import eval
from inspect_evals.gaia import gaia
import os


# Ensure HF_TOKEN is set
if not os.environ.get('HF_TOKEN'):
    print("ERROR: HF_TOKEN not set")
    print("Please set it: export HF_TOKEN=hf_...")
    exit(1)

# Ensure OPENAI_API_KEY is set
if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: OPENAI_API_KEY not set")
    print("Please set it: export OPENAI_API_KEY=sk-...")
    exit(1)

# print("=" * 80)
# print("Testing vanilla GPT-4o-mini on GAIA validation set")
# print("Using inspect-evals with parquet support (PR #660)")
# print("=" * 80)
#
# # This should work with the fixed version
# print("\nLoading GAIA via inspect_evals...")
# vanilla_results = eval(
#     gaia(subset="2023_level1", split="validation"),
#     model="openai/gpt-4o-mini",
#     limit=5
# )

# Configure SABRE GAIA endpoint (wraps responses in submit tool call)
os.environ['CUSTOM_API_KEY'] = os.environ['OPENAI_API_KEY']
os.environ['CUSTOM_BASE_URL'] = 'http://localhost:8011/v1'

print("=" * 80)
print("Testing SABRE on GAIA validation set")
print(f"SABRE GAIA endpoint: {os.environ['CUSTOM_BASE_URL']}")
print("=" * 80)

# Route to SABRE - let it handle all orchestration
sabre_results = eval(
  gaia(subset="2023_level1", split="validation"),
  model="openai-api/custom/gpt-4o-mini",
  limit=1  # Start small for testing
)

print(f"\nEvaluation complete!")
print(f"Results: {sabre_results}")
print("Run 'inspect view' to see detailed results")
