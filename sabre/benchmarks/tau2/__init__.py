"""
tau2-bench integration for SABRE.

This module provides integration with tau2-bench, a benchmark for evaluating
conversational agents on multi-turn dialogues with tool use and policy adherence.

Example:
    from sabre.benchmarks.tau2 import SabreTau2Agent, run_bridge

    # Start API bridge
    run_bridge(port=8765)

    # Or use CLI
    # sabre benchmark tau2 --quick
"""

from .agent import SabreTau2Agent
from .api_bridge import run_bridge
from .runner import Tau2Runner

__all__ = ['SabreTau2Agent', 'run_bridge', 'Tau2Runner']
