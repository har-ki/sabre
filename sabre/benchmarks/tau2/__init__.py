"""tau2-bench integration for SABRE.

This module provides integration with tau2-bench using SABRE's native MCP infrastructure.
The main entry point is sabre_tau2_runner.py which uses MCPClientManager and MCPHelperAdapter.
"""

from .sabre_tau2_runner import run_sabre_on_tau2_task, run_batch_evaluation

__all__ = ["run_sabre_on_tau2_task", "run_batch_evaluation"]
