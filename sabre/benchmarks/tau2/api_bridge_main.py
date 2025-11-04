"""
Main entry point for running API bridge as a subprocess.
"""

import sys
from sabre.benchmarks.tau2.api_bridge import run_bridge

if __name__ == "__main__":
    bridge_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    sabre_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8011
    run_bridge(port=bridge_port, sabre_port=sabre_port)
