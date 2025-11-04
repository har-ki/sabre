"""
Runner for tau2-bench evaluations with SABRE.

This provides CLI commands and utilities to run tau2-bench evaluations
and analyze results.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List
import json
import click


class Tau2Runner:
    """
    Manages tau2-bench evaluations for SABRE.

    Usage:
        runner = Tau2Runner()
        results = runner.run_evaluation(
            domain='retail',
            num_trials=5,
            num_tasks=10
        )
    """

    def __init__(
        self,
        tau2_path: Optional[Path] = None,
        sabre_port: int = 8011,
        bridge_port: int = 8765
    ):
        """
        Initialize tau2-bench runner.

        Args:
            tau2_path: Path to tau2-bench installation
            sabre_port: Port where SABRE server runs
            bridge_port: Port for API bridge
        """
        self.tau2_path = tau2_path or self._find_tau2_bench()
        self.sabre_port = sabre_port
        self.bridge_port = bridge_port

    def _find_tau2_bench(self) -> Optional[Path]:
        """Locate tau2-bench installation."""
        # Check if tau2 command is available
        try:
            result = subprocess.run(
                ['which', 'tau2'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                tau2_bin = Path(result.stdout.strip())
                return tau2_bin.parent.parent
        except:
            pass

        # Check common locations
        for location in [
            Path.home() / 'tau2-bench',
            Path.cwd() / 'tau2-bench',
            Path('/opt/tau2-bench')
        ]:
            if location.exists() and (location / 'data').exists():
                return location

        return None

    def check_tau2_installed(self) -> bool:
        """Check if tau2-bench is installed and accessible."""
        try:
            import importlib.util
            # Check if tau2 package is installed (package name is "tau2" not "tau2bench")
            spec = importlib.util.find_spec("tau2")
            if spec is not None:
                return True
            # Also try checking for tau2 CLI command as fallback
            result = subprocess.run(
                ['tau2', '--version'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except (FileNotFoundError, ImportError):
            return False

    def run_evaluation(
        self,
        domain: str = 'retail',
        num_trials: int = 5,
        num_tasks: Optional[int] = None,
        task_ids: Optional[List[int]] = None,
        max_concurrency: int = 5,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Run tau2-bench evaluation with SABRE.

        Args:
            domain: Domain to evaluate (retail, airline, telecom)
            num_trials: Number of trials per task
            num_tasks: Limit number of tasks (None = all)
            task_ids: Specific task IDs to run
            max_concurrency: Maximum concurrent evaluations
            output_dir: Where to save results

        Returns:
            Path to results directory
        """
        if not self.check_tau2_installed():
            raise RuntimeError(
                "tau2-bench not found. Install with:\n"
                "  pip install git+https://github.com/sierra-research/tau2-bench"
            )

        # Build tau2 command
        # Note: tau2-bench uses LiteLLM internally, so we configure via model name
        # LiteLLM will use OPENAI_API_BASE env var for openai/ prefixed models
        cmd = [
            'tau2', 'run',
            '--domain', domain,
            '--agent-llm', 'openai/sabre',  # Use openai/ prefix for LiteLLM routing
            '--user-llm', 'gpt-4o',
            '--num-trials', str(num_trials),
            '--max-concurrency', str(max_concurrency)
        ]

        if num_tasks:
            cmd.extend(['--num-tasks', str(num_tasks)])

        if task_ids:
            cmd.extend(['--task-ids'] + [str(tid) for tid in task_ids])

        if output_dir:
            cmd.extend(['--output-dir', str(output_dir)])

        print(f"Running tau2-bench evaluation...")
        print(f"Domain: {domain}")
        print(f"Trials: {num_trials}")
        if num_tasks:
            print(f"Tasks: {num_tasks}")
        print(f"\nCommand: {' '.join(cmd)}\n")

        # Run evaluation
        try:
            # Set environment variables for LiteLLM to use our API bridge
            env = os.environ.copy()
            env['OPENAI_API_BASE'] = f'http://localhost:{self.bridge_port}/v1'

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Evaluation failed with code {process.returncode}")

        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user")
            process.terminate()
            raise

        # Return results path
        if output_dir:
            results_dir = output_dir
        else:
            # Default tau2-bench location
            results_dir = Path('data/tau2/simulations')

        return results_dir

    def analyze_results(self, results_path: Path) -> dict:
        """
        Analyze tau2-bench results.

        Args:
            results_path: Path to results JSON file

        Returns:
            Dictionary with analysis metrics
        """
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found: {results_path}")

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Calculate basic metrics
        analysis = {
            'total_tasks': len(results),
            'successful': sum(1 for r in results if r.get('success', False)),
            'failed': sum(1 for r in results if not r.get('success', False)),
        }

        if analysis['total_tasks'] > 0:
            analysis['success_rate'] = analysis['successful'] / analysis['total_tasks']

        return analysis


# CLI commands

@click.group()
def cli():
    """SABRE tau2-bench evaluation tools"""
    pass


@cli.command()
@click.option('--domain', default='retail',
              type=click.Choice(['retail', 'airline', 'telecom']),
              help='Domain to evaluate')
@click.option('--trials', default=5, help='Number of trials per task')
@click.option('--tasks', default=None, type=int, help='Number of tasks to run (default: all)')
@click.option('--task-ids', default=None, help='Specific task IDs (comma-separated)')
@click.option('--quick', is_flag=True, help='Quick test (5 tasks, 1 trial)')
@click.option('--bridge-port', default=8765, help='API bridge port')
def run(domain, trials, tasks, task_ids, quick, bridge_port):
    """Run tau2-bench evaluation with SABRE"""
    if quick:
        trials = 1
        tasks = 5
        click.echo("Running quick evaluation (5 tasks, 1 trial)")

    task_id_list = None
    if task_ids:
        task_id_list = [int(tid.strip()) for tid in task_ids.split(',')]

    runner = Tau2Runner(bridge_port=bridge_port)

    try:
        results = runner.run_evaluation(
            domain=domain,
            num_trials=trials,
            num_tasks=tasks,
            task_ids=task_id_list
        )

        click.echo(f"\n✓ Evaluation complete!")
        click.echo(f"✓ Results saved to: {results}")

    except RuntimeError as e:
        click.echo(f"\n✗ Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n\nEvaluation cancelled")
        sys.exit(130)


@cli.command()
@click.argument('results_path', type=click.Path(exists=True))
def analyze(results_path):
    """Analyze tau2-bench results"""
    runner = Tau2Runner()

    try:
        analysis = runner.analyze_results(Path(results_path))

        click.echo("\n=== Results Analysis ===")
        click.echo(f"Total tasks: {analysis['total_tasks']}")
        click.echo(f"Successful: {analysis['successful']}")
        click.echo(f"Failed: {analysis['failed']}")
        if 'success_rate' in analysis:
            click.echo(f"Success rate: {analysis['success_rate']:.1%}")

    except Exception as e:
        click.echo(f"Error analyzing results: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
