"""
Script to run tau2-bench evaluation using SABRE agent.

This script replaces tau2-bench's default LLMAgent with SabreAgent.

Usage:
    # Run quick evaluation (2-3 tasks per domain)
    python run_tau2_with_sabre.py --quick

    # Run full evaluation
    python run_tau2_with_sabre.py

    # Run specific domain
    python run_tau2_with_sabre.py --domain retail

    # Run with specific model
    python run_tau2_with_sabre.py --model gpt-4o-mini --quick
"""

import argparse
import os
import sys
from pathlib import Path

# Add tau2-bench to path
tau2_bench_path = Path(__file__).parent.parent.parent.parent.parent / "tau2-bench" / "src"
if tau2_bench_path.exists():
    sys.path.insert(0, str(tau2_bench_path))
else:
    print(f"Warning: tau2-bench not found at {tau2_bench_path}")
    print("Please set TAU2_BENCH_PATH environment variable or install tau2-bench")

from loguru import logger

# tau2-bench imports
from tau2.benchmark import benchmark
from tau2.data import load_tasks_from_domain

# SABRE imports
from sabre.benchmarks.tau2.sabre_agent import SabreAgent


def run_tau2_benchmark(
    domains: list[str] = None,
    quick: bool = False,
    model: str = None,
    max_turns: int = 10,
    output_dir: str = None
):
    """
    Run tau2-bench evaluation with SABRE agent.

    Args:
        domains: List of domains to evaluate (default: all)
        quick: Run quick evaluation (few tasks per domain)
        model: Model to use (default: from env or "gpt-4o")
        max_turns: Maximum conversation turns
        output_dir: Directory to save results (default: tau2-bench data dir)
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.getenv("TAU2_DATA_DIR", "/tmp/tau2-bench/data")

    output_path = Path(output_dir) / "simulations"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running tau2-bench with SABRE agent")
    logger.info(f"Model: {model or os.getenv('OPENAI_MODEL', 'gpt-4o')}")
    logger.info(f"Quick mode: {quick}")
    logger.info(f"Max turns: {max_turns}")
    logger.info(f"Output dir: {output_path}")

    # Determine domains
    if domains is None:
        domains = ["retail", "airline", "banking"]  # Default domains

    # Run evaluation for each domain
    all_results = []

    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating domain: {domain}")
        logger.info(f"{'='*60}\n")

        # Load tasks
        try:
            tasks = load_tasks_from_domain(
                domain=domain,
                num_tasks=3 if quick else None  # Quick mode: 3 tasks per domain
            )
        except Exception as e:
            logger.error(f"Failed to load tasks for domain {domain}: {e}")
            continue

        logger.info(f"Loaded {len(tasks)} tasks for domain {domain}")

        # Run tasks
        for i, task in enumerate(tasks, 1):
            logger.info(f"\nTask {i}/{len(tasks)}: {task.task_id}")
            logger.info(f"Description: {task.description[:100]}...")

            # Create SABRE agent for this task
            agent = SabreAgent(
                tools=task.tools,
                domain_policy=task.domain_policy,
                model=model
            )

            # Run benchmark
            try:
                result = benchmark(
                    agent=agent,
                    task=task,
                    max_turns=max_turns,
                    output_dir=str(output_path)
                )

                all_results.append(result)

                # Log results
                logger.info(f"Task {i} completed:")
                logger.info(f"  Reward: {result.reward:.2f}")
                logger.info(f"  Actions: {result.actions_completed}/{result.required_actions}")
                logger.info(f"  Turns: {result.num_turns}")

            except Exception as e:
                logger.error(f"Task {i} failed: {e}", exc_info=True)
                continue

    # Summary
    if all_results:
        avg_reward = sum(r.reward for r in all_results) / len(all_results)
        total_actions = sum(r.actions_completed for r in all_results)
        required_actions = sum(r.required_actions for r in all_results)

        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total tasks: {len(all_results)}")
        logger.info(f"Average reward: {avg_reward:.2f}")
        logger.info(f"Actions completed: {total_actions}/{required_actions} ({100*total_actions/required_actions:.1f}%)")
        logger.info(f"Results saved to: {output_path}")

        return all_results
    else:
        logger.error("No results generated")
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tau2-bench evaluation with SABRE agent"
    )

    parser.add_argument(
        "--domain",
        type=str,
        action="append",
        help="Domain to evaluate (can be specified multiple times). Default: all domains"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation (2-3 tasks per domain)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: from OPENAI_MODEL env or gpt-4o)"
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns (default: 10)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: TAU2_DATA_DIR/simulations)"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Run evaluation
    try:
        results = run_tau2_benchmark(
            domains=args.domain,
            quick=args.quick,
            model=args.model,
            max_turns=args.max_turns,
            output_dir=args.output_dir
        )

        # Exit with appropriate code
        if results:
            avg_reward = sum(r.reward for r in results) / len(results)
            sys.exit(0 if avg_reward > 0.5 else 1)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
