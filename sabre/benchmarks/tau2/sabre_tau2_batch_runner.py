"""Batch runner for tau2-bench evaluation with detailed reporting.

This script runs multiple tau2-bench tasks and generates comprehensive reports
including success/failure statistics, detailed per-task results, and summary
metrics across all tasks.

Usage:
    # Run first 10 retail tasks
    python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner \
        --domain retail --num-tasks 10 --output results.json

    # Run specific task range
    python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner \
        --domain retail --start 0 --end 20 --output results.json

    # Run all retail tasks
    python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner \
        --domain retail --all --output results.json
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from sabre.benchmarks.tau2.sabre_tau2_dialogue_runner import run_sabre_dialogue_mode

logger = logging.getLogger(__name__)


async def run_batch_evaluation(
    domain: str = "retail",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    num_tasks: Optional[int] = None,
    model: str = "gpt-4o",
    max_turns: int = 20,
    output_file: Optional[str] = None,
) -> dict:
    """Run batch evaluation on multiple tau2 tasks.

    Args:
        domain: Domain to evaluate (default: retail)
        start_idx: Starting task index (default: 0)
        end_idx: Ending task index (exclusive), None for all tasks
        num_tasks: Number of tasks to run (alternative to end_idx)
        model: Model to use (default: gpt-4o)
        max_turns: Max conversation turns per task
        output_file: Optional output JSON file for detailed results

    Returns:
        Dict with batch results and summary statistics
    """
    # Determine task range
    if num_tasks is not None:
        end_idx = start_idx + num_tasks
    elif end_idx is None:
        # Default to first 10 tasks if nothing specified
        end_idx = start_idx + 10

    task_ids = [str(i) for i in range(start_idx, end_idx)]

    print(f"╔═══════════════════════════════════════════════════════════════╗")
    print(f"║  SABRE x tau2-bench Batch Evaluation                         ║")
    print(f"╠═══════════════════════════════════════════════════════════════╣")
    print(f"║  Domain:       {domain:<48}║")
    print(f"║  Model:        {model:<48}║")
    print(f"║  Task Range:   {start_idx} to {end_idx-1} ({len(task_ids)} tasks){' '*(48-len(f'{start_idx} to {end_idx-1} ({len(task_ids)} tasks)'))}║")
    print(f"║  Max Turns:    {max_turns:<48}║")
    print(f"╚═══════════════════════════════════════════════════════════════╝")
    print()

    # Track results
    results = []
    passed = 0
    failed = 0
    errors = 0
    start_time = time.time()

    # Run each task
    for idx, task_id in enumerate(task_ids, 1):
        print(f"\n{'='*70}")
        print(f"Task {idx}/{len(task_ids)}: {task_id}")
        print(f"{'='*70}")

        try:
            # Run single task evaluation
            result = await run_sabre_dialogue_mode(
                task_id=task_id,
                domain=domain,
                model=model,
                max_turns=max_turns,
            )

            # Track success/failure
            if result.get("tau2_correct"):
                passed += 1
                status = "✓ PASS"
            else:
                failed += 1
                status = "✗ FAIL"

            # Add to results
            results.append(result)

            # Print summary for this task
            print(f"\nStatus: {status}")
            print(f"  tau2 Score:         {result.get('tau2_score', 'N/A')}")
            print(f"  DB Reward:          {result.get('db_reward', 'N/A')}")
            print(f"  COMMUNICATE Reward: {result.get('communicate_reward', 'N/A')}")
            print(f"  Conversation Turns: {result.get('conversation_turns', 'N/A')}")

        except Exception as e:
            errors += 1
            error_result = {
                "task_id": task_id,
                "domain": domain,
                "mode": "dialogue",
                "status": "error",
                "error": str(e),
                "tau2_score": 0.0,
                "tau2_correct": False,
            }
            results.append(error_result)
            print(f"\n✗ ERROR: {e}")
            logger.exception(f"Error running task {task_id}")

    # Calculate summary statistics
    end_time = time.time()
    elapsed = end_time - start_time

    # Compute aggregate metrics
    total_tasks = len(task_ids)
    pass_rate = (passed / total_tasks * 100) if total_tasks > 0 else 0

    avg_score = sum(r.get("tau2_score", 0) for r in results) / total_tasks if total_tasks > 0 else 0
    avg_db = sum(r.get("db_reward", 0) or 0 for r in results if r.get("db_reward") is not None) / max(1, sum(1 for r in results if r.get("db_reward") is not None))
    avg_comm = sum(r.get("communicate_reward", 0) or 0 for r in results if r.get("communicate_reward") is not None) / max(1, sum(1 for r in results if r.get("communicate_reward") is not None))
    avg_turns = sum(r.get("conversation_turns", 0) for r in results) / total_tasks if total_tasks > 0 else 0

    # Build batch result
    batch_result = {
        "metadata": {
            "domain": domain,
            "model": model,
            "task_range": f"{start_idx}-{end_idx-1}",
            "total_tasks": total_tasks,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        },
        "summary": {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": pass_rate,
            "avg_tau2_score": avg_score,
            "avg_db_reward": avg_db,
            "avg_communicate_reward": avg_comm,
            "avg_conversation_turns": avg_turns,
        },
        "task_results": results,
    }

    # Print final summary
    print(f"\n\n{'='*70}")
    print(f"BATCH EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tasks:       {total_tasks}")
    print(f"Passed:            {passed} ({pass_rate:.1f}%)")
    print(f"Failed:            {failed}")
    print(f"Errors:            {errors}")
    print(f"{'='*70}")
    print(f"Average tau2 Score:         {avg_score:.3f}")
    print(f"Average DB Reward:          {avg_db:.3f}")
    print(f"Average COMMUNICATE Reward: {avg_comm:.3f}")
    print(f"Average Conversation Turns: {avg_turns:.1f}")
    print(f"{'='*70}")
    print(f"Elapsed Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}\n")

    # Save detailed results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(batch_result, f, indent=2)

        print(f"✓ Detailed results saved to: {output_file}")

        # Also save a human-readable report
        report_file = output_path.with_suffix(".txt")
        with open(report_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("SABRE x tau2-bench Batch Evaluation Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Domain:       {domain}\n")
            f.write(f"Model:        {model}\n")
            f.write(f"Task Range:   {start_idx} to {end_idx-1} ({total_tasks} tasks)\n")
            f.write(f"Timestamp:    {batch_result['metadata']['timestamp']}\n")
            f.write(f"Elapsed Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)\n\n")

            f.write("=" * 70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Tasks:                {total_tasks}\n")
            f.write(f"Passed:                     {passed} ({pass_rate:.1f}%)\n")
            f.write(f"Failed:                     {failed}\n")
            f.write(f"Errors:                     {errors}\n")
            f.write(f"Average tau2 Score:         {avg_score:.3f}\n")
            f.write(f"Average DB Reward:          {avg_db:.3f}\n")
            f.write(f"Average COMMUNICATE Reward: {avg_comm:.3f}\n")
            f.write(f"Average Conversation Turns: {avg_turns:.1f}\n\n")

            f.write("=" * 70 + "\n")
            f.write("INDIVIDUAL TASK RESULTS\n")
            f.write("=" * 70 + "\n\n")

            for result in results:
                task_id = result.get("task_id", "unknown")
                status = "✓ PASS" if result.get("tau2_correct") else ("✗ ERROR" if result.get("status") == "error" else "✗ FAIL")
                score = result.get("tau2_score", 0.0)
                db_reward = result.get("db_reward", "N/A")
                comm_reward = result.get("communicate_reward", "N/A")
                turns = result.get("conversation_turns", "N/A")

                f.write(f"Task {task_id}: {status}\n")
                f.write(f"  tau2 Score:         {score}\n")
                f.write(f"  DB Reward:          {db_reward}\n")
                f.write(f"  COMMUNICATE Reward: {comm_reward}\n")
                f.write(f"  Conversation Turns: {turns}\n")

                if result.get("status") == "error":
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")

                if result.get("details"):
                    f.write(f"  Details: {result['details']}\n")

                f.write("\n")

        print(f"✓ Human-readable report saved to: {report_file}")

    return batch_result


def main():
    """CLI entry point for batch evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SABRE on multiple tau2-bench tasks (BATCH MODE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run first 10 retail tasks
  python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner --domain retail --num-tasks 10

  # Run tasks 0-19
  python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner --start 0 --end 20

  # Run all retail tasks (114 tasks)
  python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner --all --domain retail

  # Run with output file
  python -m sabre.benchmarks.tau2.sabre_tau2_batch_runner --num-tasks 5 --output results/retail_batch.json
        """
    )

    parser.add_argument("--domain", default="retail", help="Domain (default: retail)")
    parser.add_argument("--model", default="gpt-4o", help="Model (default: gpt-4o)")
    parser.add_argument("--start", type=int, default=0, help="Starting task index (default: 0)")
    parser.add_argument("--end", type=int, help="Ending task index (exclusive)")
    parser.add_argument("--num-tasks", type=int, help="Number of tasks to run (alternative to --end)")
    parser.add_argument("--all", action="store_true", help="Run all tasks in domain")
    parser.add_argument("--max-turns", type=int, default=20, help="Max conversation turns per task")
    parser.add_argument("--output", help="Output JSON file for detailed results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Determine end index
    if args.all:
        # Load tasks file to get count
        import os
        tau2_data_dir = os.environ.get("TAU2_DATA_DIR")
        if not tau2_data_dir:
            print("Error: TAU2_DATA_DIR environment variable not set")
            sys.exit(1)

        tasks_file = Path(tau2_data_dir) / "tau2" / "domains" / args.domain / "tasks.json"
        with open(tasks_file) as f:
            tasks_data = json.load(f)
            total_tasks = len(tasks_data)

        end_idx = total_tasks
        print(f"Running ALL {total_tasks} tasks in {args.domain} domain")
    else:
        end_idx = args.end

    # Setup event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    import nest_asyncio
    nest_asyncio.apply(loop)

    # Run batch evaluation
    result = loop.run_until_complete(
        run_batch_evaluation(
            domain=args.domain,
            start_idx=args.start,
            end_idx=end_idx,
            num_tasks=args.num_tasks,
            model=args.model,
            max_turns=args.max_turns,
            output_file=args.output,
        )
    )

    # Exit with appropriate code
    if result["summary"]["errors"] > 0 or result["summary"]["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
