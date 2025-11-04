"""
SABRE CLI Entry Point.

Starts the server in background and launches the client.
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from sabre.common.paths import get_logs_dir, get_pid_file, ensure_dirs, migrate_from_old_structure, cleanup_all


def start_server():
    """Start the SABRE server in background"""
    # Migrate from old structure if needed
    migrate_from_old_structure()

    # Ensure all directories exist
    ensure_dirs()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try to read from ~/.openai/key
        key_file = Path.home() / ".openai" / "key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
        else:
            print("Error: OPENAI_API_KEY not set and ~/.openai/key not found", file=sys.stderr)
            sys.exit(1)

    # Get optional configuration from environment
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")
    port = os.getenv("PORT", "8011")

    # Get logs directory
    log_dir = get_logs_dir()
    server_log = log_dir / "server.log"

    # Start server process
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key

    # Pass optional config to server
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    if model:
        env["OPENAI_MODEL"] = model
    env["PORT"] = port

    # Open log file (keep it open for subprocess)
    print("Starting server...")
    log_f = open(server_log, "w")

    try:
        server_process = subprocess.Popen(
            [sys.executable, "-m", "sabre.server"],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Proper process management
        )

        # Write PID file
        pid_file = get_pid_file()
        pid_file.write_text(str(server_process.pid))

        # Wait for server to start (give it more time for initialization)
        time.sleep(3)

        # Check if process crashed
        if server_process.poll() is not None:
            print("Error: Server failed to start", file=sys.stderr)
            print(f"Check {server_log} for details", file=sys.stderr)
            log_f.close()
            pid_file.unlink(missing_ok=True)
            sys.exit(1)

        # Check health endpoint
        import requests

        max_retries = 40  # Increased to 20 seconds (40 * 0.5s)
        for i in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    print(f"Server started (PID: {server_process.pid}, port: {port})")
                    print(f"Server logs: {server_log}")
                    return server_process
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass

            # Check if process died
            if server_process.poll() is not None:
                print("Error: Server process died during startup", file=sys.stderr)
                print(f"Check {server_log} for details", file=sys.stderr)
                log_f.close()
                pid_file.unlink(missing_ok=True)
                sys.exit(1)

            time.sleep(0.5)

        # Timeout
        print("Error: Server did not become healthy within timeout", file=sys.stderr)
        print(f"Check {server_log} for details", file=sys.stderr)
        server_process.terminate()
        log_f.close()
        pid_file.unlink(missing_ok=True)
        sys.exit(1)

    except Exception as e:
        print(f"Error: Failed to start server: {e}", file=sys.stderr)
        log_f.close()
        sys.exit(1)


def stop_server():
    """Stop the SABRE server using PID file or process search"""
    pid_file = get_pid_file()
    pid = None

    # Try PID file first
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            print("Invalid PID in file", file=sys.stderr)
            pid_file.unlink(missing_ok=True)

    # If no PID from file, search for running server process
    if pid is None:
        try:
            result = subprocess.run(["pgrep", "-f", "sabre.server"], capture_output=True, text=True)
            if result.returncode == 0:
                pids = [int(p) for p in result.stdout.strip().split("\n") if p]
                if pids:
                    pid = pids[0]  # Take first match
                    print(f"Found server process via search (PID: {pid})")
        except Exception as e:
            print(f"Could not search for server process: {e}", file=sys.stderr)

    if pid is None:
        print("No server PID file found and no running server process detected.", file=sys.stderr)
        return 1

    # Try to kill the process
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopping server (PID: {pid})...")

        # Wait up to 5 seconds for graceful shutdown
        for _ in range(50):
            try:
                os.kill(pid, 0)  # Check if process exists
                time.sleep(0.1)
            except ProcessLookupError:
                # Process is gone
                break
        else:
            # Force kill if still running
            print("Server didn't stop gracefully, force killing...")
            os.kill(pid, signal.SIGKILL)

        print("Server stopped")
        pid_file.unlink(missing_ok=True)
        return 0

    except ProcessLookupError:
        print(f"Server process {pid} not found. Cleaning up PID file.", file=sys.stderr)
        pid_file.unlink(missing_ok=True)
        return 1
    except PermissionError:
        print(f"Permission denied when trying to kill process {pid}", file=sys.stderr)
        return 1


def cleanup(force: bool = False):
    """Clean up all SABRE XDG directories"""

    # Get cleanup info (without actually removing)
    result = cleanup_all(force=False)

    if not result["directories"]:
        print("No SABRE directories found to clean up.")
        return 0

    # Show what will be removed
    print("The following directories will be removed:")
    print()

    def format_size(size_bytes):
        """Format size in human-readable form"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    for directory in result["directories"]:
        size = result["sizes"][directory]
        print(f"  {directory} ({format_size(size)})")

    print()
    print(f"Total size: {format_size(result['total_size'])}")
    print()

    # Ask for confirmation unless force is True
    if not force:
        try:
            response = input("Are you sure you want to delete these directories? [y/N]: ")
            if response.lower() not in ("y", "yes"):
                print("Cleanup cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nCleanup cancelled.")
            return 0

    # Perform cleanup
    try:
        cleanup_all(force=True)
        print("Cleanup complete!")
        return 0
    except Exception as e:
        print(f"Error during cleanup: {e}", file=sys.stderr)
        return 1


async def run_client():
    """Run the client"""
    from sabre.client.client import main

    return await main()


def run_benchmark(args):
    """Run benchmark evaluation"""
    from sabre.benchmarks.tau2 import api_bridge, runner as tau2_runner
    import multiprocessing
    import httpx

    print("Starting SABRE tau2-bench evaluation...")

    # Check for OpenAI API key (required by API bridge)
    if not os.getenv("OPENAI_API_KEY"):
        print("\n✗ OPENAI_API_KEY environment variable not set", file=sys.stderr)
        print("Set it with: export OPENAI_API_KEY=your_key", file=sys.stderr)
        return 1

    # Check if SABRE server is running
    port = os.getenv("PORT", "8011")
    try:
        response = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
        if response.status_code != 200:
            raise Exception("SABRE server not responding")
    except:
        print(f"\n✗ SABRE server not running on port {port}", file=sys.stderr)
        print(f"Start it with: OPENAI_API_KEY=your_key uv run sabre-server", file=sys.stderr)
        return 1

    print(f"✓ SABRE server is running on port {port}")

    # Start API bridge in background process using subprocess for better env handling
    bridge_port = args.bridge_port if hasattr(args, 'bridge_port') else 8765
    print(f"Starting API bridge on port {bridge_port}...")

    # Use subprocess instead of multiprocessing for better environment variable handling
    import subprocess as sp
    bridge_process = sp.Popen(
        [sys.executable, "-m", "sabre.benchmarks.tau2.api_bridge_main", str(bridge_port), str(port)],
        env=os.environ.copy(),
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        text=True
    )
    time.sleep(3)  # Wait for bridge to start

    try:
        # Check bridge health
        try:
            response = httpx.get(f"http://localhost:{bridge_port}/health", timeout=2.0)
            if response.status_code == 200:
                print(f"✓ API bridge is ready\n")
        except:
            print("✗ Failed to start API bridge", file=sys.stderr)
            bridge_process.terminate()
            return 1

        # Run evaluation
        domain = args.domain if hasattr(args, 'domain') else 'retail'
        trials = 1 if args.quick else (args.trials if hasattr(args, 'trials') else 5)
        tasks = 5 if args.quick else (args.tasks if hasattr(args, 'tasks') else None)

        task_id_list = None
        if hasattr(args, 'task_ids') and args.task_ids:
            task_id_list = [int(tid.strip()) for tid in args.task_ids.split(',')]

        runner = tau2_runner.Tau2Runner(
            bridge_port=bridge_port,
            sabre_port=int(port)
        )

        results = runner.run_evaluation(
            domain=domain,
            num_trials=trials,
            num_tasks=tasks,
            task_ids=task_id_list
        )

        print(f"\n✓ Evaluation complete!")
        print(f"✓ Results saved to: {results}")
        return 0

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1
    finally:
        # Clean up bridge process
        bridge_process.terminate()
        try:
            bridge_process.wait(timeout=2)
        except sp.TimeoutExpired:
            bridge_process.kill()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SABRE CLI")

    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_subparsers = benchmark_parser.add_subparsers(dest='benchmark_type', help='Benchmark type')

    # tau2 benchmark
    tau2_parser = benchmark_subparsers.add_parser('tau2', help='Run tau2-bench evaluation')
    tau2_parser.add_argument('--domain', choices=['retail', 'airline', 'telecom'],
                            default='retail', help='Domain to evaluate')
    tau2_parser.add_argument('--trials', type=int, default=5,
                            help='Number of trials per task')
    tau2_parser.add_argument('--tasks', type=int,
                            help='Number of tasks to run (default: all)')
    tau2_parser.add_argument('--task-ids',
                            help='Specific task IDs (comma-separated)')
    tau2_parser.add_argument('--quick', action='store_true',
                            help='Quick test (5 tasks, 1 trial)')
    tau2_parser.add_argument('--bridge-port', type=int, default=8765,
                            help='API bridge port')

    # Original flags (top level for backwards compatibility)
    parser.add_argument("--stop", action="store_true", help="Stop the running server")
    parser.add_argument("--clean", action="store_true", help="Clean up all SABRE XDG directories")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt (for --clean)")

    args = parser.parse_args()

    # Handle benchmark command
    if args.command == 'benchmark':
        if args.benchmark_type == 'tau2':
            return run_benchmark(args)
        else:
            print("Error: Please specify a benchmark type (e.g., tau2)", file=sys.stderr)
            return 1

    # Handle --clean flag
    if args.clean:
        return cleanup(force=args.force)

    # Handle --stop flag
    if args.stop:
        return stop_server()

    # Normal operation: start server and run client
    server_process = None

    try:
        # Start server
        server_process = start_server()

        # Run client
        exit_code = asyncio.run(run_client())

        return exit_code

    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0

    finally:
        # Cleanup: kill server and remove PID file
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

            # Remove PID file
            get_pid_file().unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
