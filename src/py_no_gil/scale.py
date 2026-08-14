import argparse
import os
import subprocess
import sys
import time

from py_no_gil.pi import (
    argv_without_compare_flag,
    parse_execution_time,
)

MODULES = ("pi", "primes", "fractal", "nbody")
DEFAULT_WORKER_COUNTS = "1,2,4,8"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="parallel-scale",
        description=(
            "Sweep thread counts for a workload module and print a scaling curve. "
            "With the GIL, pure-Python CPU work shows a flat line; without the GIL, "
            "time drops until you run out of cores."
        ),
    )
    parser.add_argument(
        "--module",
        "-m",
        choices=MODULES,
        default="primes",
        help="The workload module to benchmark.",
    )
    parser.add_argument(
        "--workers",
        default=DEFAULT_WORKER_COUNTS,
        help=f"Comma-separated list of thread counts to sweep (default: {DEFAULT_WORKER_COUNTS}).",
    )
    parser.add_argument(
        "--compare-gil",
        action="store_true",
        help=(
            "Also run each configuration with PYTHON_GIL=1 (GIL enabled), "
            "so you can compare the scaling curve with and without the GIL."
        ),
    )

    args, remaining = parser.parse_known_args(argv)
    args.module_args = [a for a in argv_without_compare_flag(remaining) if a != "--"]
    return args


def _run_module(module, module_args, worker_count, env=None, runner=subprocess.run):
    """Run module as subprocess with --workers <N>, return elapsed seconds."""
    cmd = [
        sys.executable,
        "-m",
        f"py_no_gil.{module}",
        *module_args,
        "--workers",
        str(worker_count),
    ]
    start = time.perf_counter()
    result = runner(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        detail = f"{result.stdout or ''}{result.stderr or ''}".strip()
        raise RuntimeError(
            f"Child process failed (exit {result.returncode}).\n{detail}"
        )
    # Parse execution time from module output for a cleaner number
    try:
        exec_time = parse_execution_time(result.stdout)
    except ValueError:
        exec_time = elapsed
    return exec_time


def _make_gil_env(gil_on):
    env = dict(os.environ)
    env["PYTHON_GIL"] = "1" if gil_on else "0"
    return env


def sweep(module, module_args, worker_counts, gil_on=False, runner=subprocess.run):
    """Run module with each worker count, return list of (workers, seconds)."""
    env = _make_gil_env(gil_on)
    results = []
    for wc in worker_counts:
        t = _run_module(module, module_args, wc, env=env, runner=runner)
        results.append((wc, t))
    return results


def _format_bar(time_seconds, max_time, width=40):
    """Return a string bar proportional to time (longer bar = more time)."""
    if max_time == 0:
        return ""
    filled = int(width * time_seconds / max_time)
    return "■" * filled + "□" * (width - filled)


def _print_table(results, title):
    print(title)
    print()
    max_time = max(t for _, t in results)
    base_time = results[0][1]
    print(f"{'Workers':>7} | {'Time (s)':>10} | {'Speedup':>8} | Chart")
    print(f"{'-' * 7} | {'-' * 10} | {'-' * 8} | {'-' * 48}")
    for wc, t in results:
        speedup = base_time / t if t > 0 else 0
        bar = _format_bar(t, max_time)
        print(f"{wc:>7} | {t:>10.4f} | {speedup:>7.2f}x | {bar}")
    print()


MODULE_USAGE = {
    "pi": "monte-carlo --samples 16_777_216",
    "primes": "count --start 1 --stop 1000000",
    "fractal": "mandelbrot --width 400 --height 300",
    "nbody": "count --particles 500 --steps 50",
}


class ModuleError(Exception):
    """Raised when a module invocation fails."""


def run_parallel_scale(argv=None, runner=subprocess.run):
    args = parse_args(argv)

    worker_counts = [int(w) for w in args.workers.split(",")]

    try:
        results_no_gil = sweep(
            args.module,
            args.module_args,
            worker_counts,
            gil_on=False,
            runner=runner,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(file=sys.stderr)
        print("Compatible modules:", file=sys.stderr)
        for mod in MODULES:
            print(f"  {mod}:  parallel-scale --module {mod} -- <args>", file=sys.stderr)
            print(
                f"       e.g. uv run parallel-scale --module {mod} -- {MODULE_USAGE[mod]}",
                file=sys.stderr,
            )
        print(file=sys.stderr)
        print(
            "Add --compare-gil to also benchmark with the GIL enabled.", file=sys.stderr
        )
        raise SystemExit(1)

    print(f"Scaling sweep for py_no_gil.{args.module}")
    print(f"Module args: {args.module_args if args.module_args else '(defaults)'}")
    print()

    _print_table(
        results_no_gil,
        "--- GIL disabled (PYTHON_GIL=0 / free-threading) ---",
    )

    if args.compare_gil:
        results_gil = sweep(
            args.module, args.module_args, worker_counts, gil_on=True, runner=runner
        )
        _print_table(results_gil, "--- GIL enabled (PYTHON_GIL=1) ---")

        for i, (wc, t) in enumerate(results_no_gil):
            print(
                f"  Workers={wc}: GIL-off {t:.4f}s  "
                f"GIL-on {results_gil[i][1]:.4f}s  "
                f"speedup vs GIL-on: {results_gil[i][1] / t:.2f}x"
            )
        print()


if __name__ == "__main__":
    run_parallel_scale()
