import argparse
import csv
import json
import os
import subprocess
import sys
import time

from py_no_gil.pi import (
    argv_without_compare_flag,
    parse_execution_time,
)
from py_no_gil.benchmark import environment_details, positive_int

MODULES = ("pi", "primes", "fractal", "nbody")
DEFAULT_WORKER_COUNTS = "1,2,4,8"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="parallel-scale",
        description=(
            "Benchmark a workload at several thread counts. Each point runs the "
            "chosen workload after warm-ups and uses its median execution time."
        ),
        epilog=(
            "Pass workload arguments after --; they are forwarded unchanged, while "
            "parallel-scale supplies --workers, --warmup, and --repeat.\n\n"
            "Text output includes speedup relative to the first worker count and "
            "parallel efficiency (speedup divided by workers). --output-format json "
            "or csv is suitable for plotting or further analysis.\n\n"
            "Examples:\n"
            "  parallel-scale --module primes --workers 1,2,4,8 -- count --start 1 --stop 1000000\n"
            "  parallel-scale --module nbody --compare-gil --output-format csv -- count --particles 500 --steps 50\n\n"
            "With the GIL enabled, CPU-bound pure-Python work normally cannot run "
            "in parallel. Free-threaded speedup varies with workload, hardware, and contention."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--module",
        "-m",
        choices=MODULES,
        default="primes",
        help="Workload module to benchmark; supply its arguments after --.",
    )
    parser.add_argument(
        "--repeat",
        type=positive_int,
        default=3,
        help="Measured runs per worker count; workloads report their median (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=1,
        help="Unmeasured warm-up runs per worker count (default: 1).",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json", "csv"),
        default="text",
        help="text includes a chart; json and csv are machine-readable (default: text).",
    )
    parser.add_argument(
        "--workers",
        default=DEFAULT_WORKER_COUNTS,
        help=f"Comma-separated thread counts; first is the speedup baseline (default: {DEFAULT_WORKER_COUNTS}).",
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


def _run_module(
    module,
    module_args,
    worker_count,
    repeat=3,
    warmup=1,
    env=None,
    runner=subprocess.run,
):
    """Run module as subprocess with --workers <N>, return elapsed seconds."""
    cmd = [
        sys.executable,
        "-m",
        f"py_no_gil.{module}",
        *module_args,
        "--workers",
        str(worker_count),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
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


def sweep(
    module,
    module_args,
    worker_counts,
    gil_on=False,
    repeat=3,
    warmup=1,
    runner=subprocess.run,
):
    """Run module with each worker count, return list of (workers, seconds)."""
    env = _make_gil_env(gil_on)
    results = []
    for wc in worker_counts:
        t = _run_module(
            module,
            module_args,
            wc,
            repeat=repeat,
            warmup=warmup,
            env=env,
            runner=runner,
        )
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
    print(
        f"{'Workers':>7} | {'Time (s)':>10} | {'Speedup':>8} | {'Efficiency':>10} | Chart"
    )
    print(f"{'-' * 7} | {'-' * 10} | {'-' * 8} | {'-' * 10} | {'-' * 48}")
    for wc, t in results:
        speedup = base_time / t if t > 0 else 0
        bar = _format_bar(t, max_time)
        efficiency = speedup / wc if wc else 0
        print(f"{wc:>7} | {t:>10.4f} | {speedup:>7.2f}x | {efficiency:>9.1%} | {bar}")
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
            repeat=args.repeat,
            warmup=args.warmup,
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

    results_gil = None
    if args.compare_gil:
        results_gil = sweep(
            args.module,
            args.module_args,
            worker_counts,
            gil_on=True,
            repeat=args.repeat,
            warmup=args.warmup,
            runner=runner,
        )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "benchmark": "scaling",
                    "module": args.module,
                    "module_args": args.module_args,
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "environment": environment_details(),
                    "gil_disabled": results_no_gil,
                    "gil_enabled": results_gil,
                },
                sort_keys=True,
            )
        )
        return
    if args.output_format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["gil", "workers", "seconds", "speedup", "efficiency"])
        for gil, results in (("disabled", results_no_gil), ("enabled", results_gil)):
            if results is None:
                continue
            base = results[0][1]
            for workers, seconds in results:
                speedup = base / seconds if seconds else 0
                writer.writerow(
                    [
                        gil,
                        workers,
                        f"{seconds:.6f}",
                        f"{speedup:.6f}",
                        f"{speedup / workers:.6f}",
                    ]
                )
        return

    print(f"Scaling sweep for py_no_gil.{args.module}")
    print(f"Module args: {args.module_args if args.module_args else '(defaults)'}")
    print(f"Measured runs per point: {args.repeat}; warm-up runs: {args.warmup}")
    print()
    _print_table(results_no_gil, "--- GIL disabled (PYTHON_GIL=0 / free-threading) ---")
    if results_gil is not None:
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
