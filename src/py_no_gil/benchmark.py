"""Shared benchmark reporting for the demonstration commands."""

import io
import json
import statistics
import sys
import sysconfig
import time
import argparse
from contextlib import redirect_stdout


def positive_int(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def add_benchmark_arguments(parser):
    parser.add_argument(
        "--repeat",
        type=positive_int,
        default=3,
        help="Measured runs; median, minimum, and maximum are reported (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=1,
        help="Unmeasured warm-up runs before timing (default: 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON benchmark record instead of human-readable output.",
    )


def run_benchmark(workload, *, repeat, warmup, quiet=False):
    """Run a callable, returning its final result and robust timing statistics."""
    sink = io.StringIO()
    for _ in range(warmup):
        with redirect_stdout(sink) if quiet else _null_context():
            workload()

    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        with redirect_stdout(sink) if quiet else _null_context():
            result = workload()
        samples.append(time.perf_counter() - start)
    return result, {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def environment_details():
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "free_threading_build": sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
        "gil_enabled": gil_enabled,
    }


def print_benchmark(*, name, workers, parameters, timing, result, output_json):
    record = {
        "benchmark": name,
        "workers": workers,
        "parameters": parameters,
        "timing": timing,
        "result": result,
        "environment": environment_details(),
    }
    if output_json:
        print(json.dumps(record, default=str, sort_keys=True))
        return
    environment = record["environment"]
    print(f"Python interpreter: {environment['python_version']}")
    print(f"Free-threading build: {environment['free_threading_build']}")
    print(f"GIL enabled: {environment['gil_enabled']}")
    print(
        f"Workers: {workers}; warm-up runs: {parameters['warmup']}; measured runs: {parameters['repeat']}"
    )
    print(
        f"Execution time: {timing['median_seconds']:.4f} seconds (median; min {timing['min_seconds']:.4f}, max {timing['max_seconds']:.4f})"
    )
