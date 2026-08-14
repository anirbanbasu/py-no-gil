import argparse
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from py_no_gil.pi import compare_gil, default_worker_count, split_work

DEFAULT_START = 1
DEFAULT_STOP = 1_000_000


def is_prime(value):
    if value < 2:
        return False
    if value < 4:
        return True
    if value % 2 == 0:
        return False
    limit = math.isqrt(value)
    divisor = 3
    while divisor <= limit:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def count_primes_in_range(start, stop):
    return sum(1 for value in range(start, stop) if is_prime(value))


def list_primes_in_range(start, stop):
    return [value for value in range(start, stop) if is_prime(value)]


def search_range_worker(sub_start, sub_stop):
    return list_primes_in_range(sub_start, sub_stop)


def search_primes_parallel(worker_count, start, stop, mode="count"):
    total = stop - start
    ranges = list(split_work(total, worker_count))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                search_range_worker, start + chunk_start, start + chunk_stop
            )
            for chunk_start, chunk_stop in ranges
            if chunk_stop > chunk_start
        ]

    partial_results = [future.result() for future in futures]

    if mode == "count":
        return sum(len(result) for result in partial_results)
    if mode == "list":
        return [prime for result in partial_results for prime in result]
    raise ValueError(f"Unknown mode: {mode}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Search for primes in a range using multiple CPU cores."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    count_parser = subparsers.add_parser("count", help="Count primes in the range.")
    list_parser = subparsers.add_parser("list", help="List primes in the range.")

    for mode_parser in (count_parser, list_parser):
        mode_parser.add_argument(
            "--start",
            type=int,
            default=DEFAULT_START,
            help="Start of the range (inclusive).",
        )
        mode_parser.add_argument(
            "--stop",
            type=int,
            default=DEFAULT_STOP,
            help="Stop of the range (exclusive).",
        )
        mode_parser.add_argument(
            "--workers",
            type=int,
            default=default_worker_count(),
            help=f"Number of worker threads to run. Available CPU cores: {default_worker_count()}.",
        )
        mode_parser.add_argument(
            "--compare-gil",
            action="store_true",
            help=(
                "Run the same workload with PYTHON_GIL=1 and PYTHON_GIL=0, "
                "then print a timing comparison."
            ),
        )

    return parser.parse_args(argv)


def run_parallel_primes(argv=None, runner=subprocess.run):
    args = parse_args(argv)
    if args.compare_gil:
        child_argv = sys.argv[1:] if argv is None else argv
        print(compare_gil(child_argv, runner=runner, module="py_no_gil.primes"))
        return

    worker_count = max(2, args.workers) if default_worker_count() > 1 else 1

    print(f"Detected {os.cpu_count()} cores.")
    print(
        f"Global Interpreter Lock (GIL) is {'enabled. Disable it using PYTHON_GIL=0' if sys._is_gil_enabled() else 'disabled. Enable it using PYTHON_GIL=1'}."
    )
    print(f"Python interpreter: {sys.version}")
    print(
        f"Searching for primes in [{args.start}, {args.stop}) across {worker_count} workers."
    )

    start_time = time.perf_counter()

    if args.mode == "count":
        result = search_primes_parallel(
            worker_count, args.start, args.stop, mode="count"
        )
        print(f"Prime count: {result}")
    else:
        result = search_primes_parallel(
            worker_count, args.start, args.stop, mode="list"
        )
        print(f"Primes: {', '.join(str(p) for p in result)}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    run_parallel_primes()
