import argparse
import math
import os
import sys
import random
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext

DEFAULT_TOTAL_SAMPLES = 16_777_216
DEFAULT_SERIES_TERMS = 32
DEFAULT_PRECISION = 64
DEFAULT_METHOD = "monte-carlo"
CHUDNOVSKY_BASE = -262537412640768000


def default_worker_count():
    available_cores = os.cpu_count() or 1
    if available_cores > 1:
        return available_cores
    return 1


def split_work(total_items, worker_count):
    base_size, remainder = divmod(total_items, worker_count)
    start_index = 0

    for worker_index in range(worker_count):
        stop_index = start_index + base_size + (1 if worker_index < remainder else 0)
        yield start_index, stop_index
        start_index = stop_index


def monte_carlo_worker(sample_count, seed):
    local_random = random.Random(seed)
    random_value = local_random.random
    hit_count = 0

    for _ in range(sample_count):
        x_coord = random_value()
        y_coord = random_value()
        if x_coord * x_coord + y_coord * y_coord <= 1.0:
            hit_count += 1

    return hit_count


def monte_carlo_pi_parallel(total_samples, worker_count):
    """
    Estimate π using the Monte Carlo method in parallel across multiple worker threads.
    The total number of samples is divided among the workers, and each worker independently
    generates random points and counts how many fall inside the unit circle.
    The results are then aggregated to produce the final estimate of π.

    Args:
        total_samples: Total number of random samples to generate across all workers.
        worker_count: Number of worker threads to use for the computation.

    Returns:
        An estimate of π based on the Monte Carlo method.
    """
    print(
        f"Running Monte Carlo estimation with {total_samples} samples across {worker_count} workers."
    )
    samples_per_worker = list(split_work(total_samples, worker_count))
    seeds = [random.SystemRandom().randrange(1 << 63) for _ in range(worker_count)]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(monte_carlo_worker, stop - start, seed)
            for (start, stop), seed in zip(samples_per_worker, seeds, strict=True)
            if stop > start
        ]

    hit_count = sum(future.result() for future in futures)
    return 4.0 * hit_count / total_samples


def machin_arctan_chunk(inverse, start_index, stop_index, precision):
    with localcontext() as context:
        context.prec = precision
        inverse_decimal = Decimal(inverse)
        x_value = Decimal(1) / inverse_decimal
        x_squared = x_value * x_value
        power = x_value ** (2 * start_index + 1)
        sign = Decimal(-1) if start_index % 2 else Decimal(1)
        partial_sum = Decimal(0)

        for term_index in range(start_index, stop_index):
            partial_sum += sign * (power / Decimal(2 * term_index + 1))
            power *= x_squared
            sign = -sign

        return partial_sum


def machin_pi_parallel(term_count, worker_count, precision):
    """
    Compute π using a Machin-like arctan identity in parallel across multiple worker threads.
    The arctan series for the specified inverse values (5 and 239) are computed
    in parallel, with the total number of terms divided among the workers. The results
    are then combined to produce the final estimate of π.

    Args:
        term_count: Total number of terms to compute in each arctan series.
        worker_count: Number of worker threads to use for the computation.
        precision: The decimal precision to use for the calculations.

    Returns:
        An estimate of π based on the Machin-like arctan identity.
    """
    print(
        f"Computing π using Machin-like arctan identity with {term_count} terms per series across {worker_count} workers at precision {precision}."
    )
    ranges = list(split_work(term_count, worker_count))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        arctan_futures = []
        for start_index, stop_index in ranges:
            if stop_index <= start_index:
                continue
            arctan_futures.append(
                executor.submit(
                    machin_arctan_chunk, 5, start_index, stop_index, precision
                )
            )
            arctan_futures.append(
                executor.submit(
                    machin_arctan_chunk, 239, start_index, stop_index, precision
                )
            )

    with localcontext() as context:
        context.prec = precision
        arctan_five = sum(future.result() for future in arctan_futures[0::2])
        arctan_two_thirty_nine = sum(future.result() for future in arctan_futures[1::2])
        return Decimal(16) * arctan_five - Decimal(4) * arctan_two_thirty_nine


def chudnovsky_term(term_index, precision):
    with localcontext() as context:
        context.prec = precision
        numerator = Decimal(math.factorial(6 * term_index)) * Decimal(
            13591409 + 545140134 * term_index
        )
        denominator = (
            Decimal(math.factorial(3 * term_index))
            * (Decimal(math.factorial(term_index)) ** 3)
            * (Decimal(CHUDNOVSKY_BASE) ** term_index)
        )
        return numerator / denominator


def chudnovsky_chunk(start_index, stop_index, precision):
    with localcontext() as context:
        context.prec = precision
        partial_sum = Decimal(0)

        for term_index in range(start_index, stop_index):
            partial_sum += chudnovsky_term(term_index, precision)

        return partial_sum


def chudnovsky_pi_parallel(term_count, worker_count, precision):
    """
    Compute π using the Chudnovsky series in parallel across multiple worker threads.
    The Chudnovsky series is a rapidly converging series for π, and each term
    can be computed independently. The total number of terms is divided among the workers,
    and the results are combined to produce the final estimate of π. The Chudnovsky
    series converges very rapidly, so even a small number of terms can yield a
    high-precision estimate of π.

    Args:
        term_count: Total number of terms to compute in the Chudnovsky series.
        worker_count: Number of worker threads to use for the computation.
        precision: The decimal precision to use for the calculations.

    Returns:
        An estimate of π based on the Chudnovsky series.
    """
    print(
        f"Computing π using the Chudnovsky series with {term_count} terms across {worker_count} workers at precision {precision}."
    )
    ranges = list(split_work(term_count, worker_count))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(chudnovsky_chunk, start_index, stop_index, precision)
            for start_index, stop_index in ranges
            if stop_index > start_index
        ]

    with localcontext() as context:
        context.prec = precision
        series_sum = sum(future.result() for future in futures)
        constant = Decimal(426880) * Decimal(10005).sqrt()
        return constant / series_sum


def bbp_chunk(start_index, stop_index, precision):
    with localcontext() as context:
        context.prec = precision
        partial_sum = Decimal(0)

        for term_index in range(start_index, stop_index):
            sixteen_power = Decimal(16) ** term_index
            partial_sum += (
                Decimal(4) / Decimal(8 * term_index + 1)
                - Decimal(2) / Decimal(8 * term_index + 4)
                - Decimal(1) / Decimal(8 * term_index + 5)
                - Decimal(1) / Decimal(8 * term_index + 6)
            ) / sixteen_power

        return partial_sum


def bbp_pi_parallel(term_count, worker_count, precision):
    """
    Compute π using the Bailey-Borwein-Plouffe (BBP) formula in parallel
    across multiple worker threads. The BBP formula allows for the computation
    of π to arbitrary precision and can be computed in a digit-extraction manner.
    Each term of the series can be computed independently,
    making it well-suited for parallelisation. The total number of terms is
    divided among the workers, and the results are combined to produce the
    final estimate of π.

    Args:
        term_count: Total number of terms to compute in the BBP series.
        worker_count: Number of worker threads to use for the computation.
        precision: The decimal precision to use for the calculations.

    Returns:
        An estimate of π based on the BBP formula.
    """
    print(
        f"Computing π using the Bailey-Borwein-Plouffe (BBP) formula with {term_count} terms across {worker_count} workers at precision {precision}."
    )
    ranges = list(split_work(term_count, worker_count))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(bbp_chunk, start_index, stop_index, precision)
            for start_index, stop_index in ranges
            if stop_index > start_index
        ]

    with localcontext() as context:
        context.prec = precision
        return sum(future.result() for future in futures)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute π using multiple CPU cores.")
    subparsers = parser.add_subparsers(dest="method", required=True)

    monte_carlo_parser = subparsers.add_parser(
        "monte-carlo", help="Estimate π via Monte Carlo sampling."
    )
    monte_carlo_parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help="Total Monte Carlo samples across all threads.",
    )

    machin_parser = subparsers.add_parser(
        "machin", help="Compute π with a Machin-like arctan identity."
    )
    machin_parser.add_argument(
        "--terms",
        type=int,
        default=DEFAULT_SERIES_TERMS,
        help="Number of terms in each arctan series.",
    )
    machin_parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision for intermediate and final results.",
    )

    chudnovsky_parser = subparsers.add_parser(
        "chudnovsky", help="Compute π using the Chudnovsky series."
    )
    chudnovsky_parser.add_argument(
        "--terms",
        type=int,
        default=DEFAULT_SERIES_TERMS,
        help="Number of Chudnovsky terms to sum.",
    )
    chudnovsky_parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision for intermediate and final results.",
    )

    bbp_parser = subparsers.add_parser("bbp", help="Compute π using the BBP series.")
    bbp_parser.add_argument(
        "--terms",
        type=int,
        default=DEFAULT_SERIES_TERMS,
        help="Number of BBP terms to sum.",
    )
    bbp_parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision for intermediate and final results.",
    )

    for method_parser in [
        monte_carlo_parser,
        machin_parser,
        chudnovsky_parser,
        bbp_parser,
    ]:
        method_parser.add_argument(
            "--workers",
            type=int,
            default=default_worker_count(),
            help=f"Number of worker threads to run. Available CPU cores: {default_worker_count()}.",
        )

    return parser.parse_args()


def run_parallel_pi():
    args = parse_args()
    worker_count = max(2, args.workers) if default_worker_count() > 1 else 1

    print(f"Detected {os.cpu_count()} cores.")
    print(
        f"Global Interpreter Lock (GIL) is {'enabled. Disable it using PYTHON_GIL=0' if sys._is_gil_enabled() else 'disabled. Enable it using PYTHON_GIL=1'}."
    )
    print(f"Python interpreter: {sys.version}")
    start_time = time.perf_counter()

    match args.method:
        case "bbp":
            pi_estimate = bbp_pi_parallel(args.terms, worker_count, args.precision)
        case "chudnovsky":
            pi_estimate = chudnovsky_pi_parallel(
                args.terms, worker_count, args.precision
            )
        case "machin":
            pi_estimate = machin_pi_parallel(args.terms, worker_count, args.precision)
        case "monte-carlo":
            pi_estimate = monte_carlo_pi_parallel(args.samples, worker_count)
        case _:
            # This will not happen due to argparse's required subparsers, but we include it for completeness
            raise ValueError(f"Unknown method: {args.method}")

    end_time = time.perf_counter()

    print(f"Calculated π: {pi_estimate}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    run_parallel_pi()
