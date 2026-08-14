import argparse
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from py_no_gil.pi import compare_gil, default_worker_count, split_work

DEFAULT_PARTICLES = 100
DEFAULT_STEPS = 100
DEFAULT_DT = 0.01
DEFAULT_G = 0.1


def _compute_force_on_particle(i_idx, particles, G, n):
    xi, yi, _, _, mi = particles[i_idx]
    fx = 0.0
    fy = 0.0
    for j_idx in range(n):
        if j_idx == i_idx:
            continue
        xj, yj, _, _, mj = particles[j_idx]
        dx = xj - xi
        dy = yj - yi
        dist_sq = dx * dx + dy * dy
        if dist_sq == 0:
            continue
        dist = math.sqrt(dist_sq)
        force_mag = G * mi * mj / dist_sq
        fx += force_mag * dx / dist
        fy += force_mag * dy / dist
    return (fx, fy)


def _worker_force_range(start_i, end_i, particles, G, n):
    results = []
    for i in range(start_i, end_i):
        fx, fy = _compute_force_on_particle(i, particles, G, n)
        results.append((i, fx, fy))
    return results


def compute_forces_parallel(particles, G, n, worker_count):
    """Compute forces for each particle in parallel.

    Each worker computes forces for a slice of the force loop.
    Returns a dict mapping particle index to (fx, fy).
    """
    worker_count = max(1, min(worker_count, n))
    ranges = list(split_work(n, worker_count))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_worker_force_range, start, stop, particles, G, n)
            for start, stop in ranges
            if stop > start
        ]
        partial_results = [future.result() for future in futures]

    force_map = {}
    for results in partial_results:
        for i, fx, fy in results:
            force_map[i] = (fx, fy)
    return force_map


def simulate_step(particles, G, dt, n, worker_count):
    """Perform one integration step using parallel force computation."""
    force_map = compute_forces_parallel(particles, G, n, worker_count)

    for i in range(n):
        fx, fy = force_map[i]
        _, _, vxi, vyi, mi = particles[i]
        particles[i] = (
            particles[i][0] + vxi * dt + fx / mi * dt * dt / 2,
            particles[i][1] + vyi * dt + fy / mi * dt * dt / 2,
            vxi + fx / mi * dt,
            vyi + fy / mi * dt,
            mi,
        )
    return particles


def simulate(particles, G, dt, steps, n, worker_count=None):
    """Run the N-body simulation for multiple steps.

    Each particle is a tuple: (x, y, vx, vy, mass).
    """
    if worker_count is None:
        worker_count = default_worker_count()
    for _ in range(steps):
        particles = simulate_step(particles, G, dt, n, worker_count)
    return particles


def search_nbody_parallel(worker_count, particles, G, dt, steps, mode="run", n=None):
    """Run N-body simulation in parallel.

    Args:
        worker_count: number of worker threads
        particles: initial particle state
        G: gravitational constant
        dt: time step
        steps: number of simulation steps
        mode: "run" (default), "count", or "list"
        n: number of particles (defaults to len(particles))

    Returns:
        - mode="count": 1 (simulations completed)
        - mode="list" or "run": final particle state
    """
    if n is None:
        n = len(particles)

    p = [list(pi) for pi in particles]
    result = simulate(p, G, dt, steps, n, worker_count)
    final = [tuple(pi) for pi in result]

    if mode == "count":
        return 1
    return final


def _make_initial_particles(n):
    import random

    random.seed(42)
    particles = []
    for i in range(n):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        particles.append((x, y, 0.0, 0.0, 1.0))
    return particles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="parallel-nbody",
        description="N-body particle simulation using multiple CPU cores.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run N-body simulation")
    count_p = subparsers.add_parser("count", help="Run simulation and report count")
    list_p = subparsers.add_parser(
        "list", help="Run simulation and list final positions"
    )

    for sub in (run_p, count_p, list_p):
        sub.add_argument(
            "--particles",
            type=int,
            default=DEFAULT_PARTICLES,
            help="Number of particles.",
        )
        sub.add_argument(
            "--steps",
            type=int,
            default=DEFAULT_STEPS,
            help="Number of simulation steps.",
        )
        sub.add_argument(
            "--dt",
            type=float,
            default=DEFAULT_DT,
            help="Time step for integration.",
        )
        sub.add_argument(
            "--grav-const",
            type=float,
            default=DEFAULT_G,
            help="Gravitational constant.",
        )
        sub.add_argument(
            "--workers",
            type=int,
            default=default_worker_count(),
            help=f"Number of worker threads. Available CPU cores: {default_worker_count()}.",
        )
        sub.add_argument(
            "--compare-gil",
            action="store_true",
            help=(
                "Run the same workload with PYTHON_GIL=1 and PYTHON_GIL=0, "
                "then print a timing comparison."
            ),
        )

    return parser.parse_args(argv)


def run_parallel_nbody(argv=None, runner=subprocess.run):
    args = parse_args(argv)
    if args.compare_gil:
        child_argv = sys.argv[1:] if argv is None else argv
        print(compare_gil(child_argv, runner=runner, module="py_no_gil.nbody"))
        return

    worker_count = max(2, args.workers) if default_worker_count() > 1 else 1
    n = args.particles

    print(f"Detected {os.cpu_count()} cores.")
    print(
        f"Global Interpreter Lock (GIL) is "
        f"{'enabled. Disable it using PYTHON_GIL=0' if sys._is_gil_enabled() else 'disabled. Enable it using PYTHON_GIL=1'}."
    )
    print(f"Python interpreter: {sys.version}")
    print(
        f"Simulating {n} particles for {args.steps} steps across {worker_count} workers."
    )

    start_time = time.perf_counter()

    particles = _make_initial_particles(n)
    if args.command == "count":
        result = search_nbody_parallel(
            worker_count,
            particles,
            args.grav_const,
            args.dt,
            args.steps,
            mode="count",
            n=n,
        )
        print(f"Simulation count: {result}")
    else:
        result = search_nbody_parallel(
            worker_count,
            particles,
            args.grav_const,
            args.dt,
            args.steps,
            mode="list",
            n=n,
        )
        count = len(result)
        print(f"Final particle count: {count}")
        print(f"Sample positions: {result[:3]}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    run_parallel_nbody()
