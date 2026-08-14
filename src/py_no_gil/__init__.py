"""Top-level command for the py-no-gil demonstrations."""

import argparse
import sys


COMMANDS = {
    "pi": ("parallel-pi", "Compute π with several parallel algorithms."),
    "primes": ("parallel-primes", "Search for prime numbers in parallel."),
    "fractal": ("parallel-fractal", "Render Mandelbrot and Julia sets in parallel."),
    "nbody": ("parallel-nbody", "Run a parallel N-body simulation."),
    "scale": ("parallel-scale", "Sweep worker counts for any workload."),
}


def _parser():
    command_list = "\n".join(
        f"  {name:<8} {description} (also: {script})"
        for name, (script, description) in COMMANDS.items()
    )
    return argparse.ArgumentParser(
        prog="py-no-gil",
        description="Examples and benchmarks for CPython free-threading.",
        epilog=(
            "Commands:\n"
            f"{command_list}\n\n"
            "All workloads accept --workers, --warmup, --repeat, and --json. "
            "Use --compare-gil to run a workload with the GIL enabled and disabled.\n\n"
            "Examples:\n"
            "  uv run py-no-gil primes count --workers 4 --repeat 5\n"
            "  uv run py-no-gil scale --module primes --workers 1,2,4,8 -- count\n\n"
            "For command-specific options, run: uv run py-no-gil <command> --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in COMMANDS:
        command = arguments.pop(0)
        _run_command(command, arguments)
        return

    parser = _parser()
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return

    _run_command(args.command, [])


def _run_command(command, argv):
    if command == "pi":
        from py_no_gil.pi import run_parallel_pi

        run_parallel_pi(argv)
    elif command == "primes":
        from py_no_gil.primes import run_parallel_primes

        run_parallel_primes(argv)
    elif command == "fractal":
        from py_no_gil.fractal import run_parallel_fractal

        run_parallel_fractal(argv)
    elif command == "nbody":
        from py_no_gil.nbody import run_parallel_nbody

        run_parallel_nbody(argv)
    else:
        from py_no_gil.scale import run_parallel_scale

        run_parallel_scale(argv)
