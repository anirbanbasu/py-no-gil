import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from py_no_gil.pi import compare_gil, default_worker_count, split_work
from py_no_gil.benchmark import (
    add_benchmark_arguments,
    positive_int,
    print_benchmark,
    run_benchmark,
)

DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_ITER = 256
DEFAULT_PREVIEW_WIDTH = 80
DEFAULT_JULIA_C_REAL = -0.8
DEFAULT_JULIA_C_IMAG = 0.156


def escape_iterations(z_real, z_imag, c_real, c_imag, max_iter):
    for iteration in range(max_iter):
        if z_real * z_real + z_imag * z_imag > 4.0:
            return iteration
        z_real, z_imag = (
            z_real * z_real - z_imag * z_imag + c_real,
            2.0 * z_real * z_imag + c_imag,
        )
    return max_iter


def pixel_complex(px, py, width, height, xmin, xmax, ymin, ymax):
    real = xmin + (px + 0.5) * (xmax - xmin) / width
    imag = ymax - (py + 0.5) * (ymax - ymin) / height
    return real, imag


def color_from_iterations(iterations, max_iter):
    if iterations >= max_iter:
        return (0, 0, 0)
    t_value = iterations / max_iter
    red = int(127.5 * (1 + math.sin(2 * math.pi * t_value)))
    green = int(127.5 * (1 + math.sin(2 * math.pi * t_value + 2.094)))
    blue = int(127.5 * (1 + math.sin(2 * math.pi * t_value + 4.189)))
    return (red, green, blue)


def write_ppm(path, width, height, pixels):
    output_path = Path(path)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytes(channel for pixel in pixels for channel in pixel)
    output_path.write_bytes(header + body)


def format_ansi_preview(pixels, width, height, preview_width):
    preview_width = max(1, min(preview_width, width))
    preview_height = max(1, round(height * preview_width / width / 2))
    lines = []

    for preview_y in range(preview_height):
        cells = []
        for preview_x in range(preview_width):
            source_x = min(width - 1, preview_x * width // preview_width)
            top_y = min(height - 1, (preview_y * 2) * height // (preview_height * 2))
            bottom_y = min(
                height - 1, (preview_y * 2 + 1) * height // (preview_height * 2)
            )
            top = pixels[top_y * width + source_x]
            bottom = pixels[bottom_y * width + source_x]
            cells.append(
                f"\x1b[38;2;{top[0]};{top[1]};{top[2]}m"
                f"\x1b[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀"
            )
        lines.append("".join(cells) + "\x1b[0m")

    return "\n".join(lines)


def default_bounds(kind, width, height):
    if kind == "mandelbrot":
        xmin, xmax = -2.5, 1.0
    else:
        xmin, xmax = -1.5, 1.5
    x_range = xmax - xmin
    y_range = x_range * height / width
    return xmin, xmax, -y_range / 2.0, y_range / 2.0


def render_row(
    y, width, height, max_iter, kind, xmin, xmax, ymin, ymax, c_real, c_imag
):
    row = []
    for x in range(width):
        real, imag = pixel_complex(x, y, width, height, xmin, xmax, ymin, ymax)
        if kind == "mandelbrot":
            iterations = escape_iterations(0.0, 0.0, real, imag, max_iter)
        else:
            iterations = escape_iterations(real, imag, c_real, c_imag, max_iter)
        row.append(color_from_iterations(iterations, max_iter))
    return row


def render_rows(
    start_y,
    stop_y,
    width,
    height,
    max_iter,
    kind,
    xmin,
    xmax,
    ymin,
    ymax,
    c_real,
    c_imag,
):
    rows = [
        render_row(
            y, width, height, max_iter, kind, xmin, xmax, ymin, ymax, c_real, c_imag
        )
        for y in range(start_y, stop_y)
    ]
    return start_y, rows


def render_fractal(
    width,
    height,
    max_iter,
    worker_count,
    kind,
    xmin,
    xmax,
    ymin,
    ymax,
    c_real=0.0,
    c_imag=0.0,
):
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                render_rows,
                start_y,
                stop_y,
                width,
                height,
                max_iter,
                kind,
                xmin,
                xmax,
                ymin,
                ymax,
                c_real,
                c_imag,
            )
            for start_y, stop_y in split_work(height, worker_count)
            if stop_y > start_y
        ]

    chunks = [future.result() for future in futures]
    chunks.sort(key=lambda item: item[0])
    pixels = []
    for _, rows in chunks:
        for row in rows:
            pixels.extend(row)
    return pixels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render a Mandelbrot or Julia set using multiple CPU cores."
    )
    subparsers = parser.add_subparsers(dest="kind", required=True)

    mandelbrot_parser = subparsers.add_parser(
        "mandelbrot", help="Render the Mandelbrot set."
    )
    mandelbrot_parser.add_argument(
        "--output",
        default="mandelbrot.ppm",
        help="Path of the P6 PPM file to write.",
    )

    julia_parser = subparsers.add_parser("julia", help="Render a Julia set.")
    julia_parser.add_argument(
        "--c-real",
        type=float,
        default=DEFAULT_JULIA_C_REAL,
        help="Real part of the Julia parameter c.",
    )
    julia_parser.add_argument(
        "--c-imag",
        type=float,
        default=DEFAULT_JULIA_C_IMAG,
        help="Imaginary part of the Julia parameter c.",
    )
    julia_parser.add_argument(
        "--output",
        default="julia.ppm",
        help="Path of the P6 PPM file to write.",
    )

    for kind_parser in [mandelbrot_parser, julia_parser]:
        kind_parser.add_argument(
            "--width",
            type=int,
            default=DEFAULT_WIDTH,
            help="Image width in pixels.",
        )
        kind_parser.add_argument(
            "--height",
            type=int,
            default=DEFAULT_HEIGHT,
            help="Image height in pixels.",
        )
        kind_parser.add_argument(
            "--iter",
            type=int,
            default=DEFAULT_ITER,
            help="Escape-time iteration limit (higher values produce finer detail).",
        )
        kind_parser.add_argument(
            "--preview-width",
            type=int,
            default=DEFAULT_PREVIEW_WIDTH,
            help="Width of the ANSI terminal preview in characters.",
        )
        kind_parser.add_argument(
            "--workers",
            type=positive_int,
            default=default_worker_count(),
            help=f"Number of worker threads to run. Available CPU cores: {default_worker_count()}.",
        )
        kind_parser.add_argument(
            "--compare-gil",
            action="store_true",
            help=(
                "Run the same workload with PYTHON_GIL=1 and PYTHON_GIL=0, "
                "then print a timing comparison."
            ),
        )
        add_benchmark_arguments(kind_parser)

    return parser.parse_args(argv)


def run_parallel_fractal(argv=None, runner=subprocess.run):
    args = parse_args(argv)
    if args.compare_gil:
        child_argv = sys.argv[1:] if argv is None else argv
        print(
            compare_gil(
                child_argv,
                runner=runner,
                module="py_no_gil.fractal",
                output_json=args.json,
            )
        )
        return

    worker_count = args.workers
    xmin, xmax, ymin, ymax = default_bounds(args.kind, args.width, args.height)
    c_real = getattr(args, "c_real", 0.0)
    c_imag = getattr(args, "c_imag", 0.0)

    if not args.json:
        print(f"Detected {os.cpu_count()} cores.")
        print(
            f"Global Interpreter Lock (GIL) is {'enabled. Disable it using PYTHON_GIL=0' if sys._is_gil_enabled() else 'disabled. Enable it using PYTHON_GIL=1'}."
        )
    if not args.json and args.kind == "julia":
        print(
            f"Computing julia set with c={c_real}+{c_imag}i "
            f"({args.width}x{args.height}, iter={args.iter}) "
            f"across {worker_count} workers."
        )
    elif not args.json:
        print(
            f"Computing mandelbrot set "
            f"({args.width}x{args.height}, iter={args.iter}) "
            f"across {worker_count} workers."
        )

    pixels, timing = run_benchmark(
        lambda: render_fractal(
            args.width,
            args.height,
            args.iter,
            worker_count,
            args.kind,
            xmin,
            xmax,
            ymin,
            ymax,
            c_real=c_real,
            c_imag=c_imag,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
        quiet=args.json,
    )

    output_path = Path(args.output)
    write_ppm(output_path, args.width, args.height, pixels)
    if not args.json:
        print(f"Wrote {output_path}")
        print(format_ansi_preview(pixels, args.width, args.height, args.preview_width))
    print_benchmark(
        name="fractal",
        workers=worker_count,
        parameters={
            "kind": args.kind,
            "width": args.width,
            "height": args.height,
            "max_iter": args.iter,
            "output": str(output_path),
            "repeat": args.repeat,
            "warmup": args.warmup,
        },
        timing=timing,
        result={"output": str(output_path), "pixels": len(pixels)},
        output_json=args.json,
    )


if __name__ == "__main__":
    run_parallel_fractal()
