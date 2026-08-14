# Python with no GIL, a.k.a. py-no-gil

This repository contains some code examples demonstrating the use of free-threading in Python, i.e., without the Global Interpreter Lock (GIL). Since Python 3.14, the GIL is disabled by default, allowing for true multi-threading in Python. The GIL may be enabled using `PYTHON_GIL=1`.

## Usage

Run the code using the command line interface (CLI) as follows:

```bash
uv run py-no-gil
```

This will display the available scripts.

### Computing π

This is a simple demonstration of computing π in Python using multiple threads without the Global Interpreter Lock (GIL). It includes four methods for estimating π: the Monte-Carlo estimation, the Machin-like arctan identity, the Chudnovsky series, and the Bailey-Borwein-Plouffe (BBP) formula. The program detects the number of CPU cores available and uses that information to determine how many worker threads to create for parallel computation.

To run the π estimation, use the following command:

```bash
uv run parallel-pi --help
```

The options specific to a method can be seen by running the following, e.g., for the BBP method:

```bash
uv run parallel-pi bbp --help
```

To compare the same workload with the GIL enabled and disabled, add `--compare-gil`. This runs the method twice (`PYTHON_GIL=1`, then `PYTHON_GIL=0`) and prints a timing table:

```bash
uv run parallel-pi bbp --compare-gil
uv run parallel-pi monte-carlo --compare-gil --samples 1000000
```

### Prime search

This demo partitions a range `[start, stop)` across worker threads and either counts or lists the prime numbers in that range. It uses the same parallel-work-splitting strategy as the π and fractal examples.

```bash
uv run parallel-primes --help
uv run parallel-primes count --start 1 --stop 1000000
uv run parallel-primes list --start 100 --stop 200
```

To compare the same workload with the GIL enabled and disabled:

```bash
uv run parallel-primes count --compare-gil
uv run parallel-primes count --compare-gil --stop 1000000
```

### Mandelbrot and Julia sets

This demo splits a fractal image by rows across worker threads, writes a P6 `.ppm` you can open in an image viewer, and prints a small truecolor ANSI preview in the terminal.

```bash
uv run parallel-fractal --help
uv run parallel-fractal mandelbrot
uv run parallel-fractal julia --c-real -0.8 --c-imag 0.156
```

To compare the same render with the GIL enabled and disabled:

```bash
uv run parallel-fractal mandelbrot --compare-gil
uv run parallel-fractal julia --compare-gil --width 400 --height 300
```
