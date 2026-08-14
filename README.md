# Python with no GIL, a.k.a. py-no-gil

This repository contains code examples demonstrating free-threading in Python, i.e., execution without the Global Interpreter Lock (GIL). Free-threading is available in a separately built CPython interpreter (enabled with `--disable-gil` when building CPython); it is not the default Python build. A free-threaded build can run with the GIL enabled or disabled using `PYTHON_GIL=1` or `PYTHON_GIL=0` respectively. Check the running interpreter with `sysconfig.get_config_var("Py_GIL_DISABLED")` and its current setting with `sys._is_gil_enabled()`.

## Usage

Run the code using the command line interface (CLI) as follows:

```bash
uv run py-no-gil
```

This will display the available scripts.

## Benchmarking

Every workload accepts `--workers N`, `--warmup N`, and `--repeat N`. One warm-up and three measured runs are the defaults; the reported execution time is the median, with minimum and maximum shown alongside it. Set `--workers 1` to establish a serial baseline—this value is respected exactly.

Every workload also accepts `--json`, which prints one machine-readable JSON record containing the workload parameters, result summary, timing samples, interpreter details, free-threaded-build status, and current GIL state. For example:

```bash
uv run parallel-primes count --start 1 --stop 1000000 --workers 4 --repeat 5 --json
```

Benchmark results vary with CPU topology, memory bandwidth, thermal limits, system load, allocation overhead, task granularity, and contention. Free-threading removes the GIL bottleneck for compatible CPU-bound Python work; it does not guarantee linear scaling.

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

### N-body simulation

This demo simulates a system of particles under mutual gravitational attraction. The force calculation — a loop over all particle pairs — is split across worker threads, matching the parallel-work-splitting strategy used by the π and fractal examples.

```bash
uv run parallel-nbody --help
uv run parallel-nbody count --particles 1000 --steps 200
uv run parallel-nbody list --particles 500 --steps 100
```

To compare the same workload with the GIL enabled and disabled:

```bash
uv run parallel-nbody count --compare-gil
uv run parallel-nbody list --compare-gil --particles 1000 --steps 100
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

### Scaling sweep

This utility sweeps over thread counts (`--workers 1,2,4,8,...`) for **any** of the workload modules above (π, primes, fractal, nbody) and prints a scaling table with an ASCII bar chart. With the GIL enabled you should see a flat time curve (the GIL serializes pure-Python CPU work); without the GIL the time should drop as you add threads until you run out of cores.

```bash
uv run parallel-scale --module primes --workers 1,2,4,8 -- count --start 1 --stop 1000000
uv run parallel-scale --module nbody --workers 1,2,4,8 --compare-gil -- count --particles 500 --steps 50
```

The `--compare-gil` flag runs each configuration twice (GIL on and off) and prints both curves side-by-side so you can see the contrast. Note that the `--compare-gil` flag is not a flag for the chosen module so, it should precede the `--`.

The table includes parallel efficiency (speedup divided by worker count). Use `--output-format json` or `--output-format csv` when you want to save or graph the sweep:

```bash
uv run parallel-scale --module primes --workers 1,2,4,8 --output-format csv -- count --start 1 --stop 1000000
```

For example, running `uv run parallel-scale --module pi -- monte-carlo` shows something like the following.

```bash
Scaling sweep for py_no_gil.pi
Module args: ['monte-carlo']

--- GIL disabled (PYTHON_GIL=0 / free-threading) ---

Workers |   Time (s) |  Speedup | Chart
------- | ---------- | -------- | ------------------------------------------------
      1 |     0.8739 |    1.00x | ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
      2 |     0.7458 |    1.17x | ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■□□□□□□
      4 |     0.5444 |    1.61x | ■■■■■■■■■■■■■■■■■■■■■■■■□□□□□□□□□□□□□□□□
      8 |     0.3726 |    2.35x | ■■■■■■■■■■■■■■■■■□□□□□□□□□□□□□□□□□□□□□□□
```
