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
