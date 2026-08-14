import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from py_no_gil.scale import (
    DEFAULT_WORKER_COUNTS,
    MODULES,
    parse_args,
    run_parallel_scale,
    sweep,
)


class ParseScaleArgsTests(unittest.TestCase):
    def test_defaults(self):
        args = parse_args(["--module", "primes"])
        self.assertEqual(args.module, "primes")
        self.assertEqual(args.workers, DEFAULT_WORKER_COUNTS)
        self.assertFalse(args.compare_gil)
        self.assertEqual(args.module_args, [])

    def test_accepts_worker_list_and_module_args(self):
        args = parse_args(
            [
                "--module",
                "fractal",
                "--workers",
                "1,2,4,8",
                "--compare-gil",
                "--",
                "mandelbrot",
                "--width",
                "200",
                "--height",
                "200",
            ]
        )
        self.assertEqual(args.module, "fractal")
        self.assertEqual(args.workers, "1,2,4,8")
        self.assertTrue(args.compare_gil)
        self.assertEqual(
            args.module_args, ["mandelbrot", "--width", "200", "--height", "200"]
        )

    def test_drops_compare_gil_from_module_args(self):
        args = parse_args(
            ["--module", "primes", "--compare-gil", "--", "count", "--start", "1"]
        )
        self.assertNotIn("--compare-gil", args.module_args)
        self.assertTrue(args.compare_gil)

    def test_all_modules_are_valid_choices(self):
        for module in MODULES:
            args = parse_args(["--module", module])
            self.assertEqual(args.module, module)


class SweepTests(unittest.TestCase):
    def test_sweep_returns_results_for_each_worker_count(self):
        results = sweep("primes", ["count", "--start", "1", "--stop", "100"], [1, 2])
        self.assertEqual(len(results), 2)
        self.assertEqual([wc for wc, _ in results], [1, 2])
        for wc, t in results:
            self.assertIsInstance(t, float)
            self.assertGreaterEqual(t, 0)

    def test_sweep_with_gil_on_sets_env(self):
        results = sweep(
            "primes", ["count", "--start", "1", "--stop", "100"], [1], gil_on=True
        )
        self.assertEqual(len(results), 1)
        wc, t = results[0]
        self.assertEqual(wc, 1)
        self.assertGreaterEqual(t, 0)


class RunParallelScaleTests(unittest.TestCase):
    def test_count_prints_scaling_table_and_execution_time(self):
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_scale(
                [
                    "--module",
                    "primes",
                    "--workers",
                    "1,2",
                    "--",
                    "count",
                    "--start",
                    "1",
                    "--stop",
                    "100",
                ]
            )

        output = buffer.getvalue()
        self.assertIn("Scaling sweep for py_no_gil.primes", output)
        self.assertIn("Workers", output)
        self.assertIn("Time (s)", output)
        self.assertIn("Speedup", output)
        self.assertIn("Chart", output)

    def test_list_prints_scaling_table(self):
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_scale(
                [
                    "--module",
                    "primes",
                    "--workers",
                    "1,2",
                    "--",
                    "list",
                    "--start",
                    "10",
                    "--stop",
                    "20",
                ]
            )

        output = buffer.getvalue()
        self.assertIn("Scaling sweep for py_no_gil.primes", output)
        self.assertIn("Workers", output)
        self.assertIn("Time (s)", output)

    def test_compare_flag_runs_with_gil_on_then_off(self):
        calls = []

        def fake_runner(cmd, env, **kwargs):
            calls.append({"gil": env.get("PYTHON_GIL"), "cmd": cmd})
            gil = env.get("PYTHON_GIL")
            if gil == "1":
                duration = "4.0000"
            else:
                # Vary timing by worker count to simulate scaling
                wc_str = "1"
                for i, a in enumerate(cmd):
                    if a == "--workers" and i + 1 < len(cmd):
                        wc_str = cmd[i + 1]
                        break
                # GIL off: more workers = faster
                durations = {"1": "2.0000", "2": "1.0000", "4": "0.5000"}
                duration = durations.get(wc_str, "1.0000")
            return SimpleNamespace(
                returncode=0,
                stdout=f"Execution time: {duration} seconds\n",
                stderr="",
            )

        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_scale(
                [
                    "--module",
                    "primes",
                    "--workers",
                    "1,2,4",
                    "--compare-gil",
                    "--",
                    "count",
                    "--start",
                    "1",
                    "--stop",
                    "100",
                ],
                runner=fake_runner,
            )

        output = buffer.getvalue()
        gil_values = [call["gil"] for call in calls]
        self.assertEqual(gil_values, ["0", "0", "0", "1", "1", "1"])
        self.assertIn("--- GIL disabled (PYTHON_GIL=0 / free-threading) ---", output)
        self.assertIn("--- GIL enabled (PYTHON_GIL=1) ---", output)
        self.assertIn("speedup vs GIL-on", output)


if __name__ == "__main__":
    unittest.main()
