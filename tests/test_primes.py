import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from py_no_gil.primes import (
    count_primes_in_range,
    is_prime,
    list_primes_in_range,
    parse_args,
    run_parallel_primes,
    search_primes_parallel,
)


class IsPrimeTests(unittest.TestCase):
    def test_rejects_values_below_two(self):
        for value in (0, 1, -3):
            with self.subTest(value=value):
                self.assertFalse(is_prime(value))

    def test_recognizes_small_primes(self):
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(97))

    def test_rejects_composites(self):
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(100))


class CountPrimesInRangeTests(unittest.TestCase):
    def test_counts_primes_in_half_open_interval(self):
        self.assertEqual(count_primes_in_range(10, 20), 4)


class ListPrimesInRangeTests(unittest.TestCase):
    def test_lists_primes_in_half_open_interval(self):
        self.assertEqual(list_primes_in_range(10, 20), [11, 13, 17, 19])


class SearchPrimesParallelTests(unittest.TestCase):
    def test_count_matches_single_threaded_result(self):
        kwargs = {"start": 1, "stop": 500, "mode": "count"}
        self.assertEqual(search_primes_parallel(worker_count=1, **kwargs), 95)
        self.assertEqual(search_primes_parallel(worker_count=4, **kwargs), 95)

    def test_list_matches_single_threaded_result(self):
        kwargs = {"start": 50, "stop": 100, "mode": "list"}
        single = search_primes_parallel(worker_count=1, **kwargs)
        split = search_primes_parallel(worker_count=3, **kwargs)
        self.assertEqual(single, [53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
        self.assertEqual(split, single)


class ParsePrimesArgsTests(unittest.TestCase):
    def test_count_defaults(self):
        args = parse_args(["count"])
        self.assertEqual(args.mode, "count")
        self.assertEqual(args.start, 1)
        self.assertEqual(args.stop, 1_000_000)
        self.assertFalse(args.compare_gil)

    def test_list_accepts_range_and_compare_flag(self):
        args = parse_args(["list", "--start", "100", "--stop", "200", "--compare-gil"])
        self.assertEqual(args.mode, "list")
        self.assertEqual(args.start, 100)
        self.assertEqual(args.stop, 200)
        self.assertTrue(args.compare_gil)


class RunParallelPrimesTests(unittest.TestCase):
    def test_count_prints_result_and_execution_time(self):
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_primes(["count", "--start", "1", "--stop", "100"])

        output = buffer.getvalue()
        self.assertIn("Prime count: 25", output)
        self.assertIn("Execution time:", output)

    def test_list_prints_primes_and_execution_time(self):
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_primes(["list", "--start", "10", "--stop", "20"])

        output = buffer.getvalue()
        self.assertIn("Primes: 11, 13, 17, 19", output)
        self.assertIn("Execution time:", output)

    def test_compare_flag_runs_primes_module_with_gil_on_then_off(self):
        calls = []

        def fake_runner(cmd, env, **kwargs):
            calls.append({"gil": env["PYTHON_GIL"], "cmd": cmd})
            duration = "2.0000" if env["PYTHON_GIL"] == "1" else "0.5000"
            return SimpleNamespace(
                returncode=0,
                stdout=f"Execution time: {duration} seconds\n",
                stderr="",
            )

        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_primes(
                ["count", "--compare-gil", "--stop", "1000"], runner=fake_runner
            )

        output = buffer.getvalue()
        self.assertEqual([call["gil"] for call in calls], ["1", "0"])
        for call in calls:
            self.assertEqual(
                call["cmd"][:3], [sys.executable, "-m", "py_no_gil.primes"]
            )
            self.assertNotIn("--compare-gil", call["cmd"])
        self.assertIn("4.00x", output)
        self.assertNotIn("Prime count:", output)
