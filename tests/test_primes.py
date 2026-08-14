import io
import unittest
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
    def test_modes_print_result_and_execution_time(self):
        cases = (
            (["count", "--start", "1", "--stop", "100"], "Prime count: 25"),
            (["list", "--start", "10", "--stop", "20"], "Primes: 11, 13, 17, 19"),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                buffer = io.StringIO()
                with patch("sys.stdout", buffer):
                    run_parallel_primes(argv)
                self.assertIn(expected, buffer.getvalue())
                self.assertIn("Execution time:", buffer.getvalue())
