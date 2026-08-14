import csv
import io
import json
import unittest
from unittest.mock import patch

from py_no_gil.primes import parse_args, run_parallel_primes
from py_no_gil.scale import run_parallel_scale


class BenchmarkCliTests(unittest.TestCase):
    def test_workers_one_is_preserved(self):
        self.assertEqual(parse_args(["count", "--workers", "1"]).workers, 1)

    def test_json_output_is_one_parseable_record(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            run_parallel_primes(
                [
                    "count",
                    "--start",
                    "1",
                    "--stop",
                    "20",
                    "--workers",
                    "1",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                    "--json",
                ]
            )
        record = json.loads(output.getvalue())
        self.assertEqual(record["workers"], 1)
        self.assertEqual(record["result"], {"count": 8})
        self.assertEqual(len(record["timing"]["samples_seconds"]), 2)
        self.assertIn("gil_enabled", record["environment"])

    def test_scale_csv_contains_efficiency_column(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            run_parallel_scale(
                [
                    "--module",
                    "primes",
                    "--workers",
                    "1",
                    "--repeat",
                    "1",
                    "--warmup",
                    "1",
                    "--output-format",
                    "csv",
                    "--",
                    "count",
                    "--start",
                    "1",
                    "--stop",
                    "20",
                ]
            )
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(rows[0]["workers"], "1")
        self.assertIn("efficiency", rows[0])


if __name__ == "__main__":
    unittest.main()
