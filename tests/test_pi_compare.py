import json
import sys
import unittest
from types import SimpleNamespace

from py_no_gil.pi import (
    argv_without_compare_flag,
    compare_gil,
    format_gil_comparison,
    parse_args,
    parse_execution_time,
)


class ParseExecutionTimeTests(unittest.TestCase):
    def test_reads_seconds_from_standard_output_line(self):
        output = "Calculated π: 3.14\nExecution time: 1.2345 seconds\n"
        self.assertEqual(parse_execution_time(output), 1.2345)

    def test_raises_when_time_line_is_missing(self):
        with self.assertRaises(ValueError):
            parse_execution_time("no timing here\n")


class ArgvWithoutCompareFlagTests(unittest.TestCase):
    def test_drops_compare_gil_flag(self):
        self.assertEqual(
            argv_without_compare_flag(["bbp", "--compare-gil", "--terms", "8"]),
            ["bbp", "--terms", "8"],
        )


class FormatGilComparisonTests(unittest.TestCase):
    def test_reports_speedup_of_gil_off_versus_gil_on(self):
        text = format_gil_comparison(gil_on_seconds=2.0, gil_off_seconds=0.5)
        self.assertIn("2.0000", text)
        self.assertIn("0.5000", text)
        self.assertIn("4.00x", text)


class CompareGilTests(unittest.TestCase):
    def test_json_comparison_returns_a_single_record(self):
        def fake_runner(cmd, env, **kwargs):
            duration = 2.0 if env["PYTHON_GIL"] == "1" else 0.5
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"timing": {"median_seconds": duration}}),
                stderr="",
            )

        report = json.loads(
            compare_gil(["bbp", "--json"], runner=fake_runner, output_json=True)
        )
        self.assertEqual(report["speedup_gil_disabled_vs_enabled"], 4.0)
        self.assertIn("gil_enabled", report)

    def test_runs_workload_with_gil_on_then_off(self):
        calls = []

        def fake_runner(cmd, env, **kwargs):
            calls.append({"gil": env["PYTHON_GIL"], "cmd": cmd})
            duration = "2.0000" if env["PYTHON_GIL"] == "1" else "0.5000"
            return SimpleNamespace(
                returncode=0,
                stdout=f"Execution time: {duration} seconds\n",
                stderr="",
            )

        report = compare_gil(
            ["bbp", "--compare-gil", "--terms", "4"], runner=fake_runner
        )
        self.assertEqual([call["gil"] for call in calls], ["1", "0"])
        for call in calls:
            self.assertEqual(call["cmd"][:3], [sys.executable, "-m", "py_no_gil.pi"])
            self.assertNotIn("--compare-gil", call["cmd"])
        self.assertIn("4.00x", report)

    def test_raises_when_child_process_fails(self):
        def fake_runner(cmd, env=None, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")

        with self.assertRaises(RuntimeError):
            compare_gil(["bbp"], runner=fake_runner)


class ParseArgsCompareFlagTests(unittest.TestCase):
    def test_compare_gil_defaults_to_false(self):
        args = parse_args(["bbp"])
        self.assertFalse(args.compare_gil)

    def test_compare_gil_flag_is_accepted_on_each_method(self):
        for method in ("monte-carlo", "machin", "chudnovsky", "bbp"):
            with self.subTest(method=method):
                args = parse_args([method, "--compare-gil"])
                self.assertTrue(args.compare_gil)
