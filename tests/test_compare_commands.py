import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from py_no_gil.fractal import run_parallel_fractal
from py_no_gil.nbody import run_parallel_nbody
from py_no_gil.pi import run_parallel_pi
from py_no_gil.primes import run_parallel_primes


class CompareGilCommandTests(unittest.TestCase):
    def test_each_workload_uses_the_shared_gil_comparison_harness(self):
        cases = (
            (
                "pi",
                run_parallel_pi,
                ["bbp", "--compare-gil", "--terms", "4"],
                "Calculated π:",
            ),
            (
                "primes",
                run_parallel_primes,
                ["count", "--compare-gil", "--stop", "1000"],
                "Prime count:",
            ),
            (
                "fractal",
                run_parallel_fractal,
                ["julia", "--compare-gil", "--width", "8"],
                "\x1b[38;2;",
            ),
            (
                "nbody",
                run_parallel_nbody,
                ["count", "--compare-gil", "--particles", "20", "--steps", "20"],
                "Simulation count:",
            ),
        )
        for module, command, argv, local_output in cases:
            with self.subTest(module=module):
                calls = []

                def fake_runner(cmd, env, **kwargs):
                    calls.append({"gil": env["PYTHON_GIL"], "cmd": cmd})
                    duration = "2.0000" if env["PYTHON_GIL"] == "1" else "0.5000"
                    return SimpleNamespace(
                        returncode=0,
                        stdout=f"Execution time: {duration} seconds\n",
                        stderr="",
                    )

                output = io.StringIO()
                with patch("sys.stdout", output):
                    command(argv, runner=fake_runner)

                self.assertEqual([call["gil"] for call in calls], ["1", "0"])
                for call in calls:
                    self.assertEqual(
                        call["cmd"][:3], [sys.executable, "-m", f"py_no_gil.{module}"]
                    )
                    self.assertNotIn("--compare-gil", call["cmd"])
                self.assertIn("4.00x", output.getvalue())
                self.assertNotIn(local_output, output.getvalue())


if __name__ == "__main__":
    unittest.main()
