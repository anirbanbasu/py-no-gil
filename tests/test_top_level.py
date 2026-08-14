import io
import unittest
from unittest.mock import patch

from py_no_gil import main


class TopLevelCommandTests(unittest.TestCase):
    def test_default_output_explains_common_benchmark_options(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            main([])
        text = output.getvalue()
        self.assertIn("--warmup", text)
        self.assertIn("--repeat", text)
        self.assertIn("--json", text)
        self.assertIn("scale", text)

    def test_dispatches_to_primes_command(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            main(
                [
                    "primes",
                    "count",
                    "--start",
                    "1",
                    "--stop",
                    "20",
                    "--repeat",
                    "1",
                    "--warmup",
                    "1",
                ]
            )
        self.assertIn("Prime count: 8", output.getvalue())


if __name__ == "__main__":
    unittest.main()
