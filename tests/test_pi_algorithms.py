import io
import math
import unittest
from contextlib import redirect_stdout

from py_no_gil.pi import bbp_pi_parallel, chudnovsky_pi_parallel, machin_pi_parallel


class DeterministicPiAlgorithmTests(unittest.TestCase):
    def test_deterministic_algorithms_approximate_pi(self):
        cases = (
            ("machin", machin_pi_parallel, 24),
            ("chudnovsky", chudnovsky_pi_parallel, 6),
            ("bbp", bbp_pi_parallel, 24),
        )
        for name, algorithm, terms in cases:
            with self.subTest(algorithm=name):
                with redirect_stdout(io.StringIO()):
                    result = algorithm(terms, worker_count=3, precision=50)
                self.assertAlmostEqual(float(result), math.pi, places=14)

    def test_deterministic_algorithms_match_single_and_multiple_workers(self):
        cases = (
            ("machin", machin_pi_parallel, 24),
            ("chudnovsky", chudnovsky_pi_parallel, 6),
            ("bbp", bbp_pi_parallel, 24),
        )
        for name, algorithm, terms in cases:
            with self.subTest(algorithm=name):
                with redirect_stdout(io.StringIO()):
                    single = algorithm(terms, worker_count=1, precision=50)
                    parallel = algorithm(terms, worker_count=4, precision=50)
                self.assertAlmostEqual(float(single), float(parallel), places=14)


if __name__ == "__main__":
    unittest.main()
