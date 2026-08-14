import io
import unittest
from unittest.mock import patch

from py_no_gil.nbody import (
    DEFAULT_PARTICLES,
    DEFAULT_STEPS,
    _compute_force_on_particle,
    _make_initial_particles,
    parse_args,
    run_parallel_nbody,
    search_nbody_parallel,
    simulate,
)


class IsNBodyTests(unittest.TestCase):
    def test_compute_force_on_particle_basic(self):
        particles = ((0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 1.0))
        fx, fy = _compute_force_on_particle(0, particles, G=1.0, n=2)
        self.assertGreater(fx, 0.0)
        self.assertAlmostEqual(fy, 0.0, places=10)

    def test_compute_force_self_ignored(self):
        particles = (
            (0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0, 0.0, 1.0),
        )
        fx, fy = _compute_force_on_particle(0, particles, G=1.0, n=3)
        self.assertAlmostEqual(fx, 0.0, places=10)
        self.assertAlmostEqual(fy, 0.0, places=10)


class SimulateTests(unittest.TestCase):
    def test_simulate_preserves_particle_structure(self):
        particles = _make_initial_particles(10)
        result = simulate(particles, G=0.1, dt=0.01, steps=3, n=10, worker_count=1)
        self.assertEqual(len(result), 10)
        for particle in result:
            self.assertEqual(len(particle), 5)


class SearchNBodyParallelTests(unittest.TestCase):
    def test_count_matches_single_threaded_result(self):
        particles = _make_initial_particles(50)
        kwargs = {
            "particles": particles,
            "G": 0.1,
            "dt": 0.01,
            "steps": 5,
            "mode": "count",
            "n": 50,
        }
        self.assertEqual(search_nbody_parallel(worker_count=1, **kwargs), 1)
        self.assertEqual(search_nbody_parallel(worker_count=4, **kwargs), 1)

    def test_list_matches_single_threaded_result(self):
        particles = _make_initial_particles(30)
        kwargs = {
            "particles": particles,
            "G": 0.1,
            "dt": 0.01,
            "steps": 5,
            "mode": "list",
            "n": 30,
        }
        single = search_nbody_parallel(worker_count=1, **kwargs)
        split = search_nbody_parallel(worker_count=3, **kwargs)
        self.assertEqual(len(single), 30)
        self.assertEqual(len(split), 30)
        self.assertEqual(single, split)


class ParseNBodyArgsTests(unittest.TestCase):
    def test_count_defaults(self):
        args = parse_args(["count"])
        self.assertEqual(args.command, "count")
        self.assertEqual(args.particles, DEFAULT_PARTICLES)
        self.assertEqual(args.steps, DEFAULT_STEPS)
        self.assertFalse(args.compare_gil)

    def test_list_accepts_range_and_compare_flag(self):
        args = parse_args(
            ["list", "--particles", "50", "--steps", "200", "--compare-gil"]
        )
        self.assertEqual(args.command, "list")
        self.assertEqual(args.particles, 50)
        self.assertEqual(args.steps, 200)
        self.assertTrue(args.compare_gil)


class RunParallelNBodyTests(unittest.TestCase):
    def test_count_prints_result_and_execution_time(self):
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_nbody(["count", "--particles", "50", "--steps", "50"])

        output = buffer.getvalue()
        self.assertIn("Simulation count: 1", output)
        self.assertIn("Execution time:", output)

    def test_list_prints_execution_time(self):
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            run_parallel_nbody(["list", "--particles", "20", "--steps", "20"])

        output = buffer.getvalue()
        self.assertIn("Execution time:", output)
        self.assertIn("Final particle count:", output)


if __name__ == "__main__":
    unittest.main()
