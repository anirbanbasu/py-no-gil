import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from py_no_gil.fractal import (
    color_from_iterations,
    escape_iterations,
    format_ansi_preview,
    parse_args,
    pixel_complex,
    render_fractal,
    run_parallel_fractal,
    write_ppm,
)


class EscapeIterationsTests(unittest.TestCase):
    def test_origin_stays_in_the_mandelbrot_set(self):
        self.assertEqual(escape_iterations(0.0, 0.0, 0.0, 0.0, max_iter=50), 50)

    def test_far_away_point_escapes_on_the_first_check(self):
        self.assertEqual(escape_iterations(10.0, 0.0, -0.8, 0.156, max_iter=50), 0)

    def test_point_one_plus_zero_escapes_after_a_few_iterations(self):
        self.assertEqual(escape_iterations(0.0, 0.0, 1.0, 0.0, max_iter=50), 3)


class PixelComplexTests(unittest.TestCase):
    def test_maps_pixel_centers_into_the_complex_plane(self):
        top_left = pixel_complex(
            0, 0, width=2, height=2, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0
        )
        bottom_right = pixel_complex(
            1, 1, width=2, height=2, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0
        )
        self.assertEqual(top_left, (-0.5, 0.5))
        self.assertEqual(bottom_right, (0.5, -0.5))


class ColorFromIterationsTests(unittest.TestCase):
    def test_points_that_never_escape_are_black(self):
        self.assertEqual(color_from_iterations(50, max_iter=50), (0, 0, 0))

    def test_escaped_points_are_not_black(self):
        red, green, blue = color_from_iterations(0, max_iter=50)
        self.assertNotEqual((red, green, blue), (0, 0, 0))
        self.assertTrue(0 <= red <= 255)
        self.assertTrue(0 <= green <= 255)
        self.assertTrue(0 <= blue <= 255)


class WritePpmTests(unittest.TestCase):
    def test_writes_p6_header_and_rgb_bytes(self):
        pixels = [(1, 2, 3), (4, 5, 6)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny.ppm"
            write_ppm(path, width=2, height=1, pixels=pixels)
            data = path.read_bytes()

        self.assertTrue(data.startswith(b"P6\n2 1\n255\n"))
        self.assertTrue(data.endswith(b"\x01\x02\x03\x04\x05\x06"))


class FormatAnsiPreviewTests(unittest.TestCase):
    def test_uses_half_blocks_with_truecolor_for_top_and_bottom_pixels(self):
        pixels = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]
        preview = format_ansi_preview(pixels, width=2, height=2, preview_width=2)

        self.assertIn("\x1b[38;2;255;0;0m\x1b[48;2;0;0;255m▀", preview)
        self.assertIn("\x1b[38;2;0;255;0m\x1b[48;2;255;255;0m▀", preview)
        self.assertTrue(preview.endswith("\x1b[0m"))


class RenderFractalTests(unittest.TestCase):
    def test_mandelbrot_center_pixel_is_black(self):
        pixels = render_fractal(
            width=1,
            height=1,
            max_iter=50,
            worker_count=1,
            kind="mandelbrot",
            xmin=-0.1,
            xmax=0.1,
            ymin=-0.1,
            ymax=0.1,
        )
        self.assertEqual(pixels, [(0, 0, 0)])

    def test_julia_origin_with_c_zero_is_black(self):
        pixels = render_fractal(
            width=1,
            height=1,
            max_iter=50,
            worker_count=1,
            kind="julia",
            xmin=-0.1,
            xmax=0.1,
            ymin=-0.1,
            ymax=0.1,
            c_real=0.0,
            c_imag=0.0,
        )
        self.assertEqual(pixels, [(0, 0, 0)])

    def test_splitting_rows_across_workers_matches_a_single_worker(self):
        kwargs = {
            "width": 4,
            "height": 4,
            "max_iter": 20,
            "kind": "mandelbrot",
            "xmin": -2.0,
            "xmax": 1.0,
            "ymin": -1.5,
            "ymax": 1.5,
        }
        single = render_fractal(worker_count=1, **kwargs)
        split = render_fractal(worker_count=3, **kwargs)
        self.assertEqual(len(single), 16)
        self.assertEqual(single, split)


class ParseFractalArgsTests(unittest.TestCase):
    def test_mandelbrot_defaults(self):
        args = parse_args(["mandelbrot"])
        self.assertEqual(args.kind, "mandelbrot")
        self.assertEqual(args.width, 800)
        self.assertEqual(args.height, 600)
        self.assertEqual(args.iter, 256)
        self.assertEqual(args.output, "mandelbrot.ppm")
        self.assertEqual(args.preview_width, 80)
        self.assertFalse(args.compare_gil)

    def test_julia_accepts_c_and_compare_flag(self):
        args = parse_args(
            ["julia", "--c-real", "-0.4", "--c-imag", "0.6", "--compare-gil"]
        )
        self.assertEqual(args.kind, "julia")
        self.assertEqual(args.c_real, -0.4)
        self.assertEqual(args.c_imag, 0.6)
        self.assertEqual(args.output, "julia.ppm")
        self.assertTrue(args.compare_gil)

    def test_iter_accepts_custom_iteration_limit(self):
        args = parse_args(["mandelbrot", "--iter", "512"])
        self.assertEqual(args.iter, 512)


class RunParallelFractalTests(unittest.TestCase):
    def test_writes_ppm_prints_preview_and_execution_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "out.ppm"
            buffer = io.StringIO()
            with patch("sys.stdout", buffer):
                run_parallel_fractal(
                    [
                        "mandelbrot",
                        "--width",
                        "8",
                        "--height",
                        "6",
                        "--iter",
                        "20",
                        "--workers",
                        "2",
                        "--output",
                        str(output_path),
                    ]
                )

            output = buffer.getvalue()
            data = output_path.read_bytes()

        self.assertTrue(data.startswith(b"P6\n8 6\n255\n"))
        self.assertEqual(len(data.split(b"\n", 3)[-1]), 8 * 6 * 3)
        self.assertIn("Execution time:", output)
        self.assertIn("\x1b[38;2;", output)
        self.assertIn(str(output_path), output)

    def test_compare_flag_runs_fractal_module_with_gil_on_then_off(self):
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
            run_parallel_fractal(
                ["julia", "--compare-gil", "--width", "8"], runner=fake_runner
            )

        output = buffer.getvalue()
        self.assertEqual([call["gil"] for call in calls], ["1", "0"])
        for call in calls:
            self.assertEqual(
                call["cmd"][:3], [sys.executable, "-m", "py_no_gil.fractal"]
            )
            self.assertNotIn("--compare-gil", call["cmd"])
        self.assertIn("4.00x", output)
        self.assertNotIn("\x1b[38;2;", output)
