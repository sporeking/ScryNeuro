"""Readable error output through the real Scryer module and entry points.

Run from the README runtime environment: python3 test/test_error_output.py -v
The existing bridge must match the selected Python runtime; no training is run.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ErrorOutputTests(unittest.TestCase):
    def run_scryer(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["SCRYNEURO_HOME"] = str(ROOT)
        return subprocess.run(
            ["scryer-prolog", "-f", "--no-add-history", *args],
            cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", timeout=15,
        )

    def run_goal(self, goal: str) -> str:
        result = self.run_scryer(
            "-g",
            "use_module('prolog/scryer_py'), "
            "use_module(library(iso_ext)), use_module(library(format)), "
            f"catch((({goal}) -> halt(0); halt(1)), TestFailure, "
            "(print_py_error(TestFailure), halt(1))).",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertNotIn("causes:", result.stdout)
        return result.stdout

    def test_multiline_python_text_preserves_input(self):
        output = self.run_goal(r'''
            Msg = "ValueError: 模型不存在\n第二行: \"weights.pt\"\\cache",
            Context = py_eval/2,
            Error = error(python_error(Msg), Context),
            copy_term(Error, Before),
            print_py_error(Error), Error == Before
        ''')
        self.assertEqual(
            output,
            'Python error (py_eval/2):\n'
            'ValueError: 模型不存在\n第二行: "weights.pt"\\cache\n',
        )

    def test_empty_and_atom_messages(self):
        output = self.run_goal('''
            print_py_error(error(python_error([]), demo/0)),
            print_py_error(error(python_error('legacy atom message'), demo/0))
        ''')
        self.assertEqual(
            output,
            "Python error (demo/0):\n\n"
            "Python error (demo/0):\nlegacy atom message\n",
        )

    def test_foreign_library_path_is_text(self):
        output = self.run_goal('''
            print_py_error(error(foreign_library_load_error("/模型/lib.so"), load_ffi/1))
        ''')
        self.assertEqual(output, "Foreign library could not be loaded: /模型/lib.so\n")

    def test_generic_exception_keeps_structure_and_compact_strings(self):
        output = self.run_goal('''
            print_py_error(error(existence_error(source_sink, "/models/weights.pt"), load_ffi/1)),
            print_py_error(error(payload([1,2,3], [a,42,word]), demo/0))
        ''')
        self.assertRegex(
            output,
            r'Error: error\(existence_error\(source_sink,\s*"/models/weights.pt"\),\s*load_ffi/1\)',
        )
        self.assertRegex(output, r'payload\(\[1,\s*2,\s*3\],\s*\[a,\s*42,\s*word\]\)')

    def test_partial_and_variable_payloads_are_not_instantiated(self):
        output = self.run_goal('''
            Error = error(python_error([a|Tail]), context(Context)),
            print_py_error(Error), var(Tail), var(Context),
            print_py_error(error(python_error([a,42]), demo/0)),
            print_py_error(error(Kind, demo/0)), var(Kind),
            print_py_error(Unbound), var(Unbound)
        ''')
        self.assertIn("error(python_error([a|", output)
        self.assertRegex(output, r'python_error\(\[a,\s*42\]\)')
        self.assertIn("context(", output)

    def test_real_bridge_exception_is_caught_silent_and_cleans_up(self):
        output = self.run_goal('''
            setup_call_cleanup(py_init, (
                py_handle_count(Before),
                catch((once(with_py_temp(py_eval("42", H), H, py_eval("1/0", _))),
                       Outcome = success), Error, Outcome = exception(Error)),
                Outcome = exception(error(python_error(Msg), Context)),
                Msg = [_|_], ground(Msg), atom_chars(_, Msg), Context == py_eval/2,
                py_handle_count(After), After =:= Before,
                format("caught; handles restored~n", []),
                print_py_error(Error)
            ), scryer_py:py_finalize)
        ''')
        # Nothing was automatically printed before the caller chose to display it.
        self.assertTrue(output.startswith("caught; handles restored\nPython error (py_eval/2):\n"))
        self.assertIn("ZeroDivisionError", output)
        self.assertIn("Traceback", output)
        self.assertGreaterEqual(output.count("\n"), 4)

    def test_benchmark_final_error_still_exits_nonzero(self):
        result = self.run_scryer("benchmark/bench_ffi.pl", "--", "invalid")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("FFI benchmark failed: Error: error(", result.stdout)
        self.assertRegex(result.stdout, r'domain_error\(positive_integer,\s*"invalid"\)')
        self.assertIn("iteration_count/1", result.stdout)

    def test_mnist_final_error_exits_before_loading_training(self):
        # A missing library forces py_init to throw before any Python/training call.
        env = os.environ.copy()
        env["SCRYNEURO_HOME"] = str(ROOT / "missing-error-output-test-home")
        result = subprocess.run(
            ["scryer-prolog", "-f", "--no-add-history", "benchmark/bench_mnist_cnn_ffi.pl"],
            cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("MNIST CNN benchmark failed: Python error (py_init/0):\n", result.stdout)
        self.assertIn("Could not find libscryneuro in directory:", result.stdout)
        self.assertIn(env["SCRYNEURO_HOME"], result.stdout)
        self.assertNotIn("Training", result.stdout)


if __name__ == "__main__":
    unittest.main()
