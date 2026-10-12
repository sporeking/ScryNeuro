"""C ABI lifecycle regressions; each callback case has a process timeout.

Run after building: python3 test/test_registry_lifecycle.py -v
The test interpreter must match the Python used to build the bridge.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[1]


def child(scenario: str) -> None:
    global _spy_lib, _events, _weakref
    suffix = "dylib" if sys.platform == "darwin" else "so"
    _spy_lib = ctypes.PyDLL(str(ROOT / f"libscryneuro.{suffix}"))
    for name in ("spy_init", "spy_init_home", "spy_exec", "spy_to_bool", "spy_is_none"):
        getattr(_spy_lib, name).restype = ctypes.c_int32
    for name in ("spy_eval", "spy_from_int", "spy_call0"):
        getattr(_spy_lib, name).restype = ctypes.c_ssize_t
    _spy_lib.spy_eval.argtypes = [ctypes.c_char_p]
    _spy_lib.spy_exec.argtypes = [ctypes.c_char_p]
    _spy_lib.spy_init_home.argtypes = [ctypes.c_char_p]
    _spy_lib.spy_from_int.argtypes = [ctypes.c_int64]
    _spy_lib.spy_to_int.argtypes = [ctypes.c_ssize_t]
    _spy_lib.spy_to_int.restype = ctypes.c_int64
    _spy_lib.spy_call0.argtypes = [ctypes.c_ssize_t]
    _spy_lib.spy_drop.argtypes = [ctypes.c_ssize_t]
    _spy_lib.spy_drop.restype = None
    _spy_lib.spy_finalize.restype = None
    _spy_lib.spy_handle_count.restype = ctypes.c_int64
    _spy_lib.spy_last_error.restype = ctypes.c_char_p
    assert _spy_lib.spy_init() == 0
    assert _spy_lib.spy_handle_count() == 0
    if scenario == "repeat_init":
        old = _spy_lib.spy_from_int(7)
        assert old > 0
        path_before = list(sys.path)
        for _ in range(5):
            assert _spy_lib.spy_init() == 0
            assert _spy_lib.spy_handle_count() == 1
            assert _spy_lib.spy_to_int(old) == 7
        assert sys.path == path_before
        assert _spy_lib.spy_init_home(os.fsencode(ROOT)) == 0
        assert _spy_lib.spy_handle_count() == 1
        assert _spy_lib.spy_to_int(old) == 7
        _spy_lib.spy_finalize()
        assert _spy_lib.spy_handle_count() == -1
        assert _spy_lib.spy_init() == 0
        new = _spy_lib.spy_from_int(99)
        assert new > old
        assert _spy_lib.spy_to_int(old) == 0
        assert b"Invalid handle" in _spy_lib.spy_last_error()
        assert _spy_lib.spy_to_int(new) == 99
        assert _spy_lib.spy_last_error() is None
        _spy_lib.spy_drop(new)
        assert _spy_lib.spy_to_int(new) == 0
        assert _spy_lib.spy_handle_count() == 0
        assert _spy_lib.spy_last_error() is None
    elif scenario in ("drop_callback", "finalize_callback", "weakref_callback"):
        _events = []
        _weakref = weakref
        if scenario == "weakref_callback":
            code = """
class _CallbackObject:
    pass
def _make_callback_object():
    obj = _CallbackObject()
    global _callback_weakref
    _callback_weakref = _weakref.ref(obj, lambda ref: _events.append(_spy_lib.spy_handle_count()))
    return obj
"""
        else:
            code = """
class _CallbackObject:
    def __del__(self):
        _events.append(_spy_lib.spy_handle_count())
def _make_callback_object():
    return _CallbackObject()
"""
        assert _spy_lib.spy_exec(code.encode()) == 0
        handle = _spy_lib.spy_eval(b"_make_callback_object()")
        assert handle > 0
        assert _spy_lib.spy_handle_count() == 1
        if scenario == "drop_callback":
            _spy_lib.spy_drop(handle)
            assert _events == [0], _events
            assert _spy_lib.spy_handle_count() == 0
        else:
            _spy_lib.spy_finalize()
            assert _events == [-1], _events
            assert _spy_lib.spy_init() == 0
            assert _spy_lib.spy_handle_count() == 0
            assert _spy_lib.spy_to_int(handle) == 0
            assert b"Invalid handle" in _spy_lib.spy_last_error()
    elif scenario == "method_callback":
        _events = []
        handle = _spy_lib.spy_eval(b"lambda: _spy_lib.spy_handle_count()")
        assert handle > 0
        result = _spy_lib.spy_call0(handle)
        assert result > handle
        assert _spy_lib.spy_to_int(result) == 1
        assert _spy_lib.spy_handle_count() == 2
        _spy_lib.spy_drop(result)
        _spy_lib.spy_drop(handle)
        assert _spy_lib.spy_handle_count() == 0
    else:
        raise AssertionError(f"Unknown scenario: {scenario}")
    _spy_lib.spy_finalize()


class RegistryLifecycleTests(unittest.TestCase):
    def check_child(self, scenario: str) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", scenario],
            cwd=ROOT, capture_output=True, text=True, timeout=8,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_repeat_init_and_stale_handles(self):
        self.check_child("repeat_init")

    def test_drop_callback_reentry(self):
        self.check_child("drop_callback")

    def test_finalize_callback_reentry(self):
        self.check_child("finalize_callback")

    def test_weakref_callback_reentry(self):
        self.check_child("weakref_callback")

    def test_method_callback_reentry(self):
        self.check_child("method_callback")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        unittest.main()
