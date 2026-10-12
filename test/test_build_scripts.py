"""Independent standard-library tests for build scripts and real PyO3 configuration.

Run with: python3 -m unittest discover -s test -p test_build_scripts.py -v
Shell tests use mocked tools; Rust tests compile an isolated offline Cargo fixture.
"""

import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
REAL_PYTHON = str(Path(sys.executable).resolve())

MOCK_TOOL = r'''import json, os, pathlib, shutil, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ["TOOL_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps({"tool": name, "args": sys.argv[1:],
        "python": os.environ.get("PYO3_PYTHON"),
        "forward": os.environ.get("PYO3_USE_ABI3_FORWARD_COMPATIBILITY")}) + "\n")
if name in ("python3", "chosen python"):
    if os.environ.get("BAD_PYTHON") == "1":
        print("mock interpreter failure", file=sys.stderr)
        sys.exit(17)
    print(os.environ["PROBE_VERSION"])
    print(os.environ["PROBE_LIBDIR"])
    print(os.environ["PROBE_LDLIBRARY"])
elif name == "python":
    print("ERROR: wrong default interpreter", file=sys.stderr)
    sys.exit(19)
elif name == "cargo":
    sys.exit(int(os.environ.get("CARGO_STATUS", "0")))
elif name == "cp":
    if os.environ.get("CP_STATUS"):
        sys.exit(int(os.environ["CP_STATUS"]))
    shutil.copy(sys.argv[1], sys.argv[2])
elif name == "install_name_tool":
    sys.exit(int(os.environ.get("RPATH_STATUS", "0")))
'''

SNAPSHOT = r'''import json, os, sys
print(json.dumps({"environment": dict(os.environ), "flags": sys.argv[1],
    "pipefail": sys.argv[2], "functions": sys.argv[3], "status": int(sys.argv[4])}))
'''


@unittest.skipUnless(BASH, "Bash is required")
class ShellBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scryneuro build tests ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / "project with spaces"
        self.project.mkdir()
        self.bin = self.base / "mock tools"
        self.bin.mkdir()
        self.libdir = self.base / "python libraries"
        self.libdir.mkdir()
        self.log = self.base / "tool log.jsonl"
        self.before = self.base / "before.json"
        self.after = self.base / "after.json"
        for tool in ("python", "python3", "chosen python", "cargo", "cp", "install_name_tool"):
            path = self.bin / tool
            path.write_text("#!" + REAL_PYTHON + "\n" + MOCK_TOOL, encoding="utf-8")
            path.chmod(0o755)
        for platform, suffix in (("linux", "so"), ("macos", "dylib")):
            shutil.copy(ROOT / ("build_" + platform + ".sh"), self.project)
            target = self.project / "target" / "release" / ("libscryneuro." + suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("mock built artifact", encoding="utf-8")
            for version in ("3.13", "3.14"):
                (self.libdir / ("libpython" + version + "." + suffix)).touch()

    def environment(self, platform, version="3.14.7", forward=None, explicit=False):
        env = os.environ.copy()
        for key in ("PYO3_PYTHON", "PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "PYLIB", "SCRYNEURO_HOME"):
            env.pop(key, None)
        suffix = "so" if platform == "linux" else "dylib"
        env.update(PATH=str(self.bin) + os.pathsep + env.get("PATH", ""),
            TOOL_LOG=str(self.log), PROBE_VERSION=version,
            PROBE_LIBDIR=str(self.libdir), PROBE_LDLIBRARY="libpython" + ".".join(version.split(".")[:2]) + "." + suffix,
            LD_LIBRARY_PATH="existing linux path", DYLD_LIBRARY_PATH="existing mac path",
            REAL_PYTHON=REAL_PYTHON, SNAPSHOT=SNAPSHOT)
        if forward is not None:
            env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = forward
        if explicit:
            env["PYO3_PYTHON"] = str(self.bin / "chosen python")
        return env

    def run_source(self, platform, env, options=(False, False, False), conditional=True):
        # Reset artifacts/logs when a matrix invokes several runs in this fixture.
        self.log.write_text("", encoding="utf-8")
        for suffix in ("so", "dylib"):
            (self.project / ("libscryneuro." + suffix)).unlink(missing_ok=True)
        self.after.unlink(missing_ok=True)
        commands = ["set " + ("-" if enabled else "+") + option
                    for option, enabled in zip(("e", "u", "o pipefail"), options)]
        commands += [
            '"$REAL_PYTHON" -c "$SNAPSHOT" "$-" "$(if [[ -o pipefail ]]; then printf on; else printf off; fi)" "$(declare -F)" 0 > "$1"',
            ('if source "$0"; then status=0; else status=$?; fi' if conditional
             else 'source "$0"; status=$?'),
            '"$REAL_PYTHON" -c "$SNAPSHOT" "$-" "$(if [[ -o pipefail ]]; then printf on; else printf off; fi)" "$(declare -F)" "$status" > "$2"',
            'printf "CALLER_SURVIVED\\n"',
            'exit "$status"',
        ]
        result = subprocess.run([BASH, "--noprofile", "--norc", "-c", "\n".join(commands),
            str(self.project / ("build_" + platform + ".sh")), str(self.before), str(self.after)],
            cwd=self.project, env=env, capture_output=True, text=True, timeout=20)
        before = json.loads(self.before.read_text(encoding="utf-8"))
        after = json.loads(self.after.read_text(encoding="utf-8")) if self.after.exists() else None
        tools = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        return result, before, after, tools

    def assert_preserved(self, platform, before, after, success):
        self.assertIsNotNone(after, "sourced script exited its caller")
        self.assertEqual(before["flags"], after["flags"])
        self.assertEqual(before["pipefail"], after["pipefail"])
        self.assertEqual(before["functions"], after["functions"])
        old, new = before["environment"], after["environment"]
        allowed = {"PYLIB", "SCRYNEURO_HOME", "LD_LIBRARY_PATH" if platform == "linux" else "DYLD_LIBRARY_PATH"}
        changes = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
        self.assertEqual(changes, allowed if success else set())
        for key in ("PYO3_PYTHON", "PYO3_USE_ABI3_FORWARD_COMPATIBILITY"):
            self.assertEqual(old.get(key), new.get(key))
        if success:
            self.assertEqual(new["PYLIB"], str(self.libdir))
            self.assertEqual(new["SCRYNEURO_HOME"], str(self.project))
            loader = "LD_LIBRARY_PATH" if platform == "linux" else "DYLD_LIBRARY_PATH"
            self.assertEqual(new[loader], ".:" + str(self.libdir) + ":" + old[loader])

    def test_all_options_versions_flags_interpreters_success_and_build_failure(self):
        for platform, options, version, forward, explicit, fail in itertools.product(
                ("linux", "macos"), itertools.product((False, True), repeat=3),
                ("3.13.5", "3.14.7"), (None, "", "0", "1", "caller-value"),
                (False, True), (False, True)):
            with self.subTest(platform=platform, options=options, version=version,
                              forward=forward, explicit=explicit, failure=fail):
                env = self.environment(platform, version, forward, explicit)
                if fail:
                    env["CARGO_STATUS"] = "23"
                result, before, after, tools = self.run_source(platform, env, options)
                self.assertEqual("e" in before["flags"], options[0])
                self.assertEqual("u" in before["flags"], options[1])
                self.assertEqual(before["pipefail"], "on" if options[2] else "off")
                self.assertEqual(result.returncode, 1 if fail else 0, result.stderr)
                self.assertIn("CALLER_SURVIVED", result.stdout)
                self.assert_preserved(platform, before, after, not fail)
                self.assertEqual(after["status"], result.returncode)
                cargo = [tool for tool in tools if tool["tool"] == "cargo"]
                self.assertEqual(len(cargo), 1)
                expected_python = env.get("PYO3_PYTHON", "python3")
                self.assertEqual(cargo[0]["python"], expected_python)
                self.assertEqual(cargo[0]["args"], ["build", "--release"])
                expected_forward = "1" if forward is None and version.startswith("3.14") else forward
                self.assertEqual(cargo[0]["forward"], expected_forward)
                names = [tool["tool"] for tool in tools]
                self.assertNotIn("python", names)
                suffix = "so" if platform == "linux" else "dylib"
                output = self.project / ("libscryneuro." + suffix)
                self.assertEqual(output.exists(), not fail)
                self.assertEqual(names.count("cp"), 0 if fail else 1)
                for value in (version, str(self.libdir), env["PROBE_LDLIBRARY"], expected_python):
                    self.assertIn(value, result.stdout)
                rpaths = [tool["args"] for tool in tools if tool["tool"] == "install_name_tool"]
                expected_rpaths = [] if platform == "linux" or fail else [
                    ["-add_rpath", "@loader_path/.", "libscryneuro.dylib"],
                    ["-add_rpath", str(self.libdir), "libscryneuro.dylib"]]
                self.assertEqual(rpaths, expected_rpaths)

    def test_bad_interpreter_missing_metadata_library_and_copy_failure(self):
        for platform, case in itertools.product(("linux", "macos"),
                ("nonexistent", "interpreter_failure", "missing_libdir", "missing_ldlibrary", "missing_library", "static_library", "copy_failure")):
            with self.subTest(platform=platform, case=case):
                env = self.environment(platform, forward="caller-value", explicit=True)
                if case == "nonexistent":
                    env["PYO3_PYTHON"] = str(self.bin / "does not exist")
                elif case == "interpreter_failure":
                    env["BAD_PYTHON"] = "1"
                elif case == "missing_libdir":
                    env["PROBE_LIBDIR"] = ""
                elif case == "missing_ldlibrary":
                    env["PROBE_LDLIBRARY"] = "None"
                elif case == "missing_library":
                    env["PROBE_LDLIBRARY"] = "missing shared library"
                elif case == "static_library":
                    env["PROBE_LDLIBRARY"] = "libpython3.14.a"
                    (self.libdir / env["PROBE_LDLIBRARY"]).touch()
                else:
                    env["CP_STATUS"] = "29"
                result, before, after, tools = self.run_source(platform, env, (True, True, True))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CALLER_SURVIVED", result.stdout)
                self.assert_preserved(platform, before, after, False)
                self.assertNotIn("Build complete", result.stdout)
                if case != "copy_failure":
                    self.assertIn("ERROR:", result.stderr)
                    self.assertFalse(any(tool["tool"] == "cargo" for tool in tools))
                self.assertFalse(any(tool["tool"] == "install_name_tool" for tool in tools))

    def test_plain_source_success_with_every_option_combination(self):
        for platform, options in itertools.product(("linux", "macos"), itertools.product((False, True), repeat=3)):
            with self.subTest(platform=platform, options=options):
                result, before, after, _ = self.run_source(platform, self.environment(platform), options, conditional=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_preserved(platform, before, after, True)

    def test_unexported_caller_pyo3_settings_remain_unexported(self):
        for platform in ("linux", "macos"):
            with self.subTest(platform=platform):
                env = self.environment(platform)
                commands = ('PYO3_PYTHON="$1"\nPYO3_USE_ABI3_FORWARD_COMPATIBILITY=caller-local\n'
                    'source "$0"\ndeclare -p PYO3_PYTHON PYO3_USE_ABI3_FORWARD_COMPATIBILITY\n')
                result = subprocess.run([BASH, "--noprofile", "--norc", "-euo", "pipefail", "-c", commands,
                    str(self.project / ("build_" + platform + ".sh")), str(self.bin / "chosen python")],
                    cwd=self.project, env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('declare -- PYO3_PYTHON=', result.stdout)
                self.assertIn('declare -- PYO3_USE_ABI3_FORWARD_COMPATIBILITY="caller-local"', result.stdout)
                tools = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
                cargo = [tool for tool in tools if tool["tool"] == "cargo"][-1]
                self.assertEqual(cargo["python"], str(self.bin / "chosen python"))
                self.assertEqual(cargo["forward"], "caller-local")

    def test_plain_source_returns_failure_and_respects_caller_errexit(self):
        for platform, errexit in itertools.product(("linux", "macos"), (False, True)):
            with self.subTest(platform=platform, errexit=errexit):
                env = self.environment(platform)
                env["CARGO_STATUS"] = "23"
                result, before, after, _ = self.run_source(platform, env, (errexit, True, True), conditional=False)
                self.assertNotEqual(result.returncode, 0)
                if errexit:
                    self.assertIsNone(after)
                    self.assertNotIn("CALLER_SURVIVED", result.stdout)
                else:
                    self.assert_preserved(platform, before, after, False)
                    self.assertIn("CALLER_SURVIVED", result.stdout)

    def test_executed_script_failure_and_tolerated_macos_rpath_duplicates(self):
        for platform in ("linux", "macos"):
            env = self.environment(platform)
            env["CARGO_STATUS"] = "23"
            result = subprocess.run([BASH, str(self.project / ("build_" + platform + ".sh"))],
                cwd=self.project, env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Build complete", result.stdout)
        env = self.environment("macos")
        env["RPATH_STATUS"] = "1"
        result, before, after, tools = self.run_source("macos", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_preserved("macos", before, after, True)
        self.assertEqual(sum(tool["tool"] == "install_name_tool" for tool in tools), 2)


@unittest.skipUnless(shutil.which("cargo"), "Cargo is required for real build.rs configuration checks")
class RustBuildConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="scryneuro rust build tests ")
        cls.project = Path(cls.temporary.name)
        (cls.project / "src").mkdir()
        manifest = ('[package]\nname = "scryneuro-build-script-test"\nversion = "0.0.0"\nedition = "2021"\n'
            'build = ' + json.dumps(str(ROOT / "build.rs")) + '\n'
            '[build-dependencies]\npyo3-build-config = { version = "0.23", features = ["resolve-config"] }\n')
        (cls.project / "Cargo.toml").write_text(manifest, encoding="utf-8")
        (cls.project / "src" / "main.rs").write_text(
            'fn main() { println!("LIBRARY={}", env!("SCRYNEURO_PYTHON_LIBRARY")); '
            'println!("LDLIBRARY={}", env!("SCRYNEURO_PYTHON_LDLIBRARY")); }\n', encoding="utf-8")
        cls.env = os.environ.copy()
        for key in list(cls.env):
            if key.startswith("PYO3_"):
                cls.env.pop(key)
        cls.env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
        cls.env["PYO3_PYTHON"] = REAL_PYTHON
        cls.cargo = shutil.which("cargo")
        cls.addClassCleanup(cls.temporary.cleanup)

    def run_cargo(self, env):
        return subprocess.run([self.cargo, "run", "--offline"], cwd=self.project,
            env=env, capture_output=True, text=True, timeout=120)

    def test_explicit_real_interpreter_matches_real_sysconfig(self):
        result = self.run_cargo(self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        libdir, library = sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")
        self.assertIn("LIBRARY=" + str(Path(libdir) / library), result.stdout)
        self.assertIn("LDLIBRARY=" + library, result.stdout)
        self.assertIn("interpreter=" + REAL_PYTHON, result.stderr)
        self.assertIn("LIBDIR=" + libdir, result.stderr)
        self.assertIn("LDLIBRARY=" + library, result.stderr)

    def test_default_pyo3_python_priority_over_python3(self):
        bin_dir = self.project / "default interpreter tools"
        bin_dir.mkdir(exist_ok=True)
        # PyO3's own default discovery prefers python. Its configuration retains
        # that selected executable, which build.rs must use rather than python3.
        (bin_dir / "python").symlink_to(REAL_PYTHON)
        wrong = bin_dir / "python3"
        wrong.write_text("#!/bin/sh\necho wrong-default-python3 >&2\nexit 77\n", encoding="utf-8")
        wrong.chmod(0o755)
        env = self.env.copy()
        env.pop("PYO3_PYTHON")
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        result = self.run_cargo(env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("wrong-default-python3", result.stderr)
        self.assertIn("LDLIBRARY=" + sysconfig.get_config_var("LDLIBRARY"), result.stdout)
        self.assertIn("interpreter=" + str(bin_dir / "python"), result.stderr)

    def test_invalid_explicit_interpreter_fails(self):
        env = self.env.copy()
        env["PYO3_PYTHON"] = str(self.project / "nonexistent python")
        result = self.run_cargo(env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonexistent python", result.stderr)

    def test_real_config_file_version_mismatch_is_rejected(self):
        config = self.project / "wrong version config.txt"
        config.write_text("implementation=CPython\nversion=3.12\nshared=true\nabi3=false\n"
            "lib_name=python3.12\nlib_dir=" + str(sysconfig.get_config_var("LIBDIR"))
            + "\nexecutable=" + REAL_PYTHON + "\n", encoding="utf-8")
        env = self.env.copy()
        env["PYO3_CONFIG_FILE"] = str(config)
        result = self.run_cargo(env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python version differs from PyO3's actual build configuration", result.stderr)

    def test_real_config_file_missing_library_is_rejected(self):
        fake_python = self.project / "configured python"
        fake_python.write_text("#!" + REAL_PYTHON + "\n"
            'print("3.13.5")\nprint(' + repr(str(self.project)) + ')\nprint("libpython missing.so")\n', encoding="utf-8")
        fake_python.chmod(0o755)
        config = self.project / "pyo3 config.txt"
        config.write_text("implementation=CPython\nversion=3.13\nshared=true\nabi3=false\n"
            "lib_name=python3.13\nlib_dir=" + str(self.project) + "\nexecutable=" + str(fake_python) + "\n",
            encoding="utf-8")
        env = self.env.copy()
        env["PYO3_CONFIG_FILE"] = str(config)
        result = self.run_cargo(env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Required Python shared library does not exist", result.stderr)
        self.assertIn("libpython missing.so", result.stderr)


if __name__ == "__main__":
    unittest.main()
