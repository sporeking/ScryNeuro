# Windows support

ScryNeuro supports a native 64-bit Windows build using Rust, PyO3, CPython,
and a Scryer Prolog executable built with `library(ffi)` support.

## Supported baseline

- Windows 10 or newer, x64
- Rust stable, with either the `x86_64-pc-windows-msvc` or
  `x86_64-pc-windows-gnu` target
- CPython 3.10 through 3.13, x64, including its import library
- Scryer Prolog 0.10.0 or newer for Windows, with `library(ffi)` support

All four components must use the same architecture. The produced library is
`scryneuro.dll`; Windows does not add the Unix `lib` prefix.

The Windows port was validated against the official Scryer Prolog 0.10.0 x64
build (`--version` reports `e7ac3ae`). Its executable SHA-256 is
`47217C611372AF87D79017983CBBC8A12E47390EF4F64D8FDE287142B58E5602`.
Older Windows builds must pass `test/test_windows_ffi_abi.pl` before use;
ScryNeuro depends on correct pointer and C-string FFI results.

## Build

Pass the Python executable that ScryNeuro should embed:

```powershell
.\build_windows.ps1 -Python C:\path\to\python.exe
```

The script validates Python's bitness, runtime DLL, and import library, sets
`PYO3_PYTHON`, runs `cargo build --release`, and copies the resulting
`target\release\scryneuro.dll` to the repository root.

You can also configure `PYO3_PYTHON` and omit `-Python`:

```powershell
$env:PYO3_PYTHON = 'C:\path\to\python.exe'
.\build_windows.ps1
```

## Test

Run the Windows acceptance test with explicit tool paths:

```powershell
.\test_windows.ps1 `
  -Python C:\path\to\python.exe `
  -Scryer C:\path\to\scryer-prolog.exe `
  -IncludeExistingSuites
```

The acceptance test covers:

- DLL discovery through `SCRYNEURO_HOME`
- Python initialization, finalization, and reinitialization
- `ptr`, `sint64`, `f64`, and UTF-8 `cstr` values
- both C-string arguments and C-string return values in Scryer's Windows FFI
- JSON with non-ASCII text and a 64-bit integer
- more than 255 simultaneous Python object handles
- Python exception propagation and complete handle cleanup
- public three-argument method calls and finalization with a live handle
- loading the standard-library native `_sqlite3.pyd` extension
- 1,000 repeated evaluation/free cycles
- a Windows path containing spaces and non-ASCII characters
- Python module discovery under `<SCRYNEURO_HOME>\python` from that path
- the existing smoke, Pi, 26-case low-level FFI, minimal API, and 32-case
  Prolog API suites when
  `-IncludeExistingSuites` is supplied

The script prepends the selected Python installation to `PATH` so Windows can
find `python3xx.dll`. It restores the caller's environment afterward.
When `-Python` points into a virtual environment, the test script also adds
that environment's `Lib\site-packages` to `PYTHONPATH`.

## Cross-project use

Set `SCRYNEURO_HOME` to the directory containing `scryneuro.dll`, and ensure
the matching CPython directory is on `PATH`:

```powershell
$env:SCRYNEURO_HOME = 'C:\path\to\ScryNeuro'
$env:PYTHONHOME = 'C:\path\to\Python311'
$env:PATH = 'C:\path\to\Python311;C:\path\to\Python311\DLLs;' + $env:PATH
scryer-prolog your_program.pl
```

For a virtual environment, also expose its packages before starting Scryer:

```powershell
$env:PYTHONPATH = 'C:\path\to\venv\Lib\site-packages;' + $env:PYTHONPATH
```

The Prolog loader searches for `scryneuro.dll` on Windows while retaining the
existing `.dylib` and `.so` discovery paths for macOS and Linux.

## Troubleshooting

- `Could not find scryneuro.dll`: check `SCRYNEURO_HOME` or run from the
  ScryNeuro directory.
- DLL load failure: verify that Scryer, ScryNeuro, and Python are all x64 and
  that the matching `python3xx.dll` directory is on `PATH`.
- Python import failures: rebuild after changing Python minor versions and
  install the required package into the exact Python selected by
  `PYO3_PYTHON`.
