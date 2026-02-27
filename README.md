# ScryNeuro

ScryNeuro is a high-performance bridge between **Scryer Prolog** and **Python**, designed for **neuro-symbolic AI** research. It enables Scryer Prolog programs to seamlessly call Python neural components — LLMs, deep neural networks, reinforcement learning agents, NumPy, PyTorch — while preserving Prolog's logical reasoning capabilities.

Inspired by [Jurassic.pl](https://github.com/haldai/Jurassic.pl) (SWI-Prolog ↔ Julia bridge).

## Architecture

```text
[ Scryer Prolog ] <-> [ Rust cdylib (FFI) ] <-> [ PyO3 ] <-> [ Python Runtime ]
      (Logic)            (Bridge Layer)          (Glue)        (Neural/Perception)
```

- **Scryer Prolog** — logical reasoning and top-level control flow.
- **Rust cdylib** (`libscryneuro.so` / `.dylib`) — FFI bridge with handle-based object registry.
- **PyO3** — embeds Python within Rust; manages GIL and type conversions.
- **Python** — executes neural predicates, data processing, library calls (PyTorch, NumPy, OpenAI, etc.).

---

## Installation

### System Requirements

| Component | Version | Notes |
|---|---|---|
| **Rust** | stable ≥ 1.70 | `rustup` recommended |
| **Python** | 3.10 – 3.13 | with shared library (`libpython3.x.so` / `.dylib`) |
| **Scryer Prolog** | latest git | must support `library(ffi)` |

### Step 1: Install Rust

```bash
# Install rustup (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify
rustc --version
cargo --version
```

### Step 2: Install Scryer Prolog

```bash
# Build from source (requires Rust)
git clone https://github.com/mthom/scryer-prolog.git
cd scryer-prolog
cargo install --path .

# Verify
scryer-prolog --version
```

### Step 3: Set Up Python Environment

ScryNeuro links against whatever `python3` is active at **build time**, and loads `libpython3.x.so` at **runtime**. Both must match.

#### Option A: Conda (Recommended)

```bash
# Create a dedicated environment
conda create -n scryneuro python=3.12 numpy -y
conda activate scryneuro

# Install ML libraries as needed
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  # GPU
# OR
conda install pytorch torchvision torchaudio cpuonly -c pytorch  # CPU only

# Verify shared library exists
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
# Should print something like: /home/user/miniconda3/envs/scryneuro/lib
ls $(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3*.so*
```

#### Option B: uv

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project venv
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install numpy torch

# Verify
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
```

#### Option C: System Python

```bash
# Debian/Ubuntu
sudo apt install python3-dev python3-numpy

# Fedora
sudo dnf install python3-devel python3-numpy

# macOS (Homebrew)
brew install python@3.12 numpy
```

> **Critical**: Python must be built with shared library support. Conda and system packages always have this. If using `pyenv`, build with: `PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.12`

### Step 4: Build ScryNeuro

```bash
git clone <repo-url> ScryNeuro
cd ScryNeuro

# Activate your Python environment first!
conda activate scryneuro  # or: source .venv/bin/activate

# Build
cargo build --release

# Copy the shared library to project root
cp target/release/libscryneuro.so ./     # Linux
# cp target/release/libscryneuro.dylib ./  # macOS
```

The build output should show `Building with Python 3.12.x` (matching your active environment).

### Step 5: Verify

```bash
# Linux
LD_LIBRARY_PATH=. scryer-prolog examples/basic.pl

# macOS
DYLD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
```

Expected output:
```
=== Arithmetic ===
2^10 = 1024
sum(0..99) = 4950
...
=== All basic examples complete ===
```

---

## Platform-Specific Notes

### Linux

The `RTLD_GLOBAL` mechanism in `spy_init()` automatically re-opens `libpython3.x.so` with global symbol visibility, which is required for C extensions like NumPy and PyTorch. You must ensure `libpython` is discoverable:

```bash
# Option 1: Set LD_LIBRARY_PATH (recommended for conda)
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH

# Option 2: Use LD_PRELOAD (if RTLD_GLOBAL auto-detection fails)
export LD_PRELOAD=$(python3 -c "import sysconfig, os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), 'libpython3.12.so'))")
```

**Convenience wrapper** — create a `run.sh`:
```bash
#!/bin/bash
# Activate conda env and run a Prolog script with ScryNeuro
eval "$(conda shell.bash hook)"
conda activate scryneuro
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
LD_LIBRARY_PATH=".:$PYLIB:$LD_LIBRARY_PATH" scryer-prolog "$@"
```

### macOS

On macOS, the shared library extension is `.dylib`, and the environment variable is `DYLD_LIBRARY_PATH`:

```bash
# Build
cargo build --release
cp target/release/libscryneuro.dylib ./

# Run
DYLD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
```

> **Note**: macOS SIP (System Integrity Protection) strips `DYLD_LIBRARY_PATH` from child processes in some contexts (e.g., from GUI apps, or when the binary is in `/usr/bin`). If you encounter issues:
> 1. Use `install_name_tool` to embed the rpath: `install_name_tool -add_rpath @loader_path/. target/release/libscryneuro.dylib`
> 2. Or place `libscryneuro.dylib` in a standard search path like `/usr/local/lib`.

macOS does **not** need the `RTLD_GLOBAL` workaround — Python C extensions resolve symbols differently on Darwin. The `spy_init()` code handles this via `#[cfg(target_os = "linux")]`.

### Rebuilding After Switching Python Environments

If you switch conda environments (or Python versions), **you must rebuild**:

```bash
conda activate other_env
cargo clean          # Remove old build artifacts linked to previous Python
cargo build --release
cp target/release/libscryneuro.so ./  # or .dylib on macOS
```

The build system detects Python via `python3` in your `PATH`. You can override this with:
```bash
PYO3_PYTHON=/path/to/specific/python3 cargo build --release
```

---

## Quick Start

```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

main :-
    py_init,
    X := py_eval("1 + 2"),
    py_to_int(X, Val),
    format("Result: ~d~n", [Val]),
    py_free(X),
    py_finalize.
```

```bash
LD_LIBRARY_PATH=. scryer-prolog my_program.pl    # Linux
DYLD_LIBRARY_PATH=. scryer-prolog my_program.pl  # macOS
```

---

## API Reference

### Lifecycle
- `py_init/0` — Initialize Python interpreter (with RTLD_GLOBAL on Linux).
- `py_init/1` — Initialize with options (e.g., custom library path).
- `py_finalize/0` — Shut down Python interpreter.

### Evaluation & Execution
- `py_eval(Expr, Handle)` — Evaluate Python expression, return result handle.
- `py_exec(Code)` — Execute Python statement for side effects.
- `py_exec_lines(Lines)` — Execute multi-line Python code (list of strings; needed because Scryer doesn't support `\n` in strings).

### Type Conversion
- `py_to_int(Handle, Int)` / `py_from_int(Int, Handle)` — Integer conversion.
- `py_to_float(Handle, Float)` / `py_from_float(Float, Handle)` — Float conversion.
- `py_to_str(Handle, String)` / `py_from_str(Str, Handle)` — String conversion.
- `py_to_bool(Handle, Bool)` / `py_from_bool(Bool, Handle)` — Boolean conversion.
- `py_to_json(Handle, JSON)` / `py_from_json(JSON, Handle)` — JSON serialization.

### Object Operations
- `py_import(Module, Handle)` — Import a Python module.
- `py_getattr(Obj, Attr, Handle)` — Get attribute from object.
- `py_setattr(Obj, Attr, Value)` — Set attribute on object.
- `py_call/3..6` — Call method: `py_call(Obj, Method, Result)`, up to 3 args.
- `py_invoke/2..4` — Call callable: `py_invoke(Callable, Result)`, up to 2 args.

### Collections
- `py_list_new/1`, `py_list_append/2`, `py_list_get/3`, `py_list_len/2`
- `py_dict_new/1`, `py_dict_set/3`, `py_dict_get/3`
- `py_tuple_get/3`, `py_tuple_len/2`

### Memory Management
- `py_free(Handle)` — Release Python object reference.
- `with_py(Handle, Goal)` — Execute Goal, auto-free Handle on exit (success or error).
- `py_handle_count(N)` — Query number of live handles (for debugging).

### Neural & LLM Predicates
- `nn_load(Name, Path, Options)` — Load neural network model.
- `nn_predict(Name, Input, Output)` — Run inference.
- `llm_load(Name, ModelID, Options)` — Configure LLM provider.
- `llm_generate(Name, Prompt, Response)` — Generate text.

---

## Syntactic Sugar: The `:=` Operator

| Pattern | Meaning | Example |
|---|---|---|
| `Var := py_eval(Expr)` | Evaluate expression | `X := py_eval("2**10")` |
| `Var := py_import(Mod)` | Import module | `NP := py_import("numpy")` |
| `Var := py_from_int(N)` | Create Python value | `H := py_from_int(42)` |
| `Var := Obj:"attr"` | Attribute access | `Pi := Math:"pi"` |
| `Var := Obj:method` | No-arg method call | `U := Str:upper` |
| `Var := Obj:method(A,B)` | Method with args | `R := M:sqrt(X)` |

---

## Examples

| File | Description |
|---|---|
| `examples/basic.pl` | Arithmetic, modules, collections, error handling, RAII cleanup |
| `examples/neural.pl` | MNIST classification, neuro-symbolic addition, LLM, RL agents |
| `examples/numpy_torch.pl` | NumPy vectors/matrices, PyTorch tensors, linear regression, CUDA GPU matmul |

```bash
# Run all examples
LD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
LD_LIBRARY_PATH=. scryer-prolog examples/neural.pl
LD_LIBRARY_PATH=. scryer-prolog examples/numpy_torch.pl
```

---

## Troubleshooting

### `error(existence_error(source_sink, library(ffi)), ...)`
Your Scryer Prolog build doesn't include FFI support. Rebuild from latest `main` branch.

### `ImportError: numpy.core.multiarray failed to import`
`libpython` not loaded with `RTLD_GLOBAL`. Ensure `LD_LIBRARY_PATH` includes the Python `lib/` directory, or use `LD_PRELOAD`.

### `error(domain_error(directive, use_foreign_module/2), ...)`
This is a Scryer Prolog quirk — `use_foreign_module/2` is a runtime goal, not a directive. The `scryer_py.pl` module handles this correctly via `:- initialization(...)`.

### Linking error: `cannot find -lpython3.12`
Python shared library not found. Check:
```bash
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
ls $(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3*
```
If empty, your Python was built without `--enable-shared`. Use conda or rebuild Python.

### Version mismatch crash
Build-time and runtime Python versions must match. If you switch environments, `cargo clean && cargo build --release`.

---

## Project Structure

```
ScryNeuro/
├── Cargo.toml              # Rust config: pyo3 = "0.23", libc
├── build.rs                # Python detection + linker config
├── src/
│   ├── lib.rs              # Crate entry point
│   ├── ffi.rs              # 40 exported extern "C" spy_* functions
│   ├── registry.rs         # Thread-safe handle registry (Mutex<HashMap>)
│   ├── convert.rs          # Type conversion + TLS string buffer
│   └── error.rs            # TLS error storage (spy_last_error)
├── prolog/
│   └── scryer_py.pl        # Scryer Prolog API module + := operator
├── python/
│   └── scryer_py_runtime.py # ModelRegistry, LLMManager, TensorUtils
├── examples/
│   ├── basic.pl            # Basic interop demos
│   ├── neural.pl           # Neuro-symbolic patterns
│   └── numpy_torch.pl      # NumPy + PyTorch + CUDA demos
├── test_comprehensive.pl   # 24 low-level FFI tests
├── test_prolog_api.pl      # 17 high-level API tests
├── test_minimal_api.pl     # 3 core smoke tests
└── docs/
    └── technical_report.md # Detailed Chinese technical report
```

## License

MIT
