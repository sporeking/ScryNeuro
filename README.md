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

### Plugin Architecture

NN, LLM, and RL functionality are **opt-in plugins** — separate modules loaded via `use_module`. The core (`scryer_py.pl`) only provides `py_*` predicates and the `:=` operator.

| Plugin | Module file | Predicates |
|---|---|---|
| Neural Networks | `prolog/scryer_nn.pl` | `nn_load/3,4`, `nn_predict/3,4` |
| Large Language Models | `prolog/scryer_llm.pl` | `llm_load/3,4`, `llm_generate/3,4` |
| Reinforcement Learning | `prolog/scryer_rl.pl` | `rl_create/4`, `rl_load/3,4`, `rl_save/2`, `rl_action/3,4`, `rl_train/2,3`, `rl_evaluate/3`, `rl_info/2` |

Each plugin has a matching Python runtime module (`python/scryer_*_runtime.py`) that is loaded lazily on first use.

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
    % Initialize the Python interpreter and load the bridge library.
    py_init,
    
    % Evaluate a Python expression and get a handle to the result.
    X := py_eval("1 + 2"),
    
    % Convert the Python integer handle to a Prolog integer.
    py_to_int(X, Val),
    
    % Print the result.
    format("Result: ~d~n", [Val]),
    
    % Release the Python object handle to prevent memory leaks.
    py_free(X),
    
    % Shut down the Python interpreter.
    py_finalize.
```

This example demonstrates the fundamental workflow: initialize the bridge, evaluate Python code to get a handle, convert the result to a Prolog-native value, print it, free the handle, and shut down. Every ScryNeuro program follows this pattern.

---

## Core Concepts

### Handles
Python objects are stored in a Rust-side HashMap registry. From the Prolog perspective, these objects are represented as opaque integer IDs called **handles**.

- Python objects live in a Rust-side HashMap. Prolog sees them as opaque integer IDs (handles).
- Handle `0` is the error sentinel and never represents a valid Python object. Valid handles start at 1, monotonically increasing.
- When you pass a handle back to a `py_*` function, the Rust layer looks up the actual Python object in the registry.
- Think of handles like file descriptors — opaque numbers that reference real resources.

### The GIL (Global Interpreter Lock)
Every FFI call to the Python runtime acquires the GIL automatically. Python's GIL ensures that only one thread executes Python bytecode at a time. While this is handled transparently, it means that all Python calls are serialized even from multi-threaded Prolog.

### Handle Registry
The registry, managed by `src/registry.rs`, is a thread-safe (Mutex-protected) HashMap that tracks all live Python objects being used by Prolog.
- Every `py_eval`, `py_import`, `py_from_*`, or similar function creates a new entry in the registry and increments the Python object's reference count.
- Calling `py_free/1` removes the entry from the registry and decrements the Python object's reference count.
- Once freed, a handle becomes invalid. Using a freed handle will result in an error.

### Error Handling and Sentinel Patterns
The FFI layer uses three primary patterns to signal errors:
- **Handle functions**: Return `0` on error.
- **Status functions**: Return `-1` on error, `0` on success.
- **String functions**: Return an empty string `""` on error.

The Prolog layer wraps these FFI calls with `check_handle/2` and `check_status/2`. These predicates retrieve the error message via `py_last_error/1` and throw a Prolog exception: `error(python_error(Msg), Context)`. 

Use `catch/3` to handle these errors gracefully:
```prolog
catch(
    (X := py_eval("1/0"), py_to_int(X, V)),
    error(python_error(Msg), _),
    format("Caught: ~s~n", [Msg])
).
```

### Memory Management
Every handle represents a resource in the Rust/Python layers. You must free handles when they are no longer needed to prevent memory leaks.
- `py_free/1`: Manual cleanup of a specific handle.
- `with_py(Handle, Goal)`: RAII-style cleanup. Executes the goal and then frees the handle regardless of whether the goal succeeded, failed, or threw an exception.
- `py_handle_count/1`: Diagnostic tool that returns the number of currently active handles.

### Strings in Scryer Prolog
Scryer Prolog represents double-quoted strings like `"hello"` as lists of characters (char lists). Atoms like `hello` are symbolic constants, not strings. This distinction is critical for the `:=` operator's dispatch mechanism. Note that Scryer Prolog does not support `\n` escapes in double-quoted strings, which is why `py_exec_lines/1` is provided for multi-line Python code.

### TLS String Buffer
String-returning FFI functions, such as `py_to_str` or `py_to_json`, write their results into a thread-local storage (TLS) buffer on the Rust side. The Prolog layer immediately copies the contents of this buffer into a Prolog char list. This management is transparent to the user.

---

## API Reference

### Lifecycle
These predicates manage the state of the embedded Python interpreter.

#### py_init/0
Initialize the Python interpreter with the default library path `./libscryneuro.so`. This call is idempotent, meaning it does nothing if the interpreter is already initialized. On Linux, it handles `RTLD_GLOBAL` for C extensions and adds the current directory `.` to `sys.path`.

#### py_init/1
Initialize the interpreter with a custom path to the shared library. This is also idempotent.

| Parameter | Type | Description |
|---|---|---|
| Path | String | Path to the shared library file |

#### py_finalize/0
Shuts down the Python interpreter, clears the handle registry, and retracts the initialization flag. It is safe to call even if the interpreter wasn't initialized.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

main :-
    py_init,                % Initialize with default path
    % ... your code ...
    py_finalize.            % Clean shutdown

main_custom :-
    py_init("/opt/lib/libscryneuro.so"),  % Custom path
    % ... your code ...
    py_finalize.
```

### Evaluation and Execution
Execute Python code directly from Prolog.

#### py_eval(+Code, -Handle)
Evaluates a Python **expression** and returns a handle to the result. An expression must produce a value (e.g., `1 + 1`, `len([1,2,3])`).

| Parameter | Type | Description |
|---|---|---|
| Code | String | Python expression to evaluate |
| Handle | Integer | Handle to the resulting object |

#### py_exec(+Code)
Executes Python **statements**. Use this for code that does not return a value, such as imports, variable assignments, or class definitions.

| Parameter | Type | Description |
|---|---|---|
| Code | String | Python statements to execute |

#### py_exec_lines(+Lines)
Takes a list of strings and joins them with newlines before passing the result to `py_exec`. This is the preferred way to execute multi-line Python code because Scryer Prolog doesn't support `\n` in strings.

| Parameter | Type | Description |
|---|---|---|
| Lines | List of Strings | Lines of Python code |

**Pitfall**: `py_eval` is for expressions that return a value. `py_exec` is for statements. Using `py_eval` on a statement like `import math` will result in an error.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

eval_exec_demo :-
    py_init,

    %% py_eval: evaluate an expression
    py_eval("2 ** 10", H),
    py_to_int(H, Val),
    format("2^10 = ~d~n", [Val]),
    py_free(H),

    %% py_exec: execute a statement
    py_exec("import math"),

    %% py_eval using an imported module
    py_eval("math.pi", PiH),
    py_to_float(PiH, Pi),
    format("Pi = ~f~n", [Pi]),
    py_free(PiH),

    %% py_exec_lines: multi-line Python code
    py_exec_lines([
        "class Greeter:",
        "    def __init__(self, name):",
        "        self.name = name",
        "    def greet(self):",
        "        return f'Hello, {self.name}!'"
    ]),

    %% Use the class we just defined
    py_eval("Greeter('World')", G),
    py_call(G, "greet", Greeting),
    py_to_str(Greeting, Str),
    format("~s~n", [Str]),
    py_free(Greeting),
    py_free(G).
```

### Modules
Import Python modules to access their functionality.

#### py_import(+ModuleName, -Handle)
Imports a Python module by name and returns a handle to the module object.

| Parameter | Type | Description |
|---|---|---|
| ModuleName | String | Module name (e.g., "math", "numpy") |
| Handle | Integer | Handle to the module object |

**Example:**
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

module_demo :-
    py_init,
    Math := py_import("math"),
    py_getattr(Math, "pi", PiH),
    py_to_float(PiH, Pi),
    format("math.pi = ~f~n", [Pi]),
    py_free(PiH),
    py_free(Math).
```

### Attribute Access
Read and write attributes of Python objects.

#### py_getattr(+Obj, +AttrName, -Value)
Gets the value of an attribute from a Python object.

| Parameter | Type | Description |
|---|---|---|
| Obj | Handle | The Python object |
| AttrName | String | Attribute name |
| Value | Handle | Handle to the attribute value |

#### py_setattr(+Obj, +AttrName, +Value)
Sets the value of an attribute on a Python object.

| Parameter | Type | Description |
|---|---|---|
| Obj | Handle | The Python object |
| AttrName | String | Attribute name |
| Value | Handle | Handle to the new value |

**Example:**
```prolog
:- use_module('prolog/scryer_py').

attr_demo :-
    py_init,
    py_exec_lines([
        "class Point:",
        "    def __init__(self, x, y):",
        "        self.x = x",
        "        self.y = y"
    ]),
    py_eval("Point(3, 4)", P),

    %% Get attributes
    py_getattr(P, "x", XH),
    py_to_int(XH, X),
    format("x = ~d~n", [X]),

    %% Set an attribute
    py_from_int(10, NewX),
    py_setattr(P, "x", NewX),

    %% Verify the change
    py_getattr(P, "x", XH2),
    py_to_int(XH2, X2),
    format("x after set = ~d~n", [X2]),

    maplist(py_free, [P, XH, NewX, XH2]).
```

### Method Calls
Invoke methods on Python objects.

#### py_call(+Obj, +Method, -Result)
#### py_call(+Obj, +Method, +Arg1, -Result)
#### py_call(+Obj, +Method, +Arg1, +Arg2, -Result)
#### py_call(+Obj, +Method, +Arg1, +Arg2, +Arg3, -Result)
Calls a method on an object with 0 to 3 arguments. The last argument is always the output handle.

| Parameter | Type | Description |
|---|---|---|
| Obj | Handle | The Python object |
| Method | String | Method name (must be a string/char list, NOT an atom) |
| ArgX | Handle | Argument handles |
| Result | Handle | Handle to the return value |

#### py_calln(+Obj, +Method, +Args, -Result)
Calls a method with N arguments. `Args` can be either a Prolog list of handles `[H1, H2, ...]` or a handle to an existing Python list or sequence.

**Pitfall**: The method name must be a string. `py_call(Obj, "upper", R)` works, but `py_call(Obj, upper, R)` will fail because `upper` is an atom.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

method_demo :-
    py_init,

    %% 0-arg method call
    py_from_str("hello world", S),
    py_call(S, "upper", Upper),
    py_to_str(Upper, UpperStr),
    format("upper: ~s~n", [UpperStr]),

    %% 2-arg method call: "hello world".replace("world", "prolog")
    py_from_str("world", Old),
    py_from_str("prolog", New),
    py_call(S, "replace", Old, New, Replaced),
    py_to_str(Replaced, ReplacedStr),
    format("replaced: ~s~n", [ReplacedStr]),

    %% N-arg method call with py_calln
    %% Equivalent to "hello world".split("o", 1)
    py_from_str("o", Sep),
    py_from_int(1, MaxSplit),
    py_calln(S, "split", [Sep, MaxSplit], SplitResult),
    py_to_str(SplitResult, SplitStr),
    format("split result: ~s~n", [SplitStr]),

    maplist(py_free, [S, Upper, Old, New, Replaced, Sep, MaxSplit, SplitResult]).
```

### Direct Calls (Callable Objects)
Call functions, lambdas, or class constructors directly.

#### py_invoke(+Callable, -Result)
#### py_invoke(+Callable, +Arg1, -Result)
#### py_invoke(+Callable, +Arg1, +Arg2, -Result)
#### py_invoken(+Callable, +Args, -Result)
Invokes a callable object with 0 to N arguments. Same rules apply for `Args` in `py_invoken` as in `py_calln`.

| Parameter | Type | Description |
|---|---|---|
| Callable | Handle | Any Python callable (function, class, lambda) |
| ArgX | Handle | Argument handles |
| Result | Handle | Handle to the return value |

**Key Difference**: `py_call` calls a method on an object (`obj.method(args)`), whereas `py_invoke` calls the object itself (`callable(args)`). Use `py_invoke` for functions, lambdas, and constructors.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

invoke_demo :-
    py_init,
    py_import("math", Math),

    %% Get a reference to the sqrt function
    py_getattr(Math, "sqrt", SqrtFn),

    %% Call it directly with py_invoke (not py_call!)
    py_from_float(144.0, Arg),
    py_invoke(SqrtFn, Arg, Result),
    py_to_float(Result, Val),
    format("sqrt(144) = ~f~n", [Val]),

    %% Call a lambda
    py_eval("lambda x, y: x * y + 1", Fn),
    py_from_int(3, A),
    py_from_int(4, B),
    py_invoke(Fn, A, B, R2),
    py_to_int(R2, V2),
    format("lambda(3,4) = ~d~n", [V2]),

    maplist(py_free, [Math, SqrtFn, Arg, Result, Fn, A, B, R2]).
```

### Type Conversion
Convert data between Prolog and Python types.

| Predicate | Direction | Prolog Type | Python Type |
|---|---|---|---|
| `py_to_str/2` | Py -> Pl | String (char list) | `str(obj)` |
| `py_to_repr/2` | Py -> Pl | String (char list) | `repr(obj)` |
| `py_to_int/2` | Py -> Pl | Integer | `int` |
| `py_to_float/2` | Py -> Pl | Float | `float` |
| `py_to_bool/2` | Py -> Pl | Atom (`true`/`false`) | `bool` |
| `py_from_int/2` | Pl -> Py | Integer | `int` |
| `py_from_float/2` | Pl -> Py | Float | `float` |
| `py_from_bool/2` | Pl -> Py | Atom (`true`/`false`) | `bool` |
| `py_from_str/2` | Pl -> Py | String (char list) | `str` |

#### py_to_str(+Handle, -String)
Converts a Python object to its string representation using Python's `str()` function.

| Parameter | Type | Description |
|---|---|---|
| Handle | Handle | Python object to convert |
| String | String (char list) | The string representation |

#### py_to_repr(+Handle, -String)
Converts a Python object to its repr string using Python's `repr()` function. Useful for debugging, as it shows the object's type and value in an unambiguous format (e.g., strings are shown with quotes: `'hello'`).

| Parameter | Type | Description |
|---|---|---|
| Handle | Handle | Python object to convert |
| String | String (char list) | The repr representation |

#### py_to_int(+Handle, -Value)
Extracts the integer value from a Python `int` object.

| Parameter | Type | Description |
|---|---|---|
| Handle | Handle | Python int object |
| Value | Integer | The Prolog integer value |

**Note**: Returns `0` on error. If the result could legitimately be 0, check `py_last_error/1` to distinguish.

#### py_to_float(+Handle, -Value)
Extracts the float value from a Python `float` object.

| Parameter | Type | Description |
|---|---|---|
| Handle | Handle | Python float object |
| Value | Float | The Prolog float value |

**Note**: Returns `0.0` on error.

#### py_to_bool(+Handle, -Value)
Extracts a boolean value from a Python `bool` object.

| Parameter | Type | Description |
|---|---|---|
| Handle | Handle | Python bool object |
| Value | Atom | `true` or `false` |

**Note**: Returns Prolog atoms `true`/`false`, NOT integers 1/0. Internally, the FFI returns 1 (true), 0 (false), or -1 (error), and the Prolog layer converts these.

#### py_from_int(+Value, -Handle)
Creates a Python `int` object from a Prolog integer.

| Parameter | Type | Description |
|---|---|---|
| Value | Integer | Prolog integer to convert |
| Handle | Handle | Handle to the new Python int |

#### py_from_float(+Value, -Handle)
Creates a Python `float` object from a Prolog float.

| Parameter | Type | Description |
|---|---|---|
| Value | Float | Prolog float to convert |
| Handle | Handle | Handle to the new Python float |

#### py_from_bool(+Value, -Handle)
Creates a Python `bool` object from a Prolog atom.

| Parameter | Type | Description |
|---|---|---|
| Value | Atom | `true` or `false` |
| Handle | Handle | Handle to the new Python bool |

**Note**: Only accepts `true` or `false` atoms. Any other atom or non-atom will cause an error.

#### py_from_str(+Value, -Handle)
Creates a Python `str` object from a Prolog string (char list).

| Parameter | Type | Description |
|---|---|---|
| Value | String (char list) | The Prolog string to convert |
| Handle | Handle | Handle to the new Python str |

**Pitfalls**:
- `py_to_int` returns `0` on error and when the value is actually 0. Check `py_last_error/1` if the result is ambiguous.
- `py_to_float` returns `0.0` on error.
- `py_to_bool` returns Prolog atoms `true` and `false`, not integers.
- `py_from_bool` expects atoms `true` or `false`.
- `py_to_repr` is useful for debugging as it returns the output of Python's `repr()` function.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

conversion_demo :-
    py_init,

    %% Prolog -> Python -> Prolog round-trip
    py_from_int(42, H1),
    py_to_int(H1, V1),
    format("int round-trip: ~d~n", [V1]),

    py_from_float(3.14, H2),
    py_to_float(H2, V2),
    format("float round-trip: ~f~n", [V2]),

    py_from_bool(true, H3),
    py_to_bool(H3, V3),
    format("bool round-trip: ~w~n", [V3]),

    py_from_str("hello", H4),
    py_to_str(H4, V4),
    format("str round-trip: ~s~n", [V4]),

    %% repr for debugging
    py_to_repr(H4, Repr),
    format("repr: ~s~n", [Repr]),

    maplist(py_free, [H1, H2, H3, H4]).
```

### None
Handle Python's `None` singleton.

#### py_none(-Handle)
Returns a handle to the Python `None` object.

#### py_is_none(+Handle)
Succeeds if the handle points to `None` and fails otherwise. Useful for conditional logic.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

none_demo :-
    py_init,
    py_none(N),
    ( py_is_none(N) ->
        format("It is None~n", [])
    ;   format("It is not None~n", [])
    ),
    py_free(N),

    %% A non-None value
    py_from_int(42, H),
    ( py_is_none(H) ->
        format("42 is None~n", [])
    ;   format("42 is not None~n", [])
    ),
    py_free(H).
```

### JSON Bridge
A robust way to exchange structured data between Prolog and Python.

#### py_to_json(+Handle, -JsonString)
Serializes a Python object to a JSON string using `json.dumps`.

#### py_from_json(+JsonString, -Handle)
Deserializes a JSON string into a Python object using `json.loads`.

**Pitfall**: This bridge only works for JSON-serializable objects (dicts, lists, strings, numbers, booleans, and None). Custom classes or tensors will fail.

**Tip**: The JSON bridge is often the simplest method for transferring complex data structures.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

json_demo :-
    py_init,

    %% Python -> JSON -> Prolog string
    py_eval("{'name': 'Alice', 'age': 30}", DictH),
    py_to_json(DictH, Json),
    format("JSON: ~s~n", [Json]),
    py_free(DictH),

    %% Prolog string -> JSON -> Python
    py_from_json("[1, 2, 3]", ListH),
    py_to_str(ListH, Str),
    format("Parsed: ~s~n", [Str]),
    py_free(ListH).
```

### Collections
Manipulate Python's native list and dictionary types.

#### py_list_new(-Handle)
Creates a new empty Python list `[]`.

#### py_list_append(+List, +Item)
Appends an item handle to a Python list. This operation mutates the list in-place.

#### py_list_get(+List, +Index, -Item)
Retrieves the item at the specified 0-based index.

#### py_list_len(+List, -Len)
Returns the length of the list. Returns -1 if an error occurs.

#### py_list_from_handles(+HandleList, -PyListHandle)
Converts a Prolog list of handles into a Python list object.

#### py_dict_new(-Handle)
Creates a new empty Python dictionary `{}`.

#### py_dict_set(+Dict, +Key, +Value)
Sets a key-value pair in a dictionary. `Key` must be a string, and `Value` must be a handle.

#### py_dict_get(+Dict, +Key, -Value)
Retrieves a value from a dictionary using a string key. This predicate throws an exception if the key is not found.

**Example:**
```prolog
:- use_module('prolog/scryer_py').

collection_demo :-
    py_init,

    %% Build a Python list
    py_list_new(List),
    py_from_int(10, A),
    py_from_int(20, B),
    py_from_int(30, C),
    py_list_append(List, A),
    py_list_append(List, B),
    py_list_append(List, C),
    py_list_len(List, Len),
    format("List length: ~d~n", [Len]),

    %% Access by index
    py_list_get(List, 1, Item),
    py_to_int(Item, ItemVal),
    format("List[1] = ~d~n", [ItemVal]),

    %% Build a Python dict
    py_dict_new(Dict),
    py_from_str("Alice", Name),
    py_from_int(30, Age),
    py_dict_set(Dict, "name", Name),
    py_dict_set(Dict, "age", Age),

    %% Read back
    py_dict_get(Dict, "name", NameBack),
    py_to_str(NameBack, NameStr),
    format("Dict['name'] = ~s~n", [NameStr]),

    %% Serialize entire dict to JSON
    py_to_json(Dict, Json),
    format("JSON: ~s~n", [Json]),

    %% py_list_from_handles: batch convert
    py_from_int(1, H1),
    py_from_int(2, H2),
    py_from_int(3, H3),
    py_list_from_handles([H1, H2, H3], PyList),
    py_to_str(PyList, ListStr),
    format("From handles: ~s~n", [ListStr]),

    %% Clean up
    maplist(py_free, [List, A, B, C, Item, Dict, Name, Age, NameBack,
                      H1, H2, H3, PyList]).
```

### Memory Management
Tools for managing handle lifecycles and diagnosing leaks.

#### py_free(+Handle)
Releases a handle, removing it from the registry and decrementing the Python reference count.

#### with_py(+Handle, +Goal)
RAII-style wrapper. Executes `Goal` and ensures `Handle` is freed regardless of the outcome.

#### py_handle_count(-N)
Returns the number of active handles in the registry. Useful for leak detection.

#### py_last_error(-Error)
Returns the last Python error message as a string. Returns an empty list if no error occurred.

**Example (with_py):**
```prolog
:- use_module('prolog/scryer_py').

raii_demo :-
    py_init,
    py_handle_count(Before),
    format("Handles before: ~d~n", [Before]),

    py_eval("[1, 2, 3, 4, 5]", ListH),
    with_py(ListH, (
        py_to_json(ListH, Json),
        format("List as JSON: ~s~n", [Json])
    )),
    %% ListH is automatically freed here

    py_handle_count(After),
    format("Handles after: ~d~n", [After]).
```

**Example (error checking):**
```prolog
check_error_demo :-
    py_init,
    py_eval("0", ZeroH),
    py_to_int(ZeroH, Val),
    %% Val is 0 — but is it a real 0 or an error?
    py_last_error(Err),
    ( Err = [] ->  %% empty string = no error
        format("Real zero: ~d~n", [Val])
    ;   format("Error occurred: ~s~n", [Err])
    ),
    py_free(ZeroH).
```

### Neural Network Predicates

> **Requires plugin**: `:- use_module('prolog/scryer_nn').`

Predicates for managing and running deep learning models.

#### nn_load(+Name, +Path, +Options)
#### nn_load(+Name, +Path, +Options, -Handle)
Loads a model from a file and registers it under a symbolic name.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Symbolic identifier for the model |
| Path | String | Path to the model file |
| Options | List | `Key=Value` pairs (e.g., `model_type=pytorch`) |

**Common Options for `nn_load`:**
| Option | Example | Description |
|---|---|---|
| `model_type` | `model_type=pytorch` | Framework: `pytorch`, `tensorflow`, `onnx` |
| `device` | `device=cuda` | Compute device: `cpu`, `cuda`, `cuda:0` |
| `weights_only` | `weights_only=true` | PyTorch: load weights only (safer) |

#### nn_predict(+Name, +Input, -Output)
#### nn_predict(+Name, +Input, -Output, +Options)
Runs inference using a registered model.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Identifier matching a loaded model |
| Input | Handle | Input data handle (tensor or array) |
| Output | Handle | Handle to the inference result |
| Options | List | `Key=Value` pairs for inference |

**Common Options for `nn_predict`:**
| Option | Example | Description |
|---|---|---|
| `batch_size` | `batch_size=32` | Batch size for inference |
| `no_grad` | `no_grad=true` | Disable gradient computation |

Options are formatted as `[key1=value1, key2=value2, ...]` where keys are atoms. Values can be numbers or atoms (atoms are converted to strings).

**Example:**
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').
:- use_module('prolog/scryer_nn').

neural_demo :-
    py_init,

    %% Load a PyTorch model
    nn_load(my_model, "models/classifier.pt",
            [model_type=pytorch, device=cpu, weights_only=true]),

    %% Create input tensor (via Python)
    Input := py_eval("__import__('torch').randn(1, 784)"),

    %% Run inference
    nn_predict(my_model, Input, Output),
    py_to_str(Output, OutputStr),
    format("Prediction: ~s~n", [OutputStr]),

    py_free(Input),
    py_free(Output),
    py_finalize.
```

### LLM Predicates

> **Requires plugin**: `:- use_module('prolog/scryer_llm').`

Predicates for interacting with Large Language Model providers.

#### llm_load(+Name, +ModelId, +Options)
#### llm_load(+Name, +ModelId, +Options, -Handle)
Configures an LLM provider and model.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Symbolic identifier |
| ModelId | String | Model ID (e.g., "gpt-4") |
| Options | List | Configuration (e.g., `provider=openai`) |

**Common Options for `llm_load`:**
| Option | Example | Description |
|---|---|---|
| `provider` | `provider=openai` | LLM provider |
| `api_key` | `api_key="sk-..."` | API key (string) |
| `temperature` | `temperature=0.7` | Sampling temperature |
| `max_tokens` | `max_tokens=1024` | Maximum response tokens |
| `base_url` | `base_url="http://..."` | Custom endpoint URL |

Supported providers include `openai`, `anthropic`, `huggingface`, `ollama`, and `custom`.

#### llm_generate(+Name, +Prompt, -Response)
#### llm_generate(+Name, +Prompt, -Response, +Options)
Generates text based on a prompt.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Identifier matching a loaded LLM |
| Prompt | String | Input text prompt |
| Response | String | Generated text response |
| Options | List | Parameters for generation |

**Common Options for `llm_generate`:**
| Option | Example | Description |
|---|---|---|
| `temperature` | `temperature=0.5` | Override temperature |
| `max_tokens` | `max_tokens=256` | Override max tokens |
| `stop` | `stop="\n"` | Stop sequence |

**Example:**
```prolog
:- use_module('prolog/scryer_py').
:- use_module('prolog/scryer_llm').

llm_demo :-
    py_init,
    catch(
        (
            llm_load(gpt, "gpt-4", [provider=openai]),
            llm_generate(gpt, "What is 2+2? Reply with just the number.", Response),
            format("LLM says: ~s~n", [Response])
        ),
        _Error,
        format("LLM not available (no API key or network)~n", [])
    ).
```

### RL Predicates

> **Requires plugin**: `:- use_module('prolog/scryer_rl').`

Predicates for training, evaluating, and using reinforcement learning agents via [Tianshou v2.0](https://github.com/thu-ml/tianshou).

#### rl_create(+Name, +EnvId, +Algorithm, +Options)
Creates and registers a new RL agent.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Symbolic identifier for the agent |
| EnvId | String | Gymnasium environment ID (e.g., `"CartPole-v1"`) |
| Algorithm | Atom | RL algorithm: `dqn`, `ppo`, `a2c`, `sac`, `td3`, `ddpg`, `pg`, `discrete_sac` |
| Options | List | `Key=Value` pairs |

**Common Options for `rl_create`:**
| Option | Example | Description |
|---|---|---|
| `lr` | `lr=0.001` | Learning rate |
| `gamma` | `gamma=0.99` | Discount factor |
| `hidden_sizes` | `hidden_sizes=[64,64]` | MLP hidden layer sizes |
| `n_train_envs` | `n_train_envs=4` | Number of parallel training environments |
| `buffer_size` | `buffer_size=20000` | Replay buffer capacity |
| `eps_training` | `eps_training=0.1` | Epsilon for training (DQN) |

#### rl_load(+Name, +Path, +Options)
#### rl_load(+Name, +Path, +Options, -Handle)
Loads a saved RL agent checkpoint.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Symbolic identifier |
| Path | String | Path to the checkpoint file |
| Options | List | **Required**: `env_id` (string) and `algorithm` (atom) |

#### rl_save(+Name, +Path)
Saves the current agent policy to a checkpoint file.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Identifier of a registered agent |
| Path | String | Output path for the checkpoint |

#### rl_action(+Name, +State, -Action)
#### rl_action(+Name, +State, -Action, +Options)
Queries the agent policy for an action given an observation.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Identifier of a registered agent |
| State | Handle | Handle to the observation tensor |
| Action | Handle | Handle to the selected action |
| Options | List | e.g., `[deterministic=true]` |

#### rl_train(+Name, +Options)
#### rl_train(+Name, +Options, -Metrics)
Runs the training loop for the specified agent.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Identifier of a registered agent |
| Options | List | Training configuration |
| Metrics | Handle | Handle to a dict of training metrics |

**Common Options for `rl_train`:**
| Option | Example | Description |
|---|---|---|
| `max_epochs` | `max_epochs=10` | Number of training epochs |
| `epoch_num_steps` | `epoch_num_steps=5000` | Steps per epoch |
| `batch_size` | `batch_size=64` | Mini-batch size for updates |
| `test_step_num_episodes` | `test_step_num_episodes=5` | Episodes per test phase |

#### rl_evaluate(+Name, +NumEpisodes, -Metrics)
Evaluates the agent over a fixed number of episodes.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Identifier of a registered agent |
| NumEpisodes | Integer | Number of evaluation episodes |
| Metrics | Handle | Handle to evaluation metrics dict |

#### rl_info(+Name, -Info)
Returns metadata about a registered agent.

| Parameter | Type | Description |
|---|---|---|
| Name | Atom | Identifier of a registered agent |
| Info | Handle | Handle to an info dict |

**Example:**
```prolog
:- use_module('prolog/scryer_py').
:- use_module('prolog/scryer_rl').

rl_demo :-
    py_init,

    %% Create a DQN agent for CartPole
    rl_create(agent, "CartPole-v1", dqn,
              [lr=0.001, hidden_sizes=[64,64]]),

    %% Train for 5 epochs
    rl_train(agent, [max_epochs=5, epoch_num_steps=2000], Metrics),
    py_to_str(Metrics, MetricsStr),
    format("Training metrics: ~s~n", [MetricsStr]),

    %% Evaluate
    rl_evaluate(agent, 10, EvalMetrics),
    py_to_str(EvalMetrics, EvalStr),
    format("Eval metrics: ~s~n", [EvalStr]),

    %% Save checkpoint
    rl_save(agent, "checkpoints/cartpole_dqn.pt"),

    py_free(Metrics),
    py_free(EvalMetrics),
    py_finalize.
```

---

## Syntactic Sugar: The `:=` Operator

The `:=` operator enables a more concise syntax for common operations. It uses a 3-way dispatch mechanism to distinguish between types.

### Type Recognition in Scryer Prolog
- `"hello"` is a string (a list of characters, also known as a char list).
- `hello` is an atom (a symbolic constant).
- `hello(X)` is a compound term (an atom followed by arguments).

### Dispatch Logic for `:=`

1. **`Var := Obj:"attrname"`**: When the right side of the colon is a string, it performs attribute access.
   - Translates to: `py_getattr(Obj, "attrname", Var)`
   - Example: `Pi := Math:"pi"` retrieves `math.pi`.

2. **`Var := Obj:methodname`**: When the right side of the colon is an atom, it performs a no-argument method call.
   - Translates to: `py_call(Obj, "methodname", Var)`
   - Example: `U := S:upper` calls `s.upper()`.

3. **`Var := Obj:method(Arg1, Arg2, ...)`**: When the right side is a compound term, it performs a method call with arguments.
   - 1-3 arguments: Dispatches to `py_call/4..6`.
   - 4 or more arguments: Dispatches to `py_calln/4`.
   - Example: `R := S:replace(Old, New)` calls `s.replace(old, new)`.

### Built-in Shortcuts
- `X := py_eval("expr")` → `py_eval("expr", X)`
- `M := py_import("mod")` → `py_import("mod", M)`
- `H := py_from_int(42)` → `py_from_int(42, H)`
- `H := py_from_float(3.14)` → `py_from_float(3.14, H)`
- `H := py_from_str("text")` → `py_from_str("text", H)`
- `H := py_from_json("[1,2]")` → `py_from_json("[1,2]", H)`

**Complete Example:**
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

sugar_demo :-
    py_init,

    %% Built-in shortcut: py_import
    Math := py_import("math"),

    %% String dispatch: attribute access
    Pi := Math:"pi",
    py_to_float(Pi, PiVal),
    format("math.pi = ~f~n", [PiVal]),

    %% Compound dispatch: method call with args
    py_from_float(2.0, Two),
    Sqrt := Math:sqrt(Two),
    py_to_float(Sqrt, SqrtVal),
    format("sqrt(2) = ~f~n", [SqrtVal]),

    %% Atom dispatch: no-arg method call
    S := py_from_str("hello"),
    Upper := S:upper,
    py_to_str(Upper, UpperStr),
    format("upper = ~s~n", [UpperStr]),

    %% Compound with 2 args
    Old := py_from_str("hello"),
    New := py_from_str("HI"),
    Replaced := S:replace(Old, New),
    py_to_str(Replaced, RStr),
    format("replaced = ~s~n", [RStr]),

    maplist(py_free, [Math, Pi, Two, Sqrt, S, Upper, Old, New, Replaced]).
```

### When NOT to Use `:=`

The `:=` operator does NOT support every operation. These patterns require explicit predicate calls:

```prolog
%% WRONG — := cannot wrap py_call directly
X := py_call(Obj, "method", Result).   % SYNTAX ERROR

%% CORRECT — use py_call directly
py_call(Obj, "method", Result).

%% WRONG — := cannot wrap py_invoke
R := py_invoke(Fn, Arg).               % SYNTAX ERROR

%% CORRECT — use py_invoke directly
py_invoke(Fn, Arg, R).

%% WRONG — := cannot set attributes
:= py_setattr(Obj, "name", Val).       % SYNTAX ERROR

%% CORRECT — use py_setattr directly
py_setattr(Obj, "name", Val).
```

The `:=` operator only supports:
1. `py_eval`, `py_import`, `py_from_*`, `py_from_json` (shortcut pattern)
2. `Obj:"attr"` (attribute access)
3. `Obj:method` or `Obj:method(Args...)` (method calls)

For all other operations, use the explicit predicate form.

---

## Common Patterns

**Pattern 1: Batch Data Processing**
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

%% Process a list of Prolog values through a Python function
batch_process(PrologList, Results) :-
    py_init,
    %% Define a Python function
    py_exec_lines([
        "def process_batch(items):",
        "    return [x ** 2 + 1 for x in items]"
    ]),
    %% Build a Python list from Prolog values
    py_list_new(PyList),
    maplist(add_to_list(PyList), PrologList),
    
    %% Call the function
    py_eval("process_batch", Fn),
    py_invoke(Fn, PyList, ResultH),
    
    %% Convert back
    py_to_json(ResultH, Json),
    format("Results: ~s~n", [Json]),
    
    maplist(py_free, [PyList, Fn, ResultH]).

add_to_list(PyList, Val) :-
    py_from_int(Val, H),
    py_list_append(PyList, H),
    py_free(H).
```

**Pattern 2: Error-Resilient Pipeline**
```prolog
:- use_module('prolog/scryer_py').

%% A pipeline that handles errors gracefully at each stage
safe_pipeline :-
    py_init,
    catch(
        pipeline_body,
        error(python_error(Msg), _),
        format("Pipeline failed: ~s~n", [Msg])
    ),
    py_finalize.

pipeline_body :-
    py_exec("import json"),
    
    %% Stage 1: Load data
    py_eval("json.loads('{\"values\": [1, 2, 3]}')", Data),
    with_py(Data, (
        %% Stage 2: Extract values
        py_getattr(Data, "__class__", _),  % verify it's valid
        py_to_json(Data, Json),
        format("Data: ~s~n", [Json])
    )).
```

**Pattern 3: Working with NumPy (if installed)**
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

numpy_demo :-
    py_init,
    NP := py_import("numpy"),
    
    %% Create a numpy array via py_eval
    Arr := py_eval("__import__('numpy').array([1.0, 2.0, 3.0, 4.0, 5.0])"),
    
    %% Call numpy functions on it
    Mean := NP:mean(Arr),
    py_to_float(Mean, MeanVal),
    format("Mean: ~f~n", [MeanVal]),
    
    Std := NP:std(Arr),
    py_to_float(Std, StdVal),
    format("Std: ~f~n", [StdVal]),
    
    %% Dot product
    Arr2 := py_eval("__import__('numpy').array([2.0, 0.0, 1.0, 0.0, 3.0])"),
    Dot := NP:dot(Arr, Arr2),
    py_to_float(Dot, DotVal),
    format("Dot product: ~f~n", [DotVal]),
    
    maplist(py_free, [NP, Arr, Mean, Std, Arr2, Dot]).
```

---

## Examples

| File | Description |
|---|---|
| `examples/basic.pl` | Arithmetic, modules, collections, error handling, RAII cleanup |
| `examples/neural.pl` | MNIST classification, neuro-symbolic addition, LLM, RL agents |
| `examples/numpy_torch.pl` | NumPy vectors/matrices, PyTorch tensors, linear regression, CUDA GPU matmul |
| `examples/mnist_cnn.pl` | CNN training on MNIST from scratch — model definition, training loop, evaluation, neuro-symbolic inference |
| `examples/mnist_cnn_v2.pl` | **Module pattern** (recommended): same CNN training, but Python code in a separate `.py` file |
| `examples/rl_demo.pl` | DQN agent on CartPole-v1 — create, train, evaluate, save, load |

```bash
# Run all examples
LD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
LD_LIBRARY_PATH=. scryer-prolog examples/neural.pl
LD_LIBRARY_PATH=. scryer-prolog examples/numpy_torch.pl
LD_LIBRARY_PATH=. scryer-prolog examples/mnist_cnn.pl
LD_LIBRARY_PATH=. scryer-prolog examples/mnist_cnn_v2.pl
LD_LIBRARY_PATH=. scryer-prolog examples/rl_demo.pl
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
│   ├── scryer_py.pl        # Core: py_* predicates + := operator
│   ├── scryer_nn.pl        # Plugin: nn_load, nn_predict
│   ├── scryer_llm.pl       # Plugin: llm_load, llm_generate
│   └── scryer_rl.pl        # Plugin: rl_create, rl_train, rl_action, ...
├── python/
│   ├── scryer_py_runtime.py  # Core runtime: device management, TensorUtils
│   ├── scryer_nn_runtime.py  # NN runtime: model loading + inference
│   ├── scryer_llm_runtime.py # LLM runtime: provider abstraction
│   └── scryer_rl_runtime.py  # RL runtime: Tianshou v2.0 agent wrappers
├── examples/
│   ├── basic.pl            # Basic interop demos
│   ├── neural.pl           # Neuro-symbolic patterns (NN + LLM + RL)
│   ├── numpy_torch.pl      # NumPy + PyTorch + CUDA demos
│   ├── mnist_cnn.pl        # CNN MNIST training pipeline (inline Python)
│   ├── mnist_cnn_v2.pl     # CNN MNIST training pipeline (module pattern)
│   ├── mnist_cnn_module.py # Python module for mnist_cnn_v2.pl
│   └── rl_demo.pl          # RL demo: DQN on CartPole-v1
├── test_comprehensive.pl   # 24 low-level FFI tests
├── test_prolog_api.pl      # 19 high-level API tests
├── test_minimal_api.pl     # 3 core smoke tests
├── test_rl.pl              # 17 RL plugin tests (scryer_rl.pl)
├── test_rl.py              # 15 Python RL runtime tests (scryer_rl_runtime.py)
└── docs/
    └── technical_report.md # Detailed Chinese technical report
```

## License

MIT
