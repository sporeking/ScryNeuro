# ScryNeuro

ScryNeuro 是一个连接 **Scryer Prolog** 与 **Python** 的高性能桥接框架，专为**神经符号（Neuro-Symbolic）AI** 研究而设计。它允许 Scryer Prolog 程序无缝调用 Python 的神经组件——大语言模型（LLM）、深度神经网络、强化学习 Agent、NumPy、PyTorch——同时保留 Prolog 的逻辑推理能力。

本项目受 [Jurassic.pl](https://github.com/haldai/Jurassic.pl)（SWI-Prolog ↔ Julia 桥接框架）启发。

## 架构

```text
[ Scryer Prolog ] <-> [ Rust cdylib (FFI) ] <-> [ PyO3 ] <-> [ Python 运行时 ]
     （逻辑层）          （桥接层）              （绑定）       （神经/感知层）
```

- **Scryer Prolog** — 负责逻辑推理和顶层控制流。
- **Rust cdylib** (`libscryneuro.so` / `.dylib`) — FFI 桥接，内含基于句柄的对象注册表。
- **PyO3** — 在 Rust 内嵌入 Python；管理 GIL 和类型转换。
- **Python** — 执行神经谓词、数据处理、库调用（PyTorch、NumPy、OpenAI 等）。

---

## 安装部署

### 系统要求

| 组件 | 版本 | 说明 |
|---|---|---|
| **Rust** | stable ≥ 1.70 | 推荐使用 `rustup` |
| **Python** | 3.10 – 3.13 | 必须包含共享库（`libpython3.x.so` / `.dylib`） |
| **Scryer Prolog** | 最新 git | 须支持 `library(ffi)` |

### 第一步：安装 Rust

```bash
# 安装 rustup（如未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 验证
rustc --version
cargo --version
```

### 第二步：安装 Scryer Prolog

```bash
# 从源码编译（需要 Rust）
git clone https://github.com/mthom/scryer-prolog.git
cd scryer-prolog
cargo install --path .

# 验证
scryer-prolog --version
```

### 第三步：配置 Python 环境

ScryNeuro 在**编译时**链接当前活跃的 `python3`，在**运行时**加载 `libpython3.x.so`。两者必须匹配。

#### 方案 A：Conda（推荐）

```bash
# 创建专用环境
conda create -n scryneuro python=3.12 numpy -y
conda activate scryneuro

# 安装 ML 库（按需选择）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  # GPU 版
# 或
conda install pytorch torchvision torchaudio cpuonly -c pytorch  # CPU 版

# 验证共享库是否存在
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
# 应输出类似：/home/user/miniconda3/envs/scryneuro/lib
ls $(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3*.so*
```

#### 方案 B：uv

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建项目虚拟环境
uv venv .venv --python 3.12
source .venv/bin/activate

# 安装依赖
uv pip install numpy torch

# 验证
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
```

#### 方案 C：系统 Python

```bash
# Debian / Ubuntu
sudo apt install python3-dev python3-numpy

# Fedora
sudo dnf install python3-devel python3-numpy

# macOS（Homebrew）
brew install python@3.12 numpy
```

> **关键提示**：Python 必须以共享库模式编译。Conda 和系统包默认包含共享库。如果使用 `pyenv`，编译时须指定：`PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.12`

### 第四步：编译 ScryNeuro

```bash
git clone <repo-url> ScryNeuro
cd ScryNeuro

# 先激活 Python 环境！
conda activate scryneuro  # 或：source .venv/bin/activate

# 编译
cargo build --release

# 将共享库复制到项目根目录
cp target/release/libscryneuro.so ./     # Linux
# cp target/release/libscryneuro.dylib ./  # macOS
```

编译输出中应显示 `Building with Python 3.12.x`（与当前激活的环境一致）。

### 第五步：验证

```bash
# Linux
LD_LIBRARY_PATH=. scryer-prolog examples/basic.pl

# macOS
DYLD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
```

预期输出：
```
=== Arithmetic ===
2^10 = 1024
sum(0..99) = 4950
...
=== All basic examples complete ===
```

---

## 平台差异说明

### Linux

`spy_init()` 中的 `RTLD_GLOBAL` 机制会自动将 `libpython3.x.so` 以全局符号可见模式重新打开。这是 NumPy、PyTorch 等 C 扩展库正常运行所必需的。你需要确保 `libpython` 可以被找到：

```bash
# 方式一：设置 LD_LIBRARY_PATH（推荐用于 conda）
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH

# 方式二：使用 LD_PRELOAD（如果 RTLD_GLOBAL 自动检测失败）
export LD_PRELOAD=$(python3 -c "import sysconfig, os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), 'libpython3.12.so'))")
```

**便捷启动脚本** — 创建 `run.sh`：
```bash
#!/bin/bash
# 激活 conda 环境并运行 Prolog 脚本
eval "$(conda shell.bash hook)"
conda activate scryneuro
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
LD_LIBRARY_PATH=".:$PYLIB:$LD_LIBRARY_PATH" scryer-prolog "$@"
```
```bash
chmod +x run.sh
./run.sh examples/basic.pl
```

### macOS

在 macOS 上，共享库扩展名为 `.dylib`，环境变量使用 `DYLD_LIBRARY_PATH`：

```bash
# 编译
cargo build --release
cp target/release/libscryneuro.dylib ./

# 运行
DYLD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
```

> **注意**：macOS 的 SIP（系统完整性保护）在某些情况下（从 GUI 应用启动、二进制在 `/usr/bin` 中）会剥离 `DYLD_LIBRARY_PATH`。如遇问题：
> 1. 使用 `install_name_tool` 嵌入 rpath：`install_name_tool -add_rpath @loader_path/. target/release/libscryneuro.dylib`
> 2. 或将 `libscryneuro.dylib` 放到标准搜索路径如 `/usr/local/lib`。

macOS **不需要** `RTLD_GLOBAL` 的变通措施——Python C 扩展在 Darwin 上以不同方式解析符号。`spy_init()` 中的代码通过 `#[cfg(target_os = "linux")]` 仅在 Linux 上激活。

### 切换 Python 环境后须重新编译

如果切换了 conda 环境或 Python 版本，**必须重新编译**：

```bash
conda activate other_env
cargo clean          # 清除链接到旧 Python 的构建产物
cargo build --release
cp target/release/libscryneuro.so ./  # 或 macOS 上的 .dylib
```

构建系统通过 `PATH` 中的 `python3` 检测 Python。可用以下方式覆盖：
```bash
PYO3_PYTHON=/path/to/specific/python3 cargo build --release
```

---

## 快速开始

```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

main :-
    py_init,
    X := py_eval("1 + 2"),
    py_to_int(X, Val),
    format("结果: ~d~n", [Val]),
    py_free(X),
    py_finalize.
```

```bash
LD_LIBRARY_PATH=. scryer-prolog my_program.pl    # Linux
DYLD_LIBRARY_PATH=. scryer-prolog my_program.pl  # macOS
```

---

## API 参考

### 生命周期
- `py_init/0` — 初始化 Python 解释器（Linux 上自动处理 RTLD_GLOBAL）。
- `py_init/1` — 使用选项初始化（如自定义库路径）。
- `py_finalize/0` — 关闭 Python 解释器。

### 求值与执行
- `py_eval(Expr, Handle)` — 求值 Python 表达式，返回结果句柄。
- `py_exec(Code)` — 执行 Python 语句（仅副作用）。
- `py_exec_lines(Lines)` — 执行多行 Python 代码（字符串列表；因 Scryer 不支持 `\n` 转义）。

### 类型转换
- `py_to_int(Handle, Int)` / `py_from_int(Int, Handle)` — 整数转换。
- `py_to_float(Handle, Float)` / `py_from_float(Float, Handle)` — 浮点数转换。
- `py_to_str(Handle, String)` / `py_from_str(Str, Handle)` — 字符串转换。
- `py_to_bool(Handle, Bool)` / `py_from_bool(Bool, Handle)` — 布尔值转换。
- `py_to_json(Handle, JSON)` / `py_from_json(JSON, Handle)` — JSON 序列化。

### 对象操作
- `py_import(Module, Handle)` — 导入 Python 模块。
- `py_getattr(Obj, Attr, Handle)` — 获取对象属性。
- `py_setattr(Obj, Attr, Value)` — 设置对象属性。
- `py_call/3..6` — 调用方法：`py_call(Obj, Method, Result)`，最多 3 个参数。
- `py_invoke/2..4` — 调用可调用对象：`py_invoke(Callable, Result)`，最多 2 个参数。

### 集合操作
- `py_list_new/1`、`py_list_append/2`、`py_list_get/3`、`py_list_len/2`
- `py_dict_new/1`、`py_dict_set/3`、`py_dict_get/3`
- `py_tuple_get/3`、`py_tuple_len/2`

### 内存管理
- `py_free(Handle)` — 释放 Python 对象引用。
- `with_py(Handle, Goal)` — 执行 Goal 后自动释放 Handle（无论成功或出错）。
- `py_handle_count(N)` — 查询当前存活句柄数（调试用）。

### 神经网络与 LLM 谓词
- `nn_load(Name, Path, Options)` — 加载神经网络模型。
- `nn_predict(Name, Input, Output)` — 执行推理。
- `llm_load(Name, ModelID, Options)` — 配置 LLM 提供者。
- `llm_generate(Name, Prompt, Response)` — 生成文本。

---

## 语法糖：`:=` 运算符

| 模式 | 含义 | 示例 |
|---|---|---|
| `Var := py_eval(Expr)` | 求值表达式 | `X := py_eval("2**10")` |
| `Var := py_import(Mod)` | 导入模块 | `NP := py_import("numpy")` |
| `Var := py_from_int(N)` | 创建 Python 值 | `H := py_from_int(42)` |
| `Var := Obj:"attr"` | 属性访问 | `Pi := Math:"pi"` |
| `Var := Obj:method` | 无参方法调用 | `U := Str:upper` |
| `Var := Obj:method(A,B)` | 带参方法调用 | `R := M:sqrt(X)` |

---

## 示例

| 文件 | 说明 |
|---|---|
| `examples/basic.pl` | 算术运算、模块导入、集合操作、错误处理、RAII 自动清理 |
| `examples/neural.pl` | MNIST 分类、神经符号加法、LLM 集成、强化学习 Agent |
| `examples/numpy_torch.pl` | NumPy 向量/矩阵、PyTorch 张量、线性回归、CUDA GPU 矩阵乘法 |

```bash
# 运行所有示例
LD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
LD_LIBRARY_PATH=. scryer-prolog examples/neural.pl
LD_LIBRARY_PATH=. scryer-prolog examples/numpy_torch.pl
```

---

## 常见问题排查

### `error(existence_error(source_sink, library(ffi)), ...)`
Scryer Prolog 未包含 FFI 支持。请从最新 `main` 分支重新编译。

### `ImportError: numpy.core.multiarray failed to import`
`libpython` 未以 `RTLD_GLOBAL` 模式加载。确保 `LD_LIBRARY_PATH` 包含 Python 的 `lib/` 目录，或使用 `LD_PRELOAD`。

### `error(domain_error(directive, use_foreign_module/2), ...)`
这是 Scryer Prolog 的特性——`use_foreign_module/2` 是运行时目标而非编译期指令。`scryer_py.pl` 已通过 `:- initialization(...)` 正确处理。

### 链接错误：`cannot find -lpython3.12`
Python 共享库未找到。检查：
```bash
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
ls $(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3*
```
如果为空，说明 Python 未以 `--enable-shared` 编译。使用 conda 或重新编译 Python。

### 版本不匹配导致崩溃
编译时与运行时的 Python 版本必须一致。切换环境后务必 `cargo clean && cargo build --release`。

---

## 项目结构

```
ScryNeuro/
├── Cargo.toml              # Rust 配置：pyo3 = "0.23", libc
├── build.rs                # Python 检测 + 链接器配置
├── src/
│   ├── lib.rs              # Crate 入口
│   ├── ffi.rs              # 40 个 extern "C" spy_* 导出函数
│   ├── registry.rs         # 线程安全句柄注册表（Mutex<HashMap>）
│   ├── convert.rs          # 类型转换 + TLS 字符串缓冲区
│   └── error.rs            # TLS 错误存储（spy_last_error）
├── prolog/
│   └── scryer_py.pl        # Scryer Prolog API 模块 + := 运算符
├── python/
│   └── scryer_py_runtime.py # ModelRegistry, LLMManager, TensorUtils
├── examples/
│   ├── basic.pl            # 基础交互示例
│   ├── neural.pl           # 神经符号模式示例
│   └── numpy_torch.pl      # NumPy + PyTorch + CUDA 示例
├── test_comprehensive.pl   # 24 项底层 FFI 测试
├── test_prolog_api.pl      # 17 项高层 API 测试
├── test_minimal_api.pl     # 3 项核心冒烟测试
└── docs/
    └── technical_report.md # 详细中文技术报告
```

## 开源协议

MIT
