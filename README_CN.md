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

### 插件架构（Plugin Architecture）

NN、LLM 和 RL 功能以**可选插件**形式提供——各自为独立模块，按需通过 `use_module` 加载。核心模块 `scryer_py.pl` 只提供 `py_*` 谓词和 `:=` 运算符。

| 插件 | 模块文件 | 谓词 |
|---|---|---|
| 神经网络 | `prolog/scryer_nn.pl` | `nn_load/3,4`, `nn_predict/3,4` |
| 大语言模型 | `prolog/scryer_llm.pl` | `llm_load/3,4`, `llm_generate/3,4` |
| 强化学习 | `prolog/scryer_rl.pl` | `rl_create/4`, `rl_load/3,4`, `rl_save/2`, `rl_action/3,4`, `rl_train/2,3`, `rl_evaluate/3`, `rl_info/2` |

每个插件均有对应的 Python 运行时模块（`python/scryer_*_runtime.py`），在首次使用时懒加载。
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

## 核心概念 (Core Concepts)

### 句柄（Handles）
Python 对象被统一存放在 Rust 侧的一个“哈希表（HashMap）”注册表中。而在 Prolog 这边，这些对象被表示为一个个不透明的整数 ID，我们称之为**句柄（Handle）**。

- **通俗理解**：你可以把句柄想象成去超市存包时拿到的“手牌”。Prolog 拿着这个手牌（整数 ID），就可以通过 Rust 告诉 Python：“帮我操作存放在你那里的对象”。
- 句柄 `0` 是一个特殊的错误标记（Sentinel），它永远不会代表一个有效的 Python 对象。有效的句柄从 1 开始不断递增。
- 当你把一个句柄传给 `py_*` 系列的函数时，Rust 层会在注册表中查找与之对应的真正的 Python 对象。

### GIL（全局解释器锁）
每次从 FFI（外部函数接口）调用 Python 运行时，ScryNeuro 都会自动获取 GIL。Python 的 GIL 机制保证了在同一时刻，只能有一个线程在执行 Python 字节码。虽然这些都在后台为你打理好了，但这也意味着即便你的 Prolog 程序是多线程的，所有的 Python 调用在底层仍然是串行执行的。

### 句柄注册表（Handle Registry）
注册表由 `src/registry.rs` 管理，它是一个线程安全的（由 Mutex 互斥锁保护的）哈希表，负责跟踪 Prolog 正在使用的所有活跃的 Python 对象。
- 每次调用 `py_eval`、`py_import` 或 `py_from_*` 等函数时，都会在注册表中创建一个新条目，并增加该 Python 对象的引用计数（防止其被 Python 垃圾回收机制清理）。
- 调用 `py_free/1` 会从注册表中移除该条目，并减少引用计数。
- 一旦被释放，句柄就失效了。如果继续使用一个已被释放的句柄，将会导致错误。

### 错误处理与哨兵模式
FFI（C 语言接口）层使用了三种主要的设计模式来发出错误信号：
- **句柄类函数**：发生错误时返回 `0`。
- **状态类函数**：发生错误时返回 `-1`，成功则返回 `0`。
- **字符串类函数**：发生错误时返回空字符串 `""`。

在 Prolog 层，我们用 `check_handle/2` 和 `check_status/2` 等谓词对这些底层调用进行了封装。当错误发生时，这些谓词会通过 `py_last_error/1` 获取具体的错误信息，并在 Prolog 中抛出一个异常：`error(python_error(Msg), Context)`。

你可以使用 Prolog 的 `catch/3` 来优雅地捕获并处理这些异常：
```prolog
catch(
    (X := py_eval("1/0"), py_to_int(X, V)),
    error(python_error(Msg), _),
    format("捕获到异常: ~s~n", [Msg])
).
```

### 内存管理
每一个句柄都代表着 Rust/Python 层的一段真实资源。当句柄不再被需要时，你**必须手动释放它们**以防止内存泄漏。
- `py_free/1`：手动清理某个特定的句柄。
- `with_py(Handle, Goal)`：类似于现代编程语言中的 RAII 模式或 Python 的 `with` 语句。它会先执行目标代码 `Goal`，并在执行完毕后自动释放 `Handle`，无论执行成功、失败，还是抛出异常，它都能保证句柄被安全释放。
- `py_handle_count/1`：诊断工具，返回当前活跃的句柄数量。

### Scryer Prolog 中的字符串
在 Scryer Prolog 中，用双引号包裹的字符串（比如 `"hello"`）实际上是由字符组成的**列表**（char lists）。而像 `hello` 这样没有引号的则是**原子（Atoms）**，它们是符号常量，并不是字符串。这一区别在后续的 `:=` 运算符分发机制中非常关键。另外，Scryer Prolog 的双引号字符串不支持 `\n` 转义，因此我们专门提供了 `py_exec_lines/1` 来执行多行 Python 代码。

### TLS 字符串缓冲区
在 Rust 层，像 `py_to_str` 或 `py_to_json` 这类返回字符串的 FFI 函数，会把结果先写入一个“线程局部存储”（TLS）缓冲区中。Prolog 层会立即将该缓冲区的内容复制为 Prolog 的字符列表。这种底层的内存处理对用户是完全透明的。

---

## API 参考

### 生命周期
这些谓词用于管理内嵌 Python 解释器的状态。

#### `py_init/0`
使用默认的库路径 `./libscryneuro.so` 初始化 Python 解释器。这是一个**幂等**操作（即使解释器已初始化，重复调用也不会有副作用）。在 Linux 上，它会自动处理 C 扩展所需的 `RTLD_GLOBAL` 可见性，并将当前目录 `.` 添加入 `sys.path`。

#### `py_init/1`
使用自定义路径初始化解释器。同样是幂等操作。
| 参数 | 类型 | 说明 |
|---|---|---|
| Path | 字符串 | 共享库文件的路径 |

#### `py_finalize/0`
关闭 Python 解释器，清空句柄注册表并重置初始化标记。即使解释器未初始化，调用也是安全的。

**示例:**
```prolog
:- use_module('prolog/scryer_py').

main :-
    py_init,                % 默认路径初始化
    % ... 你的代码 ...
    py_finalize.            % 干净地关闭退出
```

### 求值与执行
在 Prolog 中直接执行 Python 代码。

#### `py_eval(+Code, -Handle)`
求值一段 Python **表达式 (expression)** 并返回结果的句柄。表达式必须产生一个值（例如 `1 + 1`，`len([1,2,3])`）。
| 参数 | 类型 | 说明 |
|---|---|---|
| Code | 字符串 | 要计算的 Python 表达式 |
| Handle | 整数 | 指向计算结果对象的句柄 |

#### `py_exec(+Code)`
执行一段 Python **语句 (statement)**。适用于没有返回值的代码，例如 `import`、变量赋值或类/函数定义。
| 参数 | 类型 | 说明 |
|---|---|---|
| Code | 字符串 | 要执行的 Python 语句 |

#### `py_exec_lines(+Lines)`
接收一个字符串列表，用换行符拼接后传给 `py_exec`。这是执行多行代码的首选方式。

> **避坑指南**：`py_eval` 用于计算有返回值的表达式，`py_exec` 用于执行无返回值的语句。对 `import math` 使用 `py_eval` 会导致报错。

**示例:**
```prolog
:- use_module('prolog/scryer_py').

eval_exec_demo :-
    py_init,
    %% py_eval: 求值表达式
    py_eval("2 ** 10", H),
    py_to_int(H, Val),
    format("2^10 = ~d~n", [Val]),
    py_free(H),

    %% py_exec: 执行语句
    py_exec("import math"),

    %% py_exec_lines: 执行多行代码
    py_exec_lines([
        "class Greeter:",
        "    def __init__(self, name):",
        "        self.name = name",
        "    def greet(self):",
        "        return f'Hello, {self.name}!'"
    ]),

    %% 使用刚才定义的类
    py_eval("Greeter('World')", G),
    py_call(G, "greet", Greeting),
    py_to_str(Greeting, Str),
    format("~s~n", [Str]),
    maplist(py_free, [Greeting, G]).
```

### 模块 (Modules)
导入 Python 模块以访问其功能。

#### `py_import(+ModuleName, -Handle)`
根据名称导入 Python 模块，返回指向该模块对象的句柄。
| 参数 | 类型 | 说明 |
|---|---|---|
| ModuleName | 字符串 | 模块名称 (如 "math", "numpy") |
| Handle | 整数 | 模块对象的句柄 |

### 属性访问 (Attribute Access)
读取和设置 Python 对象的属性。

#### `py_getattr(+Obj, +AttrName, -Value)`
获取对象的属性值。
| 参数 | 类型 | 说明 |
|---|---|---|
| Obj | 句柄 | Python 对象 |
| AttrName | 字符串 | 属性名称 |
| Value | 句柄 | 属性值的句柄 |

#### `py_setattr(+Obj, +AttrName, +Value)`
设置对象的属性值。

**示例:**
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

    %% 获取属性
    py_getattr(P, "x", XH),
    py_to_int(XH, X),
    format("x = ~d~n", [X]),

    %% 设置属性
    py_from_int(10, NewX),
    py_setattr(P, "x", NewX),
    maplist(py_free, [P, XH, NewX]).
```

### 方法调用 (Method Calls)
调用 Python 对象的方法。

#### `py_call(+Obj, +Method, -Result)` (支持 0 到 3 个参数)
调用对象的方法。最后一个参数始终是输出结果的句柄。
| 参数 | 类型 | 说明 |
|---|---|---|
| Obj | 句柄 | Python 对象 |
| Method | 字符串 | 方法名（必须是字符串/字符列表，不能是原子） |
| ArgX | 句柄 | 传入的参数句柄 |
| Result | 句柄 | 返回值的句柄 |

#### `py_calln(+Obj, +Method, +Args, -Result)`
调用支持 N 个参数的方法。`Args` 可以是 Prolog 句柄列表 `[H1, H2, ...]`，也可以是现有 Python 列表的句柄。

> **避坑指南**：方法名必须是字符串。`py_call(Obj, "upper", R)` 是正确的，但 `py_call(Obj, upper, R)` 会失败，因为 `upper` 是原子。

### 直接调用 (Callable Objects)
直接调用函数、Lambda 表达式或类构造器。

#### `py_invoke(+Callable, -Result)` (支持 0 到 N 个参数)
**核心区别**：`py_call` 是调用对象身上的方法 (`obj.method(args)`)，而 `py_invoke` 是直接调用可调用对象本身 (`callable(args)`)。调用普通函数、Lambda 表达式或实例化对象时请使用 `py_invoke`。

### 类型转换 (Type Conversion)
在 Prolog 和 Python 类型之间转换数据。

| 谓词 | 方向 | Prolog 类型 | Python 类型 |
|---|---|---|---|
| `py_to_str/2` | Py -> Pl | 字符串 (字符列表) | `str(obj)` |
| `py_to_repr/2` | Py -> Pl | 字符串 (字符列表) | `repr(obj)` |
| `py_to_int/2` | Py -> Pl | 整数 | `int` |
| `py_to_float/2` | Py -> Pl | 浮点数 | `float` |
| `py_to_bool/2` | Py -> Pl | 原子 (`true`/`false`) | `bool` |
| `py_from_int/2` | Pl -> Py | 整数 | `int` |
| `py_from_float/2` | Pl -> Py | 浮点数 | `float` |
| `py_from_bool/2` | Pl -> Py | 原子 (`true`/`false`) | `bool` |
| `py_from_str/2` | Pl -> Py | 字符串 (字符列表) | `str` |

> **避坑指南**：
> - `py_to_int` 发生错误时会返回 `0`。如果预期值可能确实是 0，请配合检查 `py_last_error/1`。
> - `py_to_bool` 返回的是 Prolog 原子 `true` 和 `false`，而不是整数。
> - `py_to_repr` 对于调试非常有用，它返回 Python 中 `repr()` 的输出（如字符串会带上引号 `'hello'`）。

### None
处理 Python 中的 `None` 单例对象。
- `py_none(-Handle)`：获取指向 `None` 的句柄。
- `py_is_none(+Handle)`：检查给定句柄是否为 `None`，通常用于条件判断。

### JSON 桥接 (JSON Bridge)
在 Prolog 和 Python 之间交换复杂结构化数据的最稳妥方式。
- `py_to_json(+Handle, -JsonString)`：使用 `json.dumps` 序列化。
- `py_from_json(+JsonString, -Handle)`：使用 `json.loads` 反序列化。
> **注意**：仅适用于 JSON 可序列化的对象（字典、列表、字符串、数字、布尔值和 None）。自定义类或张量（Tensor）无法直接使用。

### 集合操作 (Collections)
操作 Python 原生的列表和字典。
- 列表：`py_list_new/1`, `py_list_append/2`, `py_list_get/3`, `py_list_len/2`, `py_list_from_handles/2`
- 字典：`py_dict_new/1`, `py_dict_set/3`, `py_dict_get/3`（按字符串 Key 获取，找不到会抛异常）

### 内存管理
用于管理句柄生命周期和诊断内存泄漏的工具。
- `py_free(+Handle)`：释放句柄，将其从注册表中移除并减少引用计数。
- `with_py(+Handle, +Goal)`：以 RAII 风格执行 `Goal`，完毕后自动释放 `Handle`。
- `py_handle_count(-N)`：获取当前活跃的句柄数量。

### 神经网络谓词

> **需要插件**：`:- use_module('prolog/scryer_nn').`

用于管理并运行深度学习模型。

#### `nn_load(+Name, +Path, +Options)`
#### `nn_load(+Name, +Path, +Options, -Handle)`
从文件加载模型并注册为一个符号名称（Atom）。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 该模型的标识符（比如 `my_model`） |
| Path | 字符串 | 模型文件路径 |
| Options | 列表 | `Key=Value` 的配置对（如 `[model_type=pytorch, device=cuda]`） |

**`nn_load` 常用选项：**
| 选项 | 示例 | 说明 |
|---|---|---|
| `model_type` | `model_type=pytorch` | 框架：`pytorch`, `tensorflow`, `onnx` |
| `device` | `device=cuda` | 计算设备：`cpu`, `cuda`, `cuda:0` |
| `weights_only` | `weights_only=true` | PyTorch：仅加载权重（更安全） |

#### `nn_predict(+Name, +Input, -Output)`
#### `nn_predict(+Name, +Input, -Output, +Options)`
使用已加载的模型执行推理。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 匹配所加载模型的标识符 |
| Input | 句柄 | 输入数据的句柄（例如张量 Tensor） |
| Output | 句柄 | 推理结果句柄 |
| Options | 列表 | `Key=Value` 推理配置对 |

**示例：**
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').
:- use_module('prolog/scryer_nn').

neural_demo :-
    py_init,
    nn_load(my_model, "models/classifier.pt",
            [model_type=pytorch, device=cpu, weights_only=true]),
    Input := py_eval("__import__('torch').randn(1, 784)"),
    nn_predict(my_model, Input, Output),
    py_to_str(Output, OutputStr),
    format("Prediction: ~s~n", [OutputStr]),
    py_free(Input), py_free(Output), py_finalize.
```

---

### LLM 谓词

> **需要插件**：`:- use_module('prolog/scryer_llm').`

用于与大语言模型提供商交互。

#### `llm_load(+Name, +ModelId, +Options)`
#### `llm_load(+Name, +ModelId, +Options, -Handle)`
配置 LLM 提供商和模型。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 标识符 |
| ModelId | 字符串 | 模型 ID（如 `"gpt-4"`） |
| Options | 列表 | 配置项（如 `provider=openai`） |

**`llm_load` 常用选项：**
| 选项 | 示例 | 说明 |
|---|---|---|
| `provider` | `provider=openai` | LLM 提供商 |
| `api_key` | `api_key="sk-..."` | API 密钥（字符串） |
| `temperature` | `temperature=0.7` | 采样温度 |
| `max_tokens` | `max_tokens=1024` | 最大生成 token 数 |
| `base_url` | `base_url="http://..."` | 自定义端点 URL |

支持的提供商：`openai`, `anthropic`, `huggingface`, `ollama`, `custom`。

#### `llm_generate(+Name, +Prompt, -Response)`
#### `llm_generate(+Name, +Prompt, -Response, +Options)`
根据提示词生成文本。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 匹配已加载 LLM 的标识符 |
| Prompt | 字符串 | 输入提示词 |
| Response | 字符串 | 生成的文本响应 |
| Options | 列表 | 生成参数 |

**示例：**
```prolog
:- use_module('prolog/scryer_py').
:- use_module('prolog/scryer_llm').

llm_demo :-
    py_init,
    catch(
        (
            llm_load(gpt, "gpt-4", [provider=openai]),
            llm_generate(gpt, "2+2 等于几？只回答数字。", Response),
            format("LLM 回答: ~s~n", [Response])
        ),
        _Error,
        format("LLM 不可用（无 API 密钥或网络）~n", [])
    ).
```

---

### RL 谓词

> **需要插件**：`:- use_module('prolog/scryer_rl').`

通过 [Tianshou v2.0](https://github.com/thu-ml/tianshou) 训练、评估和使用强化学习 Agent 的谓词。

#### `rl_create(+Name, +EnvId, +Algorithm, +Options)`
创建并注册新的 RL Agent。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | Agent 的符号标识符 |
| EnvId | 字符串 | Gymnasium 环境 ID（如 `"CartPole-v1"`） |
| Algorithm | 原子 | RL 算法：`dqn`, `ppo`, `a2c`, `sac`, `td3`, `ddpg`, `pg`, `discrete_sac` |
| Options | 列表 | `Key=Value` 配置对 |

**`rl_create` 常用选项：**
| 选项 | 示例 | 说明 |
|---|---|---|
| `lr` | `lr=0.001` | 学习率 |
| `gamma` | `gamma=0.99` | 折扣因子 |
| `hidden_sizes` | `hidden_sizes=[64,64]` | MLP 隐藏层大小 |
| `n_train_envs` | `n_train_envs=4` | 并行训练环境数量 |
| `buffer_size` | `buffer_size=20000` | 回放缓冲区容量 |
| `eps_training` | `eps_training=0.1` | 训练时的 Epsilon（DQN） |

#### `rl_load(+Name, +Path, +Options)`
#### `rl_load(+Name, +Path, +Options, -Handle)`
加载已保存的 RL Agent 检查点。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 符号标识符 |
| Path | 字符串 | 检查点文件路径 |
| Options | 列表 | **必须**包含 `env_id`（字符串）和 `algorithm`（原子） |

#### `rl_save(+Name, +Path)`
将当前 Agent 策略保存到检查点文件。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 已注册 Agent 的标识符 |
| Path | 字符串 | 检查点输出路径 |

#### `rl_action(+Name, +State, -Action)`
#### `rl_action(+Name, +State, -Action, +Options)`
根据观测查询 Agent 策略以获取动作。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 已注册 Agent 的标识符 |
| State | 句柄 | 观测张量的句柄 |
| Action | 句柄 | 选定动作的句柄 |
| Options | 列表 | 如 `[deterministic=true]` |

#### `rl_train(+Name, +Options)`
#### `rl_train(+Name, +Options, -Metrics)`
运行指定 Agent 的训练循环。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 已注册 Agent 的标识符 |
| Options | 列表 | 训练配置 |
| Metrics | 句柄 | 训练指标字典的句柄 |

**`rl_train` 常用选项：**
| 选项 | 示例 | 说明 |
|---|---|---|
| `max_epochs` | `max_epochs=10` | 训练轮数 |
| `epoch_num_steps` | `epoch_num_steps=5000` | 每轮步数 |
| `batch_size` | `batch_size=64` | 更新的 mini-batch 大小 |
| `test_step_num_episodes` | `test_step_num_episodes=5` | 每次测试阶段的回合数 |

#### `rl_evaluate(+Name, +NumEpisodes, -Metrics)`
对 Agent 进行固定回合数的评估。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 已注册 Agent 的标识符 |
| NumEpisodes | 整数 | 评估回合数 |
| Metrics | 句柄 | 评估指标字典的句柄 |

#### `rl_info(+Name, -Info)`
返回已注册 Agent 的元数据。

| 参数 | 类型 | 说明 |
|---|---|---|
| Name | 原子 | 已注册 Agent 的标识符 |
| Info | 句柄 | 信息字典的句柄 |

**示例：**
```prolog
:- use_module('prolog/scryer_py').
:- use_module('prolog/scryer_rl').

rl_demo :-
    py_init,
    rl_create(agent, "CartPole-v1", dqn,
              [lr=0.001, hidden_sizes=[64,64]]),
    rl_train(agent, [max_epochs=5, epoch_num_steps=2000], Metrics),
    py_to_str(Metrics, MetricsStr),
    format("训练指标: ~s~n", [MetricsStr]),
    rl_evaluate(agent, 10, EvalMetrics),
    py_to_str(EvalMetrics, EvalStr),
    format("评估指标: ~s~n", [EvalStr]),
    rl_save(agent, "checkpoints/cartpole_dqn.pt"),
    py_free(Metrics), py_free(EvalMetrics), py_finalize.

## 语法糖：`:=` 运算符

`:=` 运算符能够让常用的操作变得更加简洁。它使用了一种“三分发（3-way dispatch）”机制来区分不同的操作类型。

### Scryer Prolog 中的类型识别
- `"hello"` 是字符串（即字符列表）。
- `hello` 是原子（符号常量）。
- `hello(X)` 是复合项（一个原子后跟若干参数）。

### `:=` 的分发逻辑
1. **`Var := Obj:"attrname"`**：冒号右侧是**字符串**时，执行**属性访问**。
   - 等价于：`py_getattr(Obj, "attrname", Var)`
   - 示例：`Pi := Math:"pi"`
2. **`Var := Obj:methodname`**：冒号右侧是**原子**时，执行**无参数的方法调用**。
   - 等价于：`py_call(Obj, "methodname", Var)`
   - 示例：`U := S:upper`（相当于 `s.upper()`）
3. **`Var := Obj:method(Arg1, Arg2, ...)`**：冒号右侧是**复合项**时，执行**带参数的方法调用**。
   - 等价于：分发给 `py_call/4..6` 或 `py_calln/4`。
   - 示例：`R := S:replace(Old, New)`（相当于 `s.replace(old, new)`）

### 内置快捷方式
- `X := py_eval("expr")` → `py_eval("expr", X)`
- `M := py_import("mod")` → `py_import("mod", M)`
- `H := py_from_int(42)` → `py_from_int(42, H)`

### 什么时候**不该**使用 `:=`
`:=` 运算符不支持直接包装所有的方法。例如，以下写法是**错误**的：
```prolog
%% 错误示范
X := py_call(Obj, "method", Result).
R := py_invoke(Fn, Arg).
:= py_setattr(Obj, "name", Val).
```
如果是上述情况，请直接使用原生的 `py_call`、`py_invoke` 或 `py_setattr` 谓词。

---

## 常用设计模式 (Common Patterns)

**模式 1：批量数据处理**
在 Python 中定义处理函数，将 Prolog 数据转换进去处理后，再通过 JSON 桥接取回。这比在 Prolog 循环中频繁跨语言调用性能更好。
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

batch_process(PrologList, Results) :-
    py_init,
    py_exec_lines([
        "def process_batch(items):",
        "    return [x ** 2 + 1 for x in items]"
    ]),
    py_list_new(PyList),
    maplist(add_to_list(PyList), PrologList),
    
    py_eval("process_batch", Fn),
    py_invoke(Fn, PyList, ResultH),
    
    py_to_json(ResultH, Json),
    format("处理结果: ~s~n", [Json]),
    maplist(py_free, [PyList, Fn, ResultH]).

add_to_list(PyList, Val) :-
    py_from_int(Val, H),
    py_list_append(PyList, H),
    py_free(H).
```

**模式 2：容错处理流水线**
```prolog
:- use_module('prolog/scryer_py').

%% 一个在每个阶段都能从容处理错误的流水线
safe_pipeline :-
    py_init,
    catch(
        pipeline_body,
        error(python_error(Msg), _),
        format("流水线执行失败: ~s~n", [Msg])
    ),
    py_finalize.

pipeline_body :-
    py_exec("import json"),
    
    %% 第一阶段：加载数据
    py_eval("json.loads('{\"values\": [1, 2, 3]}')", Data),
    with_py(Data, (
        %% 第二阶段：提取值并验证
        py_getattr(Data, "__class__", _),  % 验证对象是否有效
        py_to_json(Data, Json),
        format("解析出的数据: ~s~n", [Json])
    )).
```

**模式 3：配合 NumPy 进行科学计算（如已安装）**
```prolog
:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').

numpy_demo :-
    py_init,
    NP := py_import("numpy"),
    
    %% 通过 py_eval 创建一个 NumPy 数组
    Arr := py_eval("__import__('numpy').array([1.0, 2.0, 3.0, 4.0, 5.0])"),
    
    %% 调用 numpy 的 mean 方法
    Mean := NP:mean(Arr),
    py_to_float(Mean, MeanVal),
    format("平均值: ~f~n", [MeanVal]),
    
    %% 调用 numpy 的 std 方法
    Std := NP:std(Arr),
    py_to_float(Std, StdVal),
    format("标准差: ~f~n", [StdVal]),
    
    %% 计算点积 (Dot product)
    Arr2 := py_eval("__import__('numpy').array([2.0, 0.0, 1.0, 0.0, 3.0])"),
    Dot := NP:dot(Arr, Arr2),
    py_to_float(Dot, DotVal),
    format("点积: ~f~n", [DotVal]),
    
    maplist(py_free, [NP, Arr, Mean, Std, Arr2, Dot]).
```
---

## 示例

| 文件 | 说明 |
|---|---|
| `examples/basic.pl` | 算术运算、模块导入、集合操作、错误处理、RAII 自动清理 |
| `examples/neural.pl` | MNIST 分类、神经符号加法、LLM 集成、强化学习 Agent |
| `examples/numpy_torch.pl` | NumPy 向量/矩阵、PyTorch 张量、线性回归、CUDA GPU 矩阵乘法 |
| `examples/mnist_cnn.pl` | 从零训练 CNN 识别 MNIST 手写数字 —— 模型定义、训练循环、评估、神经符号推理 |
| `examples/mnist_cnn_v2.pl` | **模块模式**（推荐）：同样的 CNN 训练，但 Python 代码放在独立 `.py` 文件中 |
| `examples/rl_demo.pl` | DQN Agent 训练 CartPole-v1 —— 创建、训练、评估、保存、加载 |

```bash
# 运行所有示例
LD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
LD_LIBRARY_PATH=. scryer-prolog examples/neural.pl
LD_LIBRARY_PATH=. scryer-prolog examples/numpy_torch.pl
LD_LIBRARY_PATH=. scryer-prolog examples/mnist_cnn.pl
LD_LIBRARY_PATH=. scryer-prolog examples/mnist_cnn_v2.pl
LD_LIBRARY_PATH=. scryer-prolog examples/rl_demo.pl
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
│   ├── scryer_py.pl        # 核心模块：py_* 谓词 + := 运算符
│   ├── scryer_nn.pl        # 插件：nn_load, nn_predict
│   ├── scryer_llm.pl       # 插件：llm_load, llm_generate
│   └── scryer_rl.pl        # 插件：rl_create, rl_train, rl_action, ...
├── python/
│   ├── scryer_py_runtime.py  # 核心运行时：设备管理、TensorUtils
│   ├── scryer_nn_runtime.py  # NN 运行时：模型加载与推理
│   ├── scryer_llm_runtime.py # LLM 运行时：提供商抽象层
│   └── scryer_rl_runtime.py  # RL 运行时：Tianshou v2.0 Agent 封装
├── examples/
│   ├── basic.pl            # 基础交互示例
│   ├── neural.pl           # 神经符号模式示例（NN + LLM + RL）
│   ├── numpy_torch.pl      # NumPy + PyTorch + CUDA 示例
│   ├── mnist_cnn.pl        # CNN MNIST 训练流水线（内联 Python）
│   ├── mnist_cnn_v2.pl     # CNN MNIST 训练流水线（模块模式）
│   ├── mnist_cnn_module.py # mnist_cnn_v2.pl 的 Python 模块
│   └── rl_demo.pl          # RL 示例：DQN on CartPole-v1
├── test_comprehensive.pl   # 24 项底层 FFI 测试
├── test_prolog_api.pl      # 19 项高层 API 测试
├── test_minimal_api.pl     # 3 项核心冒烟测试
├── test_rl.pl              # 17 项 RL 插件测试（scryer_rl.pl）
├── test_rl.py              # 15 项 Python RL 运行时测试（scryer_rl_runtime.py）
└── docs/
    └── technical_report.md # 详细中文技术报告
```

## 开源协议

MIT
