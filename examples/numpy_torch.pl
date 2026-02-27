%% ===========================================================================
%% ScryNeuro NumPy + PyTorch Example
%% ===========================================================================
%%
%% 演示如何通过 ScryNeuro 在 Scryer Prolog 中直接调用 NumPy 和 PyTorch。
%% 核心思路：Python 对象以"句柄"整数的形式存活在 Rust 端，Prolog 只拿着 ID
%% 操作，真正的张量数据从不离开 Python 的内存空间。
%%
%% Run:
%%   $ LD_LIBRARY_PATH=. scryer-prolog examples/numpy_torch.pl
%%
%% Prerequisites:
%%   - cargo build --release && cp target/release/libscryneuro.so ./
%%   - conda activate py312  (or any env with numpy + torch installed)

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').

%% ---------------------------------------------------------------------------
%% Example 1: NumPy —— 向量点积
%% ---------------------------------------------------------------------------

example_numpy_dot :-
    py_init,
    format("=== NumPy: 向量点积 ===~n", []),

    %% import numpy as np
    py_import("numpy", NP),

    %% 用 py_exec + eval 建立两个 ndarray
    %% 注意：Scryer 不支持 \n 转义，我们用 py_exec_lines/1 分行传
    py_exec_lines([
        "import numpy as np",
        "a = np.array([1.0, 2.0, 3.0, 4.0])",
        "b = np.array([10.0, 20.0, 30.0, 40.0])"
    ]),
    py_eval("float(np.dot(a, b))", DotH),
    py_to_float(DotH, Dot),
    format("dot([1,2,3,4], [10,20,30,40]) = ~f~n", [Dot]),
    py_free(DotH),

    %% 矩阵乘法：2x2 @ 2x2
    py_exec_lines([
        "M = np.array([[1.0,2.0],[3.0,4.0]])",
        "N = np.array([[5.0,6.0],[7.0,8.0]])",
        "MN = (M @ N).tolist()"
    ]),
    py_eval("str(MN)", StrH),
    py_to_str(StrH, MatStr),
    format("[[1,2],[3,4]] @ [[5,6],[7,8]] = ~s~n", [MatStr]),
    py_free(StrH),

    %% 统计量
    py_exec_lines([
        "data = np.random.seed(42)",
        "data = np.random.randn(1000)",
        "result_mean = float(data.mean())",
        "result_std  = float(data.std())"
    ]),
    py_eval("result_mean", MeanH), py_to_float(MeanH, Mean), py_free(MeanH),
    py_eval("result_std",  StdH),  py_to_float(StdH, Std),   py_free(StdH),
    format("randn(1000): mean=~f  std=~f~n", [Mean, Std]),

    py_free(NP).


%% ---------------------------------------------------------------------------
%% Example 2: PyTorch —— 张量运算与自动微分
%% ---------------------------------------------------------------------------

example_torch_tensor :-
    py_init,
    format("~n=== PyTorch: 张量运算 ===~n", []),

    py_exec_lines([
        "import torch",
        "x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])",
        "y = x * x"
    ]),

    %% L2 norm
    py_eval("float(x.norm())", NormH),
    py_to_float(NormH, Norm),
    format("norm([1..5]) = ~f~n", [Norm]),
    py_free(NormH),

    %% 元素乘法后求和
    py_eval("float(y.sum())", SumH),
    py_to_float(SumH, SumV),
    format("sum(x*x) for x in [1..5] = ~f~n", [SumV]),
    py_free(SumH),

    %% 判断设备（CPU / CUDA）
    py_eval("str(x.device)", DevH),
    py_to_str(DevH, Dev),
    format("tensor device: ~s~n", [Dev]),
    py_free(DevH).


%% ---------------------------------------------------------------------------
%% Example 3: PyTorch —— 简单线性回归（梯度下降 10 步）
%% ---------------------------------------------------------------------------

example_torch_regression :-
    py_init,
    format("~n=== PyTorch: 线性回归（10步梯度下降）===~n", []),

    %% 目标：学习 y = 2x + 1
    py_exec_lines([
        "import torch",
        "torch.manual_seed(0)",
        "X_data = torch.linspace(-1, 1, 20).unsqueeze(1)",
        "Y_data = 2.0 * X_data + 1.0 + 0.05 * torch.randn(20, 1)",
        "W = torch.zeros(1, 1, requires_grad=True)",
        "b = torch.zeros(1,    requires_grad=True)",
        "lr = 0.1"
    ]),

    %% 10 步梯度下降，纯 Python 执行，Prolog 只读最终结果
    py_exec_lines([
        "for _ in range(200):",
        "    pred = X_data @ W + b",
        "    loss = ((pred - Y_data)**2).mean()",
        "    loss.backward()",
        "    with torch.no_grad():",
        "        W -= lr * W.grad",
        "        b -= lr * b.grad",
        "    W.grad.zero_()",
        "    b.grad.zero_()"
    ]),

    %% 读取拟合结果
    py_eval("float(W.item())", WH), py_to_float(WH, WVal), py_free(WH),
    py_eval("float(b.item())", BH), py_to_float(BH, BVal), py_free(BH),
    py_eval("float(loss.item())", LH), py_to_float(LH, LVal), py_free(LH),

    format("拟合结果：W=~f  b=~f  (期望 W≈2.0, b≈1.0)~n", [WVal, BVal]),
    format("最终 MSE loss: ~f~n", [LVal]).


%% ---------------------------------------------------------------------------
%% Example 4: CUDA 可用时在 GPU 上做矩阵乘法
%% ---------------------------------------------------------------------------

example_cuda_matmul :-
    py_init,
    format("~n=== CUDA: GPU 矩阵乘法（如不可用则跳过）===~n", []),

    %% 用 int(bool) 转成 0/1，避免 =:= 与 Prolog 原子 true 冲突
    py_eval("int(__import__('torch').cuda.is_available())", AvailH),
    py_to_int(AvailH, Avail),
    py_free(AvailH),

    ( Avail =:= 1 ->
        py_exec_lines([
            "import torch",
            "dev = 'cuda'",
            "A = torch.randn(512, 512, device=dev)",
            "B = torch.randn(512, 512, device=dev)",
            "C = torch.mm(A, B)",
            "cuda_result = float(C.abs().mean())"
        ]),
        py_eval("cuda_result", ResH),
        py_to_float(ResH, Res),
        py_free(ResH),
        format("CUDA 512×512 matmul 平均绝对值: ~f~n", [Res])
    ;
        format("CUDA 不可用，跳过 GPU 示例。~n", [])
    ).


%% ---------------------------------------------------------------------------
%% 运行全部示例
%% ---------------------------------------------------------------------------

:- initialization((
    example_numpy_dot,
    example_torch_tensor,
    example_torch_regression,
    example_cuda_matmul,
    nl,
    format("=== numpy_torch 示例全部完成 ===~n", []),
    py_finalize
)).
