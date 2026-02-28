%% ===========================================================================
%% ScryNeuro NumPy + PyTorch Example
%% ===========================================================================
%%
%% Demonstrates how to call NumPy and PyTorch directly in Scryer Prolog via ScryNeuro.
%% Core idea: Python objects live on the Rust side as integer "handles", Prolog only operates
%% with IDs, and the real tensor data never leaves Python's memory space.
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
%% Example 1: NumPy - Vector Dot Product
%% ---------------------------------------------------------------------------

example_numpy_dot :-
    py_init,
    format("=== NumPy: Vector Dot Product ===~n", []),

    %% import numpy as np
    py_import("numpy", NP),

    %% Create two ndarrays using py_exec + eval
    %% Note: Scryer doesn't support \n escapes, so we pass multiple lines using py_exec_lines/1
    py_exec_lines([
        "import numpy as np",
        "a = np.array([1.0, 2.0, 3.0, 4.0])",
        "b = np.array([10.0, 20.0, 30.0, 40.0])"
    ]),
    py_eval("float(np.dot(a, b))", DotH),
    py_to_float(DotH, Dot),
    format("dot([1,2,3,4], [10,20,30,40]) = ~f~n", [Dot]),
    py_free(DotH),

    %% Matrix multiplication: 2x2 @ 2x2
    py_exec_lines([
        "M = np.array([[1.0,2.0],[3.0,4.0]])",
        "N = np.array([[5.0,6.0],[7.0,8.0]])",
        "MN = (M @ N).tolist()"
    ]),
    py_eval("str(MN)", StrH),
    py_to_str(StrH, MatStr),
    format("[[1,2],[3,4]] @ [[5,6],[7,8]] = ~s~n", [MatStr]),
    py_free(StrH),

    %% Statistics
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
%% Example 2: PyTorch - Tensor Operations and Auto-differentiation
%% ---------------------------------------------------------------------------

example_torch_tensor :-
    py_init,
    format("~n=== PyTorch: Tensor Operations ===~n", []),

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

    %% Element-wise multiplication followed by sum
    py_eval("float(y.sum())", SumH),
    py_to_float(SumH, SumV),
    format("sum(x*x) for x in [1..5] = ~f~n", [SumV]),
    py_free(SumH),

    %% Check device (CPU / CUDA)
    py_eval("str(x.device)", DevH),
    py_to_str(DevH, Dev),
    format("tensor device: ~s~n", [Dev]),
    py_free(DevH).


%% ---------------------------------------------------------------------------
%% Example 3: PyTorch - Simple Linear Regression (Gradient Descent)
%% ---------------------------------------------------------------------------

example_torch_regression :-
    py_init,
    format("~n=== PyTorch: Linear Regression (Gradient Descent) ===~n", []),

    %% Goal: learn y = 2x + 1
    py_exec_lines([
        "import torch",
        "torch.manual_seed(0)",
        "X_data = torch.linspace(-1, 1, 20).unsqueeze(1)",
        "Y_data = 2.0 * X_data + 1.0 + 0.05 * torch.randn(20, 1)",
        "W = torch.zeros(1, 1, requires_grad=True)",
        "b = torch.zeros(1,    requires_grad=True)",
        "lr = 0.1"
    ]),

    %% Gradient descent, pure Python execution, Prolog only reads the final result
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

    %% Read fitting results
    py_eval("float(W.item())", WH), py_to_float(WH, WVal), py_free(WH),
    py_eval("float(b.item())", BH), py_to_float(BH, BVal), py_free(BH),
    py_eval("float(loss.item())", LH), py_to_float(LH, LVal), py_free(LH),

    format("Fitting results: W=~f  b=~f  (Expected W≈2.0, b≈1.0)~n", [WVal, BVal]),
    format("Final MSE loss: ~f~n", [LVal]).


%% ---------------------------------------------------------------------------
%% Example 4: GPU Matrix Multiplication (if CUDA is available)
%% ---------------------------------------------------------------------------

example_cuda_matmul :-
    py_init,
    format("~n=== CUDA: GPU Matrix Multiplication (skip if unavailable) ===~n", []),

    %% Use int(bool) to convert to 0/1, avoiding =:= conflict with Prolog atom true
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
        format("CUDA 512x512 matmul mean absolute value: ~f~n", [Res])
    ;
        format("CUDA is unavailable, skipping GPU example.~n", [])
    ).


%% ---------------------------------------------------------------------------
%% Run all examples
%% ---------------------------------------------------------------------------

:- initialization((
    example_numpy_dot,
    example_torch_tensor,
    example_torch_regression,
    example_cuda_matmul,
    nl,
    format("=== numpy_torch examples all completed ===~n", []),
    py_finalize
)).
