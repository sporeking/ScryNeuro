%% ===========================================================================
%% ScryNeuro Example: CNN MNIST Training (Separated Python Module)
%% ===========================================================================
%%
%% This example demonstrates the RECOMMENDED pattern for ScryNeuro projects:
%%
%%   Python code lives in .py files  →  full IDE support, linting, type-checking
%%   Prolog orchestrates the pipeline →  clean logic, no embedded strings
%%
%% Compare with mnist_cnn.pl (inline Python strings) to see the difference.
%%
%% Prerequisites:
%%   - Build: cargo build --release
%%   -        cp target/release/libscryneuro.so ./     # Linux
%%   -        cp target/release/libscryneuro.dylib ./  # macOS
%%   - Python deps: pip install torch torchvision
%%
%% Run:
%%   $ LD_LIBRARY_PATH=. scryer-prolog examples/mnist_cnn_v2.pl
%%

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').

run_example(Name, Goal) :-
    ( catch(Goal, E, (format("[ERROR] ~s failed: ~q~n", [Name, E]), fail)) ->
        format("[OK] ~s~n", [Name])
    ; true
    ).

%% ---------------------------------------------------------------------------
%% Helper: add examples/ to Python's module search path so that
%% py_import("mnist_cnn_module") can find examples/mnist_cnn_module.py
%% ---------------------------------------------------------------------------

add_examples_to_path :-
    py_exec("import sys; sys.path.insert(0, 'examples')").

%% ---------------------------------------------------------------------------
%% Dict helpers: extract typed values from a Python dict
%% ---------------------------------------------------------------------------
%%
%% py_dict_get/3 returns a handle. These wrappers convert + free in one step.

dict_int(Dict, Key, Value) :-
    with_py_temp(py_dict_get(Dict, Key, H), H, py_to_int(H, Value)).

dict_float(Dict, Key, Value) :-
    with_py_temp(py_dict_get(Dict, Key, H), H, py_to_float(H, Value)).

%% ---------------------------------------------------------------------------
%% Step 1: Create the pipeline object
%% ---------------------------------------------------------------------------
%%
%% All Python state (model, device, loaders) lives inside the Pipeline object.
%% Prolog holds a single handle to it.

create_pipeline(Pipeline) :-
    with_py_temp(py_import("mnist_cnn_module", Mod), Mod,
        py_call(Mod, "create_pipeline", Pipeline)),
    format("[Step 1] Pipeline created (model defined).~n", []).

%% ---------------------------------------------------------------------------
%% Step 2: Load MNIST dataset
%% ---------------------------------------------------------------------------

load_data(Pipeline) :-
    with_py_temp(py_call(Pipeline, "load_data", Info), Info, (
        dict_int(Info, "train_size", TrSize),
        dict_int(Info, "test_size", TeSize)
    )),
    format("[Step 2] MNIST loaded: ~d training / ~d test samples.~n",
           [TrSize, TeSize]).

%% ---------------------------------------------------------------------------
%% Step 3: Setup model, optimizer, loss; detect device
%% ---------------------------------------------------------------------------

setup_training(Pipeline) :-
    with_py_temp(py_call(Pipeline, "setup", DevH), DevH, py_to_str(DevH, Dev)),
    format("[Step 3] Training on device: ~s~n", [Dev]).

%% ---------------------------------------------------------------------------
%% Step 4: Training loop — Prolog controls epochs, Python runs batches
%% ---------------------------------------------------------------------------

train_one_epoch(Pipeline, Epoch) :-
    with_py_temp(py_call(Pipeline, "train_one_epoch", Stats), Stats, (
        dict_float(Stats, "loss", Loss),
        dict_float(Stats, "accuracy", Acc)
    )),
    format("  Epoch ~d — loss: ~4f  train_acc: ~2f%~n", [Epoch, Loss, Acc]).

train_epochs(_Pipeline, Current, Max) :-
    Current > Max, !.
train_epochs(Pipeline, Current, Max) :-
    train_one_epoch(Pipeline, Current),
    Next is Current + 1,
    train_epochs(Pipeline, Next, Max).

%% ---------------------------------------------------------------------------
%% Step 5: Evaluate on test set
%% ---------------------------------------------------------------------------

evaluate(Pipeline, TestAcc) :-
    with_py_temp(py_call(Pipeline, "evaluate", AccH), AccH,
        py_to_float(AccH, TestAcc)).

%% ---------------------------------------------------------------------------
%% Step 6: Neural predicate — single-image inference
%% ---------------------------------------------------------------------------
%%
%% digit/3 is the neuro-symbolic payoff: a Prolog predicate backed by
%% a trained neural network. It maps an image index to its predicted class.

digit(Pipeline, TestIndex, PredictedClass) :-
    with_py_temp(py_from_int(TestIndex, IdxH), IdxH,
        with_py_temp(py_call(Pipeline, "predict_digit", IdxH, PredH), PredH,
            py_to_int(PredH, PredictedClass))).

true_label(Pipeline, TestIndex, Label) :-
    with_py_temp(py_from_int(TestIndex, IdxH), IdxH,
        with_py_temp(py_call(Pipeline, "true_label", IdxH, LabelH), LabelH,
            py_to_int(LabelH, Label))).

%% Neuro-symbolic addition: two images → sum of predicted digits
neuro_add(Pipeline, Idx1, Idx2, Sum) :-
    digit(Pipeline, Idx1, D1),
    digit(Pipeline, Idx2, D2),
    Sum is D1 + D2.

%% Demo: run inference on first 10 test images
demo_inference(Pipeline) :-
    format("~n[Step 6] Single-image inference (neural predicate):~n", []),
    demo_inference_loop(Pipeline, 0, 10).

demo_inference_loop(_Pipeline, I, Max) :-
    I >= Max, !.
demo_inference_loop(Pipeline, I, Max) :-
    digit(Pipeline, I, Pred),
    true_label(Pipeline, I, Truth),
    ( Pred =:= Truth ->
        Mark = "✓"
    ;
        Mark = "✗"
    ),
    format("  test[~d]: predicted=~d  truth=~d  ~s~n", [I, Pred, Truth, Mark]),
    Next is I + 1,
    demo_inference_loop(Pipeline, Next, Max).

%% ---------------------------------------------------------------------------
%% Step 7: Save model
%% ---------------------------------------------------------------------------

save_model(Pipeline, Path) :-
    with_py_temp(py_from_str(Path, PathH), PathH,
        with_py_temp(py_call(Pipeline, "save_model", PathH, SaveResult), SaveResult, true)),
    format("[Step 7] Model saved to: ~s~n", [Path]).

%% ---------------------------------------------------------------------------
%% Main: orchestrate the full pipeline
%% ---------------------------------------------------------------------------

main :-
    format("========================================~n", []),
    format(" ScryNeuro: CNN MNIST (Module Pattern) ~n", []),
    format("========================================~n~n", []),

    %% Make examples/ importable
    add_examples_to_path,

    %% Create pipeline (defines model class + instantiates pipeline)
    create_pipeline(Pipeline),

    %% Load data
    load_data(Pipeline),

    %% Setup model/optimizer/device
    setup_training(Pipeline),

    %% Train 3 epochs
    format("~n[Step 4] Training (3 epochs):~n", []),
    train_epochs(Pipeline, 1, 3),

    %% Evaluate
    format("~n[Step 5] Evaluating on test set...~n", []),
    evaluate(Pipeline, TestAcc),
    format("  Test accuracy: ~2f%~n", [TestAcc]),

    %% Neural predicate demo
    demo_inference(Pipeline),

    %% Neuro-symbolic addition demo
    format("~n[Bonus] Neuro-symbolic addition:~n", []),
    neuro_add(Pipeline, 0, 1, Sum01),
    true_label(Pipeline, 0, L0),
    true_label(Pipeline, 1, L1),
    format("  digit(test[0])=~d + digit(test[1])=~d => sum=~d  (truth: ~d+~d=",
           [L0, L1, Sum01, L0, L1]),
    TrueSum is L0 + L1,
    format("~d)~n", [TrueSum]),

    %% Save
    save_model(Pipeline, "examples/mnist_cnn_v2.pt"),

    %% Cleanup — free the pipeline handle
    py_free(Pipeline),

    format("~n========================================~n", []),
    format(" Pipeline complete!~n", []),
    format("========================================~n", []).

:- initialization((
    py_init,
    run_example("main", main),
    py_finalize
)).
