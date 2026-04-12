:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module(library(format)).

run :- bench(3).

get_time_ms(Ms) :-
    py_eval("__import__('time').perf_counter() * 1000", H),
    py_to_float(H, F),
    Ms is round(F),
    py_free(H).

bench(N) :-
    py_init,
    format("=== MNIST CNN Benchmark (FFI) ===~n~n", []),
    py_exec("import sys; sys.path.insert(0, 'examples')"),
    
    py_import("mnist_cnn_module", M),
    py_call(M, "create_pipeline", P),
    py_free(M),
    
    get_time_ms(T2s),
    py_call(P, "load_data", I),
    get_time_ms(T2e),
    D2 is T2e - T2s,
    py_dict_get(I, "train_size", H1),
    py_to_int(H1, Tr),
    py_free(H1),
    py_free(I),
    format("Loaded ~d samples, time: ~3f s~n", [Tr, D2/1000]),
    
    get_time_ms(T3s),
    py_call(P, "setup", D),
    get_time_ms(T3e),
    D3 is T3e - T3s,
    py_to_str(D, Dev),
    py_free(D),
    format("Device: ~s, setup time: ~3f s~n~n", [Dev, D3/1000]),
    
    format("Training ~d epochs...~n", [N]),
    get_time_ms(T4s),
    train(P, 1, N),
    get_time_ms(T4e),
    D4 is T4e - T4s,
    
    get_time_ms(T5s),
    py_call(P, "evaluate", A),
    get_time_ms(T5e),
    D5 is T5e - T5s,
    py_to_float(A, Acc),
    py_free(A),
    format("~nTest accuracy: ~2f, eval time: ~3f s~n", [Acc, D5/1000]),
    
    py_free(P),
    
    format("~n=== Summary ===~n", []),
    format("Setup time:    ~3f s~n", [D3/1000]),
    format("Train time:    ~3f s (~d epochs)~n", [D4/1000, N]),
    format("Eval time:     ~3f s~n", [D5/1000]),
    
    py_finalize.

train(_, E, Max) :- E > Max, !.
train(P, E, Max) :-
    py_call(P, "train_one_epoch", S),
    py_dict_get(S, "loss", Lh),
    py_to_float(Lh, Loss),
    py_free(Lh),
    py_dict_get(S, "accuracy", Ah),
    py_to_float(Ah, Acc),
    py_free(Ah),
    py_free(S),
    format("Epoch ~d: loss=~4f acc=~2f~n", [E, Loss, Acc]),
    E1 is E + 1,
    train(P, E1, Max).

:- initialization(run).