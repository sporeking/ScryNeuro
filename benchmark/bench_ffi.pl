:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module(library(format)).
:- use_module(library(lists)).
:- use_module(library(time)).

default_iterations(10000).

%% Use wall-clock time via Python's perf_counter for fair comparison
%% This matches bench_native.py which uses time.perf_counter()
get_time_ms(Ms) :-
    py_eval("__import__('time').perf_counter() * 1000", H),
    py_to_float(H, MsFloat),
    Ms is round(MsFloat),
    py_free(H).

elapsed_ms(T0, T1, Delta) :-
    Delta is T1 - T0.

run_n_times(0, _Op).
run_n_times(N, Op) :-
    N > 0,
    call(Op),
    N1 is N - 1,
    run_n_times(N1, Op).

bench_int_add(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_eval_free("1 + 1")),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("int_add: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

py_eval_free(Code) :-
    py_eval(Code, H),
    py_free(H).

bench_float_mul(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_eval_free("1.5 * 2.5")),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("float_mul: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

bench_str_concat(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_eval_free("'hello' + 'world'")),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("str_concat: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

bench_list_create(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_eval_free("[1, 2, 3, 4, 5]")),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("list_create: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

bench_builtin_call(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_eval_free("len([1, 2, 3, 4, 5])")),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("builtin_call: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

py_method_call_free(S) :-
    py_call(S, "upper", R),
    py_free(R).

bench_method_call(N) :-
    py_init,
    py_from_str("hello world", S),
    get_time_ms(T0),
    run_n_times(N, py_method_call_free(S)),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("method_call: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_free(S),
    py_finalize.

py_convert_int :-
    py_from_int(42, H),
    py_to_int(H, _),
    py_free(H).

bench_convert_int(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_convert_int),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("convert_int: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

py_convert_float :-
    py_from_float(3.14159, H),
    py_to_float(H, _),
    py_free(H).

bench_convert_float(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_convert_float),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("convert_float: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

py_convert_str :-
    py_from_str("benchmark test string", H),
    py_to_str(H, _),
    py_free(H).

bench_convert_str(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_convert_str),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("convert_str: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

py_json_roundtrip :-
    py_eval("__import__('json').dumps({'name':'test','value':42})", H1),
    py_eval("__import__('json').loads(__import__('json').dumps({'name':'test','value':42}))", H2),
    py_free(H1),
    py_free(H2).

bench_json_roundtrip(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_json_roundtrip),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("json_roundtrip: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

py_import_attr :-
    py_import("math", M),
    py_getattr(M, "pi", Pi),
    py_free(Pi),
    py_free(M).

bench_import_attr(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_import_attr),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("import_attr: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

py_list_ops :-
    py_list_new(L),
    py_from_int(1, A),
    py_list_append(L, A),
    py_free(A),
    py_from_int(2, B),
    py_list_append(L, B),
    py_free(B),
    py_list_get(L, 0, Item),
    py_free(Item),
    py_free(L).

bench_list_ops(N) :-
    py_init,
    get_time_ms(T0),
    run_n_times(N, py_list_ops),
    get_time_ms(T1),
    elapsed_ms(T0, T1, DeltaMs),
    DeltaUs is DeltaMs * 1000,
    AvgUs is DeltaUs / N,
    format("list_ops: total=~d us, avg=~f us/op~n", [DeltaUs, AvgUs]),
    py_finalize.

run_all_benchmarks :-
    default_iterations(N),
    format("=== ScryNeuro FFI Benchmark (N=~d iterations) ===~n", [N]),
    format("Using wall-clock time (Python perf_counter)~n~n", []),
    bench_int_add(N),
    bench_float_mul(N),
    bench_str_concat(N),
    bench_list_create(N),
    bench_builtin_call(N),
    bench_method_call(N),
    bench_convert_int(N),
    bench_convert_float(N),
    bench_convert_str(N),
    bench_json_roundtrip(N),
    bench_import_attr(N),
    bench_list_ops(N),
    format("~n=== Benchmark Complete ===~n", []).

:- initialization(run_all_benchmarks).