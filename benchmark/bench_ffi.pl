:- use_module('../prolog/scryer_py').
:- use_module(library(format)).
:- use_module(library(lists)).
:- use_module(library(os)).
:- use_module(library(iso_ext)).

warmup(100).
rounds(5).
default_iterations(10000).

iteration_count(N) :-
    argv(Args),
    ( Args = [] -> default_iterations(N)
    ; Args = [Chars] ->
        catch(number_chars(N, Chars), _,
              throw(error(domain_error(positive_integer, Chars), iteration_count/1))),
        ( integer(N), N > 0 -> true
        ; throw(error(domain_error(positive_integer, Chars), iteration_count/1))
        )
    ; throw(error(domain_error(benchmark_arguments, Args), iteration_count/1))
    ).

get_time_us(Us) :-
    with_py_temp(py_eval("__import__('time').perf_counter() * 1000000", H), H,
                 py_to_float(H, Us)).

run_n_times(0, _).
run_n_times(N, Goal) :-
    N > 0,
    call(Goal),
    Next is N - 1,
    run_n_times(Next, Goal).

measure_samples(0, _, _, []).
measure_samples(Rounds, N, Goal, [Average | Rest]) :-
    Rounds > 0,
    get_time_us(Start),
    run_n_times(N, Goal),
    get_time_us(End),
    Average is (End - Start) / N,
    Next is Rounds - 1,
    measure_samples(Next, N, Goal, Rest).

sample_pairs([], []).
sample_pairs([Sample | Rest], [Sample-Sample | Pairs]) :-
    sample_pairs(Rest, Pairs).

bench_operation(Name, N, Goal) :-
    warmup(Warmup),
    rounds(Rounds),
    run_n_times(Warmup, Goal),
    measure_samples(Rounds, N, Goal, Samples),
    sample_pairs(Samples, Pairs),
    keysort(Pairs, Sorted),
    nth0(0, Sorted, _-Min),
    nth0(2, Sorted, _-Median),
    nth0(4, Sorted, _-Max),
    format("~w: median=~2f us/op, min=~2f us/op, max=~2f us/op~n",
           [Name, Median, Min, Max]).

py_eval_free(Code) :-
    py_eval(Code, H),
    py_free(H).

py_method_call_free(String) :-
    py_call(String, "upper", Result),
    py_free(Result).

py_import_attr :-
    py_import("math", Module),
    py_getattr(Module, "pi", Value),
    py_free(Value),
    py_free(Module).

py_json_roundtrip(Data) :-
    py_to_json(Data, Json),
    py_from_json(Json, Result),
    py_free(Result).

run_suite(N) :-
    setup_call_cleanup(py_init, (
        warmup(Warmup),
        rounds(Rounds),
        format("=== ScryNeuro FFI (N=~d, warmup=~d, rounds=~d) ===~n",
               [N, Warmup, Rounds]),
        bench_operation(int_add, N, py_eval_free("1 + 1")),
        bench_operation(float_mul, N, py_eval_free("1.5 * 2.5")),
        bench_operation(str_concat, N, py_eval_free("'hello' + 'world'")),
        bench_operation(list_create, N, py_eval_free("[1, 2, 3, 4, 5]")),
        bench_operation(builtin_call, N, py_eval_free("len([1, 2, 3, 4, 5])")),
        with_py_temp(py_from_str("hello world", String), String,
            once(bench_operation(method_call, N, py_method_call_free(String)))),
        bench_operation(import_attr, N, py_import_attr),
        with_py_temp(py_eval("{'name': 'test', 'value': 42}", Data), Data,
            once(bench_operation(json_roundtrip, N, py_json_roundtrip(Data)))),
        py_handle_count(Count),
        ( Count =:= 0 -> true
        ; throw(error(handle_leak(Count), run_suite/1))
        )
    ), py_finalize).

main :-
    ( catch((iteration_count(N), run_suite(N)), Error,
            ( format("FFI benchmark failed: ~q~n", [Error]), halt(1) )) ->
        halt(0)
    ; format("FFI benchmark failed without an exception.~n", []), halt(1)
    ).

:- initialization(main).
