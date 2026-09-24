%% Minimal Prolog API test
:- use_module('../prolog/scryer_py').
:- use_module(library(iso_ext)).

test_eval :-
    py_eval("2 ** 10", H),
    py_to_int(H, V),
    ( V =:= 1024 ->
        write("1. py_eval: OK"), nl
    ; throw(error(test_failed(py_eval), test_minimal_api/0))
    ),
    py_free(H).

test_py_call_0 :-
    py_from_str("hello world", S),
    py_call(S, "upper", Result),
    py_to_str(Result, Str),
    write("2. py_call/3: "), write(Str), nl,
    ( Str = "HELLO WORLD" -> true
    ; throw(error(test_failed(py_call), test_minimal_api/0))
    ),
    py_free(Result),
    py_free(S).

test_py_invoke_0 :-
    py_eval("list", ListClass),
    py_invoke(ListClass, Result),
    py_list_len(Result, Len),
    write("3. py_invoke/2: len="), write(Len), nl,
    ( Len =:= 0 -> true
    ; throw(error(test_failed(py_invoke), test_minimal_api/0))
    ),
    py_free(Result),
    py_free(ListClass).

run_tests :-
    ( catch(
        setup_call_cleanup(py_init, (
            test_eval,
            test_py_call_0,
            test_py_invoke_0,
            write("=== ALL 3 TESTS PASSED ==="), nl
        ), py_finalize),
        Error,
        ( write('Minimal API tests failed: '), writeq(Error), nl, halt(1) )
      ) -> halt(0)
    ; write('Minimal API tests failed without an exception.'), nl, halt(1)
    ).

:- initialization(run_tests).
