%% Minimal Prolog API test
:- use_module('prolog/scryer_py').

test_eval :-
    py_eval("2 ** 10", H),
    py_to_int(H, V),
    ( V =:= 1024 ->
        write("1. py_eval: OK"), nl
    ; write("1. py_eval: FAIL"), nl
    ),
    py_free(H).

test_py_call_0 :-
    py_from_str("hello world", S),
    py_call(S, "upper", Result),
    py_to_str(Result, Str),
    write("2. py_call/3: "), write(Str), nl,
    py_free(Result),
    py_free(S).

test_py_invoke_0 :-
    py_eval("list", ListClass),
    py_invoke(ListClass, Result),
    py_list_len(Result, Len),
    write("3. py_invoke/2: len="), write(Len), nl,
    py_free(Result),
    py_free(ListClass).

:- initialization((
    py_init,
    test_eval,
    test_py_call_0,
    test_py_invoke_0,
    write("=== DONE ==="), nl,
    py_finalize
)).
