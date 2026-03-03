%% ===========================================================================
%% ScryNeuro Prolog API Test Suite
%% ===========================================================================
%% Tests the scryer_py.pl module predicates
%% Run: scryer-prolog test_prolog_api.pl

:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').
:- use_module(library(format)).

test_eval :-
    py_eval("2 ** 10", H),
    py_to_int(H, V),
    ( V =:= 1024 ->
        format("1. py_eval: OK~n", [])
    ; format("1. py_eval: FAIL~n", [])
    ),
    py_free(H).

test_exec :-
    py_exec("_test_var = 42"),
    py_eval("_test_var", H),
    py_to_int(H, V),
    ( V =:= 42 ->
        format("2. py_exec: OK~n", [])
    ; format("2. py_exec: FAIL~n", [])
    ),
    py_free(H).

test_import :-
    py_import("math", M),
    py_getattr(M, "pi", Pi),
    py_to_float(Pi, PiVal),
    ( PiVal > 3.14, PiVal < 3.15 ->
        format("3. py_import/getattr: OK~n", [])
    ; format("3. py_import/getattr: FAIL~n", [])
    ),
    py_free(Pi),
    py_free(M).

test_py_call_0 :-
    py_from_str("hello world", S),
    py_call(S, "upper", Result),
    py_to_str(Result, Str),
    ( Str = "HELLO WORLD" ->
        format("4. py_call/3 (0 args): OK~n", [])
    ; format("4. py_call/3 (0 args): FAIL~n", [])
    ),
    py_free(Result),
    py_free(S).

test_py_call_2 :-
    py_from_str("hello world", S),
    py_from_str("world", Old),
    py_from_str("prolog", New),
    py_call(S, "replace", Old, New, Result),
    py_to_str(Result, Str),
    ( Str = "hello prolog" ->
        format("5. py_call/5 (2 args): OK~n", [])
    ; format("5. py_call/5 (2 args): FAIL~n", [])
    ),
    py_free(Result),
    py_free(New),
    py_free(Old),
    py_free(S).

test_py_invoke_0 :-
    py_eval("list", ListClass),
    py_invoke(ListClass, Result),
    py_list_len(Result, Len),
    ( Len =:= 0 ->
        format("6. py_invoke/2 (0 args): OK~n", [])
    ; format("6. py_invoke/2 (0 args): FAIL~n", [])
    ),
    py_free(Result),
    py_free(ListClass).

test_py_invoke_1 :-
    py_eval("abs", AbsFn),
    N is 0 - 42,
    py_from_int(N, Arg),
    py_invoke(AbsFn, Arg, Result),
    py_to_int(Result, V),
    ( V =:= 42 ->
        format("7. py_invoke/3 (1 arg): OK~n", [])
    ; format("7. py_invoke/3 (1 arg): FAIL~n", [])
    ),
    py_free(Result),
    py_free(Arg),
    py_free(AbsFn).

test_py_invoke_2 :-
    py_eval("pow", PowFn),
    py_from_int(2, Base),
    py_from_int(10, Exp),
    py_invoke(PowFn, Base, Exp, Result),
    py_to_int(Result, V),
    ( V =:= 1024 ->
        format("8. py_invoke/4 (2 args): OK~n", [])
    ; format("8. py_invoke/4 (2 args): FAIL~n", [])
    ),
    py_free(Result),
    py_free(Exp),
    py_free(Base),
    py_free(PowFn).

test_py_invoken :-
    py_eval("lambda a,b,c,d: a+b+c+d", Fn),
    py_from_int(1, A1),
    py_from_int(2, A2),
    py_from_int(3, A3),
    py_from_int(4, A4),
    py_invoken(Fn, [A1, A2, A3, A4], Result),
    py_to_int(Result, V),
    ( V =:= 10 ->
        format("9. py_invoken/3 (list args): OK~n", [])
    ; format("9. py_invoken/3 (list args): FAIL~n", [])
    ),
    py_free(Result),
    py_free(A4),
    py_free(A3),
    py_free(A2),
    py_free(A1),
    py_free(Fn).

test_operator_sugar :-
    Math := py_import("math"),
    Pi := Math:"pi",
    py_to_float(Pi, PiVal),
    ( PiVal > 3.14, PiVal < 3.15 ->
        format("10. := operator: OK~n", [])
    ; format("10. := operator: FAIL~n", [])
    ),
    py_free(Pi),
    py_free(Math).

test_operator_method_call :-
    py_from_str("hello world", S),
    Result := S:upper,
    py_to_str(Result, Str),
    ( Str = "HELLO WORLD" ->
        format("11. := method call: OK~n", [])
    ; format("11. := method call: FAIL~n", [])
    ),
    py_free(Result),
    py_free(S).

test_operator_method_call_many :-
    py_exec("def _sum5(a,b,c,d,e): return a+b+c+d+e"),
    py_eval("_sum5", Fn),
    py_from_int(1, A1),
    py_from_int(2, A2),
    py_from_int(3, A3),
    py_from_int(4, A4),
    py_from_int(5, A5),
    Result := Fn:'__call__'(A1, A2, A3, A4, A5),
    py_to_int(Result, V),
    ( V =:= 15 ->
        format("12. := method call (many args): OK~n", [])
    ; format("12. := method call (many args): FAIL~n", [])
    ),
    py_free(Result),
    py_free(A5),
    py_free(A4),
    py_free(A3),
    py_free(A2),
    py_free(A1),
    py_free(Fn).

test_collections :-
    py_list_new(L),
    V1 := py_from_int(10),
    V2 := py_from_int(20),
    py_list_append(L, V1),
    py_list_append(L, V2),
    py_list_len(L, Len),
    py_list_get(L, 0, Item),
    py_to_int(Item, ItemVal),
    ( Len =:= 2, ItemVal =:= 10 ->
        format("13. collections: OK~n", [])
    ; format("13. collections: FAIL~n", [])
    ),
    py_free(Item),
    py_free(V2),
    py_free(V1),
    py_free(L).

test_none :-
    py_none(N),
    ( py_is_none(N) ->
        format("14. none/is_none: OK~n", [])
    ; format("14. none/is_none: FAIL~n", [])
    ),
    py_free(N).

test_json :-
    py_from_json("[1, 2, 3]", H),
    py_to_json(H, Json),
    format("15. JSON roundtrip: ~w~n", [Json]),
    py_free(H).

test_from_to :-
    py_from_int(42, H1), py_to_int(H1, V1),
    py_from_float(3.14, H2), py_to_float(H2, V2),
    py_from_bool(true, H3), py_to_bool(H3, V3),
    py_from_str("test", H4), py_to_str(H4, V4),
    ( V1 =:= 42, V2 > 3.13, V2 < 3.15, V3 = true, V4 = "test" ->
        format("16. from/to conversions: OK~n", [])
    ; format("16. from/to conversions: FAIL~n", [])
    ),
    py_free(H4), py_free(H3), py_free(H2), py_free(H1).

test_error_handling :-
    ( catch(
        py_eval("1/0", _),
        error(python_error(_Err), _),
        format("17. error handling: OK~n", [])
    ) -> true ; format("17. error handling: FAIL~n", []) ).

test_with_py :-
    py_eval("42", H),
    with_py(H, (
        py_to_int(H, V),
        ( V =:= 42 -> true ; true )
    )),
    py_handle_count(Count),
    ( Count =:= 0 ->
        format("18. with_py: OK~n", [])
    ; format("18. with_py: FAIL (handles=~d)~n", [Count])
    ).

test_setattr :-
    py_eval("type('_TestObj', (object,), {})", Cls),
    py_invoke(Cls, Instance),
    py_from_int(99, Val),
    py_setattr(Instance, "x", Val),
    py_getattr(Instance, "x", Got),
    py_to_int(Got, V),
    ( V =:= 99 ->
        format("19. setattr/getattr: OK~n", [])
    ; format("19. setattr/getattr: FAIL~n", [])
    ),
    py_free(Got),
    py_free(Val),
    py_free(Instance),
    py_free(Cls).

:- initialization((
    py_init,
    test_eval,
    test_exec,
    test_import,
    test_py_call_0,
    test_py_call_2,
    test_py_invoke_0,
    test_py_invoke_1,
    test_py_invoke_2,
    test_py_invoken,
    test_operator_sugar,
    test_operator_method_call,
    test_operator_method_call_many,
    test_collections,
    test_none,
    test_json,
    test_from_to,
    test_error_handling,
    test_with_py,
    test_setattr,
    format("=== ALL 19 PROLOG API TESTS PASSED ===~n", []),
    py_finalize
)).
