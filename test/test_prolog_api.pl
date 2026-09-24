%% ===========================================================================
%% ScryNeuro Prolog API Test Suite
%% ===========================================================================
%% Tests the scryer_py.pl module predicates
%% Run: scryer-prolog test/test_prolog_api.pl

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module(library(format)).
:- use_module(library(lists)).
:- use_module(library(iso_ext)).
:- use_module(library(clpz)).

fail_test(Label) :-
    throw(error(test_failed(Label), test_prolog_api/0)).

report_handle_cleanup(Label, Before) :-
    py_handle_count(After),
    ( After =:= Before ->
        format("~w: OK~n", [Label])
    ; throw(error(handle_leak(Label, Before, After), test_prolog_api/0))
    ).

expect_bound_handle_error(Goal, Context, Before) :-
    catch(
        ( call(Goal), fail_test('bound handle output accepted') ),
        Error,
        ( Error = error(uninstantiation_error(-1), Context) -> true
        ; throw(Error)
        )
    ),
    py_handle_count(After),
    ( After =:= Before -> true
    ; throw(error(handle_leak(Context, Before, After), test_prolog_api/0))
    ).

local_acquire_int(Value, Handle) :-
    py_from_int(Value, Handle).

local_goal_int_eq(Handle, Expected) :-
    py_to_int(Handle, Value),
    Value =:= Expected.

local_goal_sum_eq(H1, H2, Expected) :-
    py_to_int(H1, V1),
    py_to_int(H2, V2),
    V1 + V2 =:= Expected.

test_eval :-
    py_eval("2 ** 10", H),
    py_to_int(H, V),
    ( V =:= 1024 ->
        format("1. py_eval: OK~n", [])
    ; fail_test('1. py_eval')
    ),
    py_free(H).

test_exec :-
    py_exec("_test_var = 42"),
    py_eval("_test_var", H),
    py_to_int(H, V),
    ( V =:= 42 ->
        format("2. py_exec: OK~n", [])
    ; fail_test('2. py_exec')
    ),
    py_free(H).

test_import :-
    py_import("math", M),
    py_getattr(M, "pi", Pi),
    py_to_float(Pi, PiVal),
    ( PiVal > 3.14, PiVal < 3.15 ->
        format("3. py_import/getattr: OK~n", [])
    ; fail_test('3. py_import/getattr')
    ),
    py_free(Pi),
    py_free(M).

test_py_call_0 :-
    py_from_str("hello world", S),
    py_call(S, "upper", Result),
    py_to_str(Result, Str),
    ( Str = "HELLO WORLD" ->
        format("4. py_call/3 (0 args): OK~n", [])
    ; fail_test('4. py_call/3 (0 args)')
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
    ; fail_test('5. py_call/5 (2 args)')
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
    ; fail_test('6. py_invoke/2 (0 args)')
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
    ; fail_test('7. py_invoke/3 (1 arg)')
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
    ; fail_test('8. py_invoke/4 (2 args)')
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
    ; fail_test('9. py_invoken/3 (list args)')
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
    ; fail_test('10. := operator')
    ),
    py_free(Pi),
    py_free(Math).

test_operator_method_call :-
    py_from_str("hello world", S),
    Result := S:upper,
    py_to_str(Result, Str),
    ( Str = "HELLO WORLD" ->
        format("11. := method call: OK~n", [])
    ; fail_test('11. := method call')
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
    ; fail_test('12. := method call (many args)')
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
    ; fail_test('13. collections')
    ),
    py_free(Item),
    py_free(V2),
    py_free(V1),
    py_free(L).

test_none :-
    py_none(N),
    ( py_is_none(N) ->
        format("14. none/is_none: OK~n", [])
    ; fail_test('14. none/is_none')
    ),
    py_free(N).

test_json :-
    py_from_json("[1, 2, 3]", H),
    py_to_json(H, Json),
    ( Json = "[1, 2, 3]" ->
        format("15. JSON roundtrip: OK~n", [])
    ; fail_test('15. JSON roundtrip')
    ),
    py_free(H).

test_from_to :-
    py_from_int(42, H1), py_to_int(H1, V1),
    py_from_float(3.14, H2), py_to_float(H2, V2),
    py_from_bool(true, H3), py_to_bool(H3, V3),
    py_from_str("test", H4), py_to_str(H4, V4),
    ( V1 =:= 42, V2 > 3.13, V2 < 3.15, V3 = true, V4 = "test" ->
        format("16. from/to conversions: OK~n", [])
    ; fail_test('16. from/to conversions')
    ),
    py_free(H4), py_free(H3), py_free(H2), py_free(H1).

test_error_handling :-
    ( catch(
        ( py_eval("1/0", _), fail_test('17. error handling (no exception)') ),
        error(python_error(_), _),
        true
    ) -> true ; fail_test('17. error handling') ),
    py_handle_count(Before),
    once(with_py_temp(py_eval("lambda *args: 1 / 0", Fn), Fn, (
        catch(
            ( py_invoken(Fn, [], _), fail_test('call error missing') ),
            error(python_error(Msg), py_invoken/3),
            ( Msg = [_|_] -> true ; fail_test('call error message lost') )
        )
    ))),
    report_handle_cleanup('call error cleanup', Before),
    format("17. error handling: OK~n", []).

test_nul_string :-
    py_eval("'A' + chr(0) + 'B'", H),
    with_py(H, (
        catch(
            ( py_to_str(H, _), fail_test('NUL string unexpectedly succeeded') ),
            error(python_error(Msg), py_to_str/2),
            ( Msg = "FFI string contains a NUL byte" -> true
            ; fail_test('NUL string lost error') )
        ),
        py_to_json(H, Json),
        ( Json = "\"A\\u0000B\"" ->
            format("NUL string and JSON: OK~n", [])
        ; fail_test('NUL JSON roundtrip')
        )
    )).

test_nul_error :-
    catch(
        ( py_exec("raise ValueError('A' + chr(0) + 'B')"),
          fail_test('NUL exception unexpectedly succeeded') ),
        error(python_error(Msg), py_exec/1),
        ( char_code(Backslash, 92),
          append(_, ['A', Backslash, '0', 'B' | _], Msg) ->
            format("NUL exception message: OK~n", [])
        ; fail_test('NUL exception lost error')
        )
    ).

test_with_py :-
    py_handle_count(Before),
    py_eval("42", H),
    with_py(H, (
        py_to_int(H, V),
        V =:= 42
    )),
    report_handle_cleanup('18. with_py (success)', Before).

test_with_py_failure :-
    py_handle_count(Before),
    py_eval("42", H),
    ( with_py(H, fail) ->
        fail_test('19. with_py (failure)')
    ; report_handle_cleanup('19. with_py (failure)', Before)
    ).

test_with_py_exception :-
    py_handle_count(Before),
    py_eval("42", H),
    catch(
        ( with_py(H, throw(test_with_py_exception)),
          fail_test('20. with_py (exception not propagated)') ),
        test_with_py_exception,
        true
    ),
    report_handle_cleanup('20. with_py (exception)', Before).

test_with_py_temp :-
    py_handle_count(Before),
    with_py_temp(py_eval("21 * 2", H), H, (
        py_to_int(H, V),
        V =:= 42
    )),
    report_handle_cleanup('21. with_py_temp (success)', Before).

test_with_py_temp_exception :-
    py_handle_count(Before),
    catch(
        ( with_py_temp(py_eval("42", H), H, throw(test_with_py_temp_exception)),
          fail_test('22. with_py_temp (exception not propagated)') ),
        test_with_py_temp_exception,
        true
    ),
    report_handle_cleanup('22. with_py_temp (exception)', Before).

test_with_py_temp_acquire_failure :-
    py_handle_count(Before),
    ( with_py_temp(fail, _H, true) ->
        fail_test('23. with_py_temp (acquire failure)')
    ; report_handle_cleanup('23. with_py_temp (acquire failure)', Before)
    ).

test_with_py_many :-
    py_handle_count(Before),
    with_py_many([
        H1-py_from_int(1, H1),
        H2-py_from_int(2, H2)
    ], (
        py_list_from_handles([H1, H2], List),
        with_py(List, (
            py_list_len(List, Len),
            Len =:= 2
        ))
    )),
    report_handle_cleanup('24. with_py_many (success)', Before).

test_with_py_many_failure :-
    py_handle_count(Before),
    ( with_py_many([
          H1-py_from_int(1, H1),
          H2-py_from_int(2, H2)
      ], fail) ->
        fail_test('25. with_py_many (failure)')
    ; report_handle_cleanup('25. with_py_many (failure)', Before)
    ).

test_with_py_many_exception :-
    py_handle_count(Before),
    catch(
        ( with_py_many([
              H1-py_from_int(1, H1),
              H2-py_from_int(2, H2)
          ], throw(test_with_py_many_exception)),
          fail_test('26. with_py_many (exception not propagated)') ),
        test_with_py_many_exception,
        true
    ),
    report_handle_cleanup('26. with_py_many (exception)', Before).

test_with_py_many_partial_acquire :-
    py_handle_count(Before),
    catch(
        ( with_py_many([
              H1-py_from_int(1, H1),
              H2-py_eval("1/0", H2)
          ], true),
          fail_test('27. with_py_many (acquire did not throw)') ),
        error(python_error(_), _),
        true
    ),
    report_handle_cleanup('27. with_py_many (acquire error)', Before).

test_with_py_local_goal_context :-
    py_handle_count(Before),
    py_eval("42", H),
    with_py(H, local_goal_int_eq(H, 42)),
    report_handle_cleanup('28. with_py (caller-local goal)', Before).

test_with_py_temp_local_context :-
    py_handle_count(Before),
    with_py_temp(local_acquire_int(42, H), H, local_goal_int_eq(H, 42)),
    report_handle_cleanup('29. with_py_temp (caller-local acquire/goal)', Before).

test_with_py_many_local_goal_context :-
    py_handle_count(Before),
    with_py_many([
        H1-py_from_int(1, H1),
        H2-py_from_int(2, H2)
    ], local_goal_sum_eq(H1, H2, 3)),
    report_handle_cleanup('30. with_py_many (caller-local goal)', Before).

test_with_py_many_explicit_qualified_acquire :-
    py_handle_count(Before),
    with_py_many([
        H1-(user:local_acquire_int(1, H1)),
        H2-(user:local_acquire_int(2, H2))
    ], local_goal_sum_eq(H1, H2, 3)),
    report_handle_cleanup('31. with_py_many (qualified acquire)', Before).

test_setattr :-
    py_eval("type('_TestObj', (object,), {})", Cls),
    py_invoke(Cls, Instance),
    py_from_int(99, Val),
    py_setattr(Instance, "x", Val),
    py_getattr(Instance, "x", Got),
    py_to_int(Got, V),
    ( V =:= 99 ->
        format("32. setattr/getattr: OK~n", [])
    ; fail_test('32. setattr/getattr')
    ),
    py_free(Got),
    py_free(Val),
    py_free(Instance),
    py_free(Cls).

test_missing_library :-
    catch(
        ( py_init(""),
          fail_test('empty library path unexpectedly loaded') ),
        error(existence_error(source_sink, _), _),
        format("empty library path: OK~n", [])
    ).

test_atom_library_path :-
    Path = './libscryneuro.dylib',
    catch(
        ( py_init(Path),
          fail_test('atom library path accepted') ),
        error(type_error(list, Path), _),
        format("atom library path: OK~n", [])
    ).

test_list_build_failures :-
    py_handle_count(Before),
    catch(
        ( py_list_from_handles(42, _), fail_test('invalid list input accepted') ),
        error(type_error(list, 42), _),
        true
    ),
    catch(
        ( py_list_from_handles(_, _), fail_test('unbound list input accepted') ),
        error(instantiation_error, _),
        true
    ),
    report_handle_cleanup('invalid list input', Before),
    with_py_temp(py_from_int(7, H), H, (
        py_handle_count(During),
        catch(
            ( py_list_from_handles([H, 0], _),
              fail_test('invalid list element accepted') ),
            error(python_error(Msg), py_list_append/2),
            ( Msg = [_|_] -> true ; fail_test('list append lost error') )
        ),
        report_handle_cleanup('list append failure', During)
    )),
    expect_bound_handle_error(py_list_from_handles([], -1),
                              py_list_from_handles/2, Before),
    report_handle_cleanup('pre-bound list output', Before).

test_bound_handle_outputs :-
    py_handle_count(Before),
    expect_bound_handle_error(py_eval("42", -1), py_eval/2, Before),
    expect_bound_handle_error(py_eval("1/0", -1), py_eval/2, Before),
    expect_bound_handle_error(py_import("math", -1), py_import/2, Before),
    expect_bound_handle_error(py_from_int(42, -1), py_from_int/2, Before),
    expect_bound_handle_error(py_from_float(1.5, -1), py_from_float/2, Before),
    expect_bound_handle_error(py_from_bool(true, -1), py_from_bool/2, Before),
    expect_bound_handle_error(py_from_bool(false, -1), py_from_bool/2, Before),
    expect_bound_handle_error(py_from_str("value", -1), py_from_str/2, Before),
    expect_bound_handle_error(py_none(-1), py_none/1, Before),
    expect_bound_handle_error(py_from_json("[]", -1), py_from_json/2, Before),
    expect_bound_handle_error(py_list_new(-1), py_list_new/1, Before),
    expect_bound_handle_error(py_dict_new(-1), py_dict_new/1, Before),
    once(with_py_many([
        Obj-py_eval("type('T', (), {'f': lambda self, *args: 42})()", Obj),
        Arg-py_from_int(1, Arg),
        Callable-py_eval("lambda *args: 42", Callable),
        List-py_from_json("[1]", List),
        Dict-py_from_json("{\"x\": 1}", Dict)
    ], (
        py_handle_count(During),
        expect_bound_handle_error(py_getattr(Obj, "f", -1), py_getattr/3, During),
        expect_bound_handle_error(py_call(Obj, "f", -1), py_call/3, During),
        expect_bound_handle_error(py_call(Obj, "f", Arg, -1), py_call/4, During),
        expect_bound_handle_error(py_call(Obj, "f", Arg, Arg, -1), py_call/5, During),
        expect_bound_handle_error(py_call(Obj, "f", Arg, Arg, Arg, -1), py_call/6, During),
        expect_bound_handle_error(py_calln(Obj, "f", [], -1), py_calln/4, During),
        expect_bound_handle_error(py_calln(Obj, "f", [Arg], -1), py_calln/4, During),
        expect_bound_handle_error(py_calln(Obj, "f", List, -1), py_calln/4, During),
        expect_bound_handle_error(py_invoke(Callable, -1), py_invoke/2, During),
        expect_bound_handle_error(py_invoke(Callable, Arg, -1), py_invoke/3, During),
        expect_bound_handle_error(py_invoke(Callable, Arg, Arg, -1), py_invoke/4, During),
        expect_bound_handle_error(py_invoken(Callable, [], -1), py_invoken/3, During),
        expect_bound_handle_error(py_invoken(Callable, [Arg], -1), py_invoken/3, During),
        expect_bound_handle_error(py_invoken(Callable, List, -1), py_invoken/3, During),
        expect_bound_handle_error(py_list_get(List, 0, -1), py_list_get/3, During),
        expect_bound_handle_error(py_dict_get(Dict, "x", -1), py_dict_get/3, During)
    ))),
    report_handle_cleanup('bound handle outputs', Before).

test_constrained_handle_outputs :-
    py_handle_count(Before),
    EvalHandle #< 0,
    ( py_eval("42", EvalHandle) ->
        fail_test('constrained eval handle accepted')
    ; report_handle_cleanup('constrained eval handle', Before)
    ),
    once(with_py_temp(py_from_int(7, Item), Item, (
        py_handle_count(During),
        ListHandle #< 0,
        ( py_list_from_handles([Item], ListHandle) ->
            fail_test('constrained list handle accepted')
        ; report_handle_cleanup('constrained list handle', During)
        )
    ))),
    AllowedHandle #> 0,
    py_eval("42", AllowedHandle),
    py_to_int(AllowedHandle, 42),
    py_free(AllowedHandle),
    report_handle_cleanup('constrained handle outputs', Before).

test_stale_handles :-
    py_handle_count(Before),
    py_from_int(7, Old),
    py_finalize,
    py_init,
    py_from_int(99, New),
    ( Old =\= New -> true ; fail_test('handle reused after reinitialization') ),
    catch(
        ( py_to_int(Old, _), fail_test('stale handle accepted') ),
        error(python_error(Msg), py_to_int/2),
        ( Msg = [_|_] -> true ; fail_test('stale handle lost error') )
    ),
    py_to_int(New, Value),
    ( Value =:= 99 -> true ; fail_test('new handle points to wrong object') ),
    py_free(New),
    report_handle_cleanup('stale handle after reinitialization', Before).

run_tests :-
    ( catch(
        ( test_missing_library,
          test_atom_library_path,
          setup_call_cleanup(py_init, all_tests, py_finalize) ),
        Error,
        ( format("Prolog API tests failed: ~q~n", [Error]), halt(1) )
      ) -> halt(0)
    ; format("Prolog API tests failed without an exception.~n", []), halt(1)
    ).

all_tests :-
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
    test_list_build_failures,
    test_bound_handle_outputs,
    test_constrained_handle_outputs,
    test_none,
    test_json,
    test_from_to,
    test_error_handling,
    test_nul_string,
    test_nul_error,
    test_with_py,
    test_with_py_failure,
    test_with_py_exception,
    test_with_py_temp,
    test_with_py_temp_exception,
    test_with_py_temp_acquire_failure,
    test_with_py_many,
    test_with_py_many_failure,
    test_with_py_many_exception,
    test_with_py_many_partial_acquire,
    test_with_py_local_goal_context,
    test_with_py_temp_local_context,
    test_with_py_many_local_goal_context,
    test_with_py_many_explicit_qualified_acquire,
    test_setattr,
    test_stale_handles,
    format("=== ALL 38 PROLOG API TESTS PASSED ===~n", []).

:- initialization(run_tests).
