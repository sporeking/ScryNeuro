%% Windows end-to-end acceptance tests for the public Prolog API.
:- use_module('../prolog/scryer_py').
:- use_module(library(format)).

assert_equal(Expected, Actual, Label) :-
    ( Expected = Actual -> true
    ; throw(error(assertion_failed(Label, expected(Expected), actual(Actual)), test_windows/0))
    ).

assert_true(Goal, Label) :-
    ( call(Goal) -> true
    ; throw(error(assertion_failed(Label), test_windows/0))
    ).

allocate_handles(0, []) :- !.
allocate_handles(N, [Handle|Rest]) :-
    N > 0,
    py_from_int(N, Handle),
    Next is N - 1,
    allocate_handles(Next, Rest).

free_handles([]).
free_handles([Handle|Rest]) :-
    py_free(Handle),
    free_handles(Rest).

last_handle([Handle], Handle) :- !.
last_handle([_|Rest], Handle) :-
    last_handle(Rest, Handle).

test_large_values :-
    Large = 1099511627899,
    py_from_int(Large, Handle),
    py_to_int(Handle, Result),
    assert_equal(Large, Result, large_sint64_roundtrip),
    py_free(Handle),
    py_from_float(12345.75, FloatHandle),
    py_to_float(FloatHandle, FloatResult),
    assert_true((FloatResult > 12345.74, FloatResult < 12345.76), f64_roundtrip),
    py_free(FloatHandle).

test_unicode :-
    Text = "守门人 / 云港 / 交付凭证",
    py_from_str(Text, Handle),
    py_to_str(Handle, Result),
    assert_equal(Text, Result, utf8_roundtrip),
    py_free(Handle).

test_json :-
    Json = "{\"name\":\"守门人\",\"value\":1099511627899}",
    py_from_json(Json, Dict),
    py_dict_get(Dict, "name", NameHandle),
    py_to_str(NameHandle, Name),
    assert_equal("守门人", Name, json_utf8_value),
    py_dict_get(Dict, "value", ValueHandle),
    py_to_int(ValueHandle, Value),
    assert_equal(1099511627899, Value, json_sint64_value),
    py_free(ValueHandle),
    py_free(NameHandle),
    py_free(Dict).

test_many_handles :-
    allocate_handles(300, Handles),
    py_handle_count(Count),
    assert_equal(300, Count, handle_count_over_255),
    last_handle(Handles, Last),
    py_to_int(Last, LastValue),
    assert_equal(1, LastValue, ptr_handle_over_255),
    free_handles(Handles),
    py_handle_count(After),
    assert_equal(0, After, handle_cleanup).

test_invoke3 :-
    py_from_str("one two one", Text),
    py_from_str("one", Old),
    py_from_str("three", New),
    py_from_int(1, Count),
    py_call(Text, "replace", Old, New, Count, Result),
    py_to_str(Result, Value),
    assert_equal("three two one", Value, invoke3),
    py_free(Result),
    py_free(Count),
    py_free(New),
    py_free(Old),
    py_free(Text).

test_native_extension :-
    py_import("_sqlite3", Sqlite),
    py_getattr(Sqlite, "sqlite_version", VersionHandle),
    py_to_str(VersionHandle, Version),
    assert_true(Version = [_|_], native_pyd_import),
    py_free(VersionHandle),
    py_free(Sqlite).

test_error_boundary :-
    catch(py_eval("1/0", _), Error, true),
    assert_true(nonvar(Error), python_exception_is_reported),
    Error = error(python_error(Message), _),
    assert_true((Message = [_|_]), python_exception_has_message).

stress_eval(0) :- !.
stress_eval(N) :-
    N > 0,
    py_eval("6 * 7", Handle),
    py_to_int(Handle, Value),
    assert_equal(42, Value, repeated_eval),
    py_free(Handle),
    Next is N - 1,
    stress_eval(Next).

run_once :-
    py_init,
    py_eval("1 + 2", Basic),
    py_to_int(Basic, BasicResult),
    assert_equal(3, BasicResult, basic_eval),
    py_free(Basic),
    test_large_values,
    test_unicode,
    test_json,
    test_many_handles,
    test_invoke3,
    test_native_extension,
    test_error_boundary,
    stress_eval(1000),
    py_handle_count(FinalCount),
    assert_equal(0, FinalCount, final_handle_count),
    py_finalize.

run_tests :-
    run_once,
    %% Leave one object behind so finalize must clear the registry.
    py_init,
    py_eval("40 + 2", Handle),
    py_to_int(Handle, Value),
    assert_equal(42, Value, reinitialize),
    py_finalize,
    py_init,
    py_handle_count(AfterFinalize),
    assert_equal(0, AfterFinalize, finalize_clears_handles),
    py_finalize,
    format("WINDOWS TESTS PASSED~n", []).

main :-
    catch(
        run_tests,
        Error,
        ( format("WINDOWS TEST FAILURE: ~q~n", [Error]), halt(1) )
    ),
    halt.

:- initialization(main).
