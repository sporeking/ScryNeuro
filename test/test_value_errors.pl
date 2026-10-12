%% Regression: FFI errors must be checked before output unification.
:- use_module('../prolog/scryer_py').
:- use_module(library(iso_ext)).
:- use_module(library(format)).

expect(Goal) :-
    ( call(Goal) -> true ; throw(error(assertion_failed(Goal), values/0)) ).

expect_python_error(Goal, Context) :-
    catch((call(Goal), Outcome = success), Error, Outcome = exception(Error)),
    expect(Outcome = exception(error(python_error([_|_]), Context))).

invalid_outputs :-
    expect_python_error(py_to_int(0, 42), py_to_int/2),
    expect_python_error(py_to_float(0, 3.14), py_to_float/2),
    expect_python_error(py_to_str(0, "hello"), py_to_str/2),
    expect_python_error(py_to_repr(0, "hello"), py_to_repr/2),
    expect_python_error(py_to_json(0, "{}"), py_to_json/2),
    expect_python_error(py_list_len(0, 10), py_list_len/2),
    expect_python_error(py_to_bool(0, true), py_to_bool/2),
    expect_python_error(py_to_bool(0, false), py_to_bool/2),
    expect_python_error(py_is_none(0), py_is_none/1).

valid_outputs :-
    once(with_py_many([
        I-py_from_int(3, I), F-py_from_float(3.14, F),
        S-py_from_str("hello", S), B-py_from_bool(true, B),
        False-py_from_bool(false, False), N-py_none(N),
        L-py_from_json("[1,2]", L), D-py_from_json("{}", D)
    ], (
        expect(py_to_int(I, 3)), expect(\+ py_to_int(I, 42)),
        expect(py_to_float(F, 3.14)), expect(\+ py_to_float(F, 2.0)),
        expect(py_to_str(S, "hello")), expect(\+ py_to_str(S, "other")),
        expect(py_to_repr(S, "'hello'")), expect(\+ py_to_repr(S, "other")),
        expect(py_to_json(D, "{}")), expect(\+ py_to_json(D, "[]")),
        expect(py_list_len(L, 2)), expect(\+ py_list_len(L, 10)),
        expect(py_to_bool(B, true)), expect(\+ py_to_bool(B, false)),
        expect(py_to_bool(False, false)), expect(\+ py_to_bool(False, true)),
        expect(py_is_none(N)), expect(\+ py_is_none(I)),
        expect_python_error(py_to_int(S, 42), py_to_int/2),
        expect_python_error(py_to_float(S, 3.14), py_to_float/2),
        expect_python_error(py_list_len(I, 10), py_list_len/2),
        expect(py_to_int(I, 3)), expect(py_last_error([])),
        py_handle_count(Count), expect(py_handle_count(Count)),
        Wrong is Count + 1, expect(\+ py_handle_count(Wrong))
    ))).

repeated(0) :- !.
repeated(N) :-
    expect(invalid_outputs), expect(valid_outputs), expect(py_handle_count(0)),
    Next is N - 1, repeated(Next).

values :-
    expect(py_handle_count(0)), repeated(10),
    py_finalize,
    expect_python_error(py_handle_count(42), py_handle_count/1),
    py_init, expect(py_handle_count(0)).

run :-
    ( catch(setup_call_cleanup(py_init, values, py_finalize), E,
            (format("Value regressions failed: ~q~n", [E]), halt(1))) ->
        format("Value regressions passed.~n", []), halt(0)
    ; halt(1) ).
:- initialization(run).
