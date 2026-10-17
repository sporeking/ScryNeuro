%% Run after input guards are installed; all dangerous cases stay in this process.
:- use_module('../prolog/scryer_py').
:- use_module(library(iso_ext)).
:- use_module(library(format)).
:- use_module(library(os)).

expect(Goal) :-
    ( call(Goal) -> true ; throw(error(assertion_failed(Goal), strings/0)) ).

expect_error(Goal, Pattern) :-
    catch((call(Goal), Outcome = success), Error, Outcome = exception(Error)),
    expect(Outcome = exception(Pattern)).

string_goals(S, [
    py_eval(S, _), py_exec(S), py_exec_lines([S]), py_import(S, _),
    py_getattr(0, S, _), py_setattr(0, S, 0),
    py_call(0, S, _), py_call(0, S, 0, _),
    py_call(0, S, 0, 0, _), py_call(0, S, 0, 0, 0, _),
    py_calln(0, S, [], _), py_calln(0, S, 0, _),
    py_from_str(S, _), py_from_json(S, _),
    py_dict_get(0, S, _), py_dict_set(0, S, 0),
    py_init(S), py_init_home(S)
]).

check_goals([], _).
check_goals([Goal|Rest], Pattern) :-
    copy_term(Pattern, FreshPattern),
    expect_error(Goal, FreshPattern), expect(py_handle_count(0)),
    check_goals(Rest, Pattern).

check_input(Input, Pattern) :-
    string_goals(Input, Goals), check_goals(Goals, Pattern).

invalid_inputs :-
    check_input(42, error(type_error(list, 42), _)),
    check_input(atom, error(type_error(list, atom), _)),
    check_input(_, error(instantiation_error, _)),
    check_input([a|_], error(instantiation_error, _)),
    check_input([a|b], error(type_error(list, _), _)),
    check_input([a, 42], error(type_error(character, 42), _)),
    check_input([a, word], error(type_error(character, word), _)),
    check_input([a, _], error(instantiation_error, _)),
    char_code(Nul, 0),
    check_input([a, Nul, b], error(domain_error(nul_free_string, _), _)),
    expect_error(py_exec_lines(42), error(type_error(list, 42), _)),
    expect_error(py_exec_lines(_), error(instantiation_error, _)),
    expect_error(py_exec_lines(["pass"|_]), error(instantiation_error, _)),
    expect_error(py_exec_lines(["_guard_should_not_run = 1", 42]),
                 error(type_error(list, 42), _)),
    once(with_py_temp(py_eval("'_guard_should_not_run' in globals()", H), H,
                      expect(py_to_bool(H, false)))).

valid_inputs :-
    once(with_py_many([
        Empty-py_from_str("", Empty), Unicode-py_from_str("你好 λ 🧠", Unicode),
        D-py_dict_new(D), V-py_from_int(7, V)
    ], (
        expect(py_to_str(Empty, "")), expect(py_to_str(Unicode, "你好 λ 🧠")),
        py_dict_set(D, "键λ", V),
        once(with_py_temp(py_dict_get(D, "键λ", Got), Got,
                          expect(py_to_int(Got, 7)))),
        py_dict_set(D, "", V),
        once(with_py_temp(py_dict_get(D, "", GotEmpty), GotEmpty,
                          expect(py_to_int(GotEmpty, 7))))
    ))),
    py_exec(""), py_exec_lines([]), py_exec_lines(["", "pass", ""]),
    once(with_py_temp(py_eval("'你好 λ 🧠'", Eval), Eval,
                      expect(py_to_str(Eval, "你好 λ 🧠")))),
    once(with_py_temp(py_from_json("\"你好 λ 🧠\"", Json), Json,
                      expect(py_to_str(Json, "你好 λ 🧠")))).

strings :-
    expect(py_handle_count(0)), invalid_inputs,
    once(with_py_temp(py_from_float(16.0, V16), V16, valid_inputs_with_arg(V16))),
    expect(py_handle_count(0)),
    py_finalize,
    expect_error(py_init(42), error(type_error(list, 42), _)),
    expect_error(py_init_home(_), error(instantiation_error, _)),
    char_code(Nul, 0),
    expect_error(py_init([Nul]), error(domain_error(nul_free_string, _), _)),
    getenv("SCRYNEURO_HOME", Home), py_init_home(Home),
    expect(py_handle_count(0)).

valid_inputs_with_arg(V16) :-
    valid_inputs,
    expect(py_handle_count(1)),
    once(with_py_temp(py_import("math", Math), Math,
        once(with_py_temp(py_call(Math, "sqrt", V16, Result), Result,
                          expect(py_to_float(Result, 4.0)))))).

run :-
    ( catch(setup_call_cleanup(py_init, strings, py_finalize), E,
            (format("String regressions failed: ", []), print_py_error(E), halt(1))) ->
        format("String regressions passed.~n", []), halt(0)
    ; halt(1) ).
:- initialization(run).
