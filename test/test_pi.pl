:- use_module(library(ffi)).
:- use_module(library(iso_ext)).

lib_path(Path) :-
    ( catch((open('./libscryneuro.dylib', read, S), close(S)), _, fail) ->
        Path = "./libscryneuro.dylib"
    ; catch((open('./libscryneuro.so', read, S), close(S)), _, fail) ->
        Path = "./libscryneuro.so"
    ; catch((open('../libscryneuro.dylib', read, S), close(S)), _, fail) ->
        Path = "../libscryneuro.dylib"
    ; catch((open('../libscryneuro.so', read, S), close(S)), _, fail) ->
        Path = "../libscryneuro.so"
    ; throw(error("Could not find libscryneuro.dylib or libscryneuro.so", lib_path/1))
    ).

init :-
    lib_path(LibPath),
    ( use_foreign_module(LibPath, [
        'spy_init'([], sint32),
        'spy_finalize'([], void),
        'spy_import'([cstr], ptr),
        'spy_getattr'([ptr, cstr], ptr),
        'spy_to_float'([ptr], f64),
        'spy_to_str'([ptr], cstr),
        'spy_to_repr'([ptr], cstr),
        'spy_drop'([ptr], void),
        'spy_last_error'([], cstr)
    ]) -> true ; throw(error("Failed to load foreign module (check DYLD_LIBRARY_PATH on macOS or LD_LIBRARY_PATH on Linux)", init/0))).

test :-
    ffi:'spy_init'(Status),
    ( Status =:= 0 -> true
    ; throw(error(test_failed(spy_init), test_pi/0))
    ),
    setup_call_cleanup(true, test_pi, ffi:'spy_finalize').

test_pi :-
    write('importing math...'), nl,
    ffi:'spy_import'("math", HM),
    ( HM =\= 0 -> true
    ; throw(error(test_failed(py_import), test_pi/0))
    ),
    write('math handle: '), write(HM), nl,
    write('getting pi...'), nl,
    ffi:'spy_getattr'(HM, "pi", HPI),
    ( HPI =\= 0 -> true
    ; throw(error(test_failed(py_getattr), test_pi/0))
    ),
    write('pi handle: '), write(HPI), nl,
    write('converting to float...'), nl,
    ffi:'spy_to_float'(HPI, VPI),
    write('pi = '), write(VPI), nl,
    ( VPI > 3.14, VPI < 3.15 -> true
    ; throw(error(test_failed(py_to_float), test_pi/0))
    ),
    ffi:'spy_drop'(HPI),
    ffi:'spy_drop'(HM),
    write('pi test passed'), nl.

run_tests :-
    ( catch((init, test), Error,
            ( write('Pi test failed: '), writeq(Error), nl, halt(1) )) ->
        halt(0)
    ; write('Pi test failed without an exception.'), nl, halt(1)
    ).

:- initialization(run_tests).
