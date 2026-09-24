%% Minimal smoke test for ScryNeuro
%% use_foreign_module/2 is a runtime goal, not a directive.
:- use_module(library(ffi)).
:- use_module('../prolog/scryer_py').

%% Detect library path: try .dylib (macOS) by checking file existence,
%% fall back to .so (Linux).
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
        'spy_eval'([cstr], ptr),
        'spy_to_int'([ptr], sint64),
        'spy_to_str'([ptr], cstr),
        'spy_drop'([ptr], void),
        'spy_last_error'([], cstr),
        'spy_finalize'([], void),
        'spy_handle_count'([], sint64)
    ]) -> true ; throw(error("Failed to load foreign module (check DYLD_LIBRARY_PATH on macOS or LD_LIBRARY_PATH on Linux)", init/0))).

test :-
    %% 1. Initialize Python
    ffi:'spy_init'(Status),
    ( Status =:= 0 ->
        write('spy_init OK'), nl
    ; ffi:'spy_last_error'(Err),
      print_py_error(error(python_error(Err), spy_init/0)),
      halt(1)
    ),

    %% 2. Evaluate "1 + 2"
    ffi:'spy_eval'("1 + 2", H),
    ( H =\= 0 ->
        ffi:'spy_to_int'(H, V),
        ( V =:= 3 -> true
        ; throw(error(test_failed(spy_to_int), test_smoke/0))
        ),
        write('1 + 2 = '), write(V), nl,
        ffi:'spy_drop'(H)
    ; ffi:'spy_last_error'(Err2),
      throw(error(python_error(Err2), spy_eval/2))
    ),

    %% 3. Evaluate a string
    ffi:'spy_eval'("'hello world'", H2),
    ( H2 =\= 0 ->
        ffi:'spy_to_str'(H2, S),
        ( S = "hello world" -> true
        ; throw(error(test_failed(spy_to_str), test_smoke/0))
        ),
        write('String: '), write(S), nl,
        ffi:'spy_drop'(H2)
    ; ffi:'spy_last_error'(Err3),
      throw(error(python_error(Err3), spy_eval/2))
    ),

    %% 4. Check handle count
    ffi:'spy_handle_count'(Count),
    ( Count =:= 0 -> true
    ; throw(error(test_failed(handle_count(Count)), test_smoke/0))
    ),
    write('Live handles: '), write(Count), nl,

    %% 5. Finalize
    ffi:'spy_finalize',
    write('All tests passed!'), nl.

run_tests :-
    ( catch((init, test), Error,
            ( write('Smoke tests failed: '), writeq(Error), nl, halt(1) )) ->
        halt(0)
    ; write('Smoke tests failed without an exception.'), nl, halt(1)
    ).

:- initialization(run_tests).
