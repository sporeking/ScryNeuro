%% Minimal smoke test for ScryNeuro
%% use_foreign_module/2 is a runtime goal, not a directive.
:- use_module(library(ffi)).
:- use_module('../prolog/scryer_py').

%% Detect library path for Windows, macOS, or Linux.
lib_path(Path) :-
    ( catch((open('./libscryneuro.dylib', read, S), close(S)), _, fail) ->
        Path = "./libscryneuro.dylib"
    ; catch((open('./libscryneuro.so', read, S), close(S)), _, fail) ->
        Path = "./libscryneuro.so"
    ; catch((open('./scryneuro.dll', read, S), close(S)), _, fail) ->
        Path = "./scryneuro.dll"
    ; catch((open('../libscryneuro.dylib', read, S), close(S)), _, fail) ->
        Path = "../libscryneuro.dylib"
    ; catch((open('../libscryneuro.so', read, S), close(S)), _, fail) ->
        Path = "../libscryneuro.so"
    ; catch((open('../scryneuro.dll', read, S), close(S)), _, fail) ->
        Path = "../scryneuro.dll"
    ; throw(error("Could not find a ScryNeuro dynamic library", lib_path/1))
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
    ]) -> true ; throw(error("Failed to load the ScryNeuro foreign module (check DLL or dynamic-library search paths)", init/0))).

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
        write('1 + 2 = '), write(V), nl,
        ffi:'spy_drop'(H)
    ; ffi:'spy_last_error'(Err2),
      print_py_error(error(python_error(Err2), spy_eval/2))
    ),

    %% 3. Evaluate a string
    ffi:'spy_eval'("'hello world'", H2),
    ( H2 =\= 0 ->
        ffi:'spy_to_str'(H2, S),
        write('String: '), write(S), nl,
        ffi:'spy_drop'(H2)
    ; ffi:'spy_last_error'(Err3),
      print_py_error(error(python_error(Err3), spy_eval/2))
    ),

    %% 4. Check handle count
    ffi:'spy_handle_count'(Count),
    write('Live handles: '), write(Count), nl,

    %% 5. Finalize
    ffi:'spy_finalize',
    write('All tests passed!'), nl.

:- initialization((init, test)).
