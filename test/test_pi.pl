:- use_module(library(ffi)).

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
        'spy_finalize'([], void),
        'spy_import'([cstr], ptr),
        'spy_getattr'([ptr, cstr], ptr),
        'spy_to_float'([ptr], f64),
        'spy_to_str'([ptr], cstr),
        'spy_to_repr'([ptr], cstr),
        'spy_drop'([ptr], void),
        'spy_last_error'([], cstr)
    ]) -> true ; throw(error("Failed to load the ScryNeuro foreign module (check DLL or dynamic-library search paths)", init/0))).

test :-
    ffi:'spy_init'(_),
    write('importing math...'), nl,
    ffi:'spy_import'("math", HM),
    write('math handle: '), write(HM), nl,
    write('getting pi...'), nl,
    ffi:'spy_getattr'(HM, "pi", HPI),
    write('pi handle: '), write(HPI), nl,
    write('converting to float...'), nl,
    ffi:'spy_to_float'(HPI, VPI),
    write('pi = '), write(VPI), nl,
    ffi:'spy_drop'(HPI),
    ffi:'spy_drop'(HM),
    ffi:'spy_finalize',
    write('done'), nl.

:- initialization((init, test)).
