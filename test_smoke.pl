%% Minimal smoke test for ScryNeuro
%% use_foreign_module/2 is a runtime goal, not a directive.
:- use_module(library(ffi)).

init :-
    use_foreign_module("./libscryneuro.so", [
        'spy_init'([], sint32),
        'spy_eval'([cstr], ptr),
        'spy_to_int'([ptr], sint64),
        'spy_to_str'([ptr], cstr),
        'spy_drop'([ptr], void),
        'spy_last_error'([], cstr),
        'spy_finalize'([], void),
        'spy_handle_count'([], sint64)
    ]).

test :-
    %% 1. Initialize Python
    ffi:'spy_init'(Status),
    ( Status =:= 0 ->
        write('spy_init OK'), nl
    ; ffi:'spy_last_error'(Err),
      write('spy_init FAILED: '), write(Err), nl,
      halt(1)
    ),

    %% 2. Evaluate "1 + 2"
    ffi:'spy_eval'("1 + 2", H),
    ( H =\= 0 ->
        ffi:'spy_to_int'(H, V),
        write('1 + 2 = '), write(V), nl,
        ffi:'spy_drop'(H)
    ; ffi:'spy_last_error'(Err2),
      write('spy_eval FAILED: '), write(Err2), nl
    ),

    %% 3. Evaluate a string
    ffi:'spy_eval'("'hello world'", H2),
    ( H2 =\= 0 ->
        ffi:'spy_to_str'(H2, S),
        write('String: '), write(S), nl,
        ffi:'spy_drop'(H2)
    ; ffi:'spy_last_error'(Err3),
      write('spy_eval str FAILED: '), write(Err3), nl
    ),

    %% 4. Check handle count
    ffi:'spy_handle_count'(Count),
    write('Live handles: '), write(Count), nl,

    %% 5. Finalize
    ffi:'spy_finalize',
    write('All tests passed!'), nl.

:- initialization((init, test)).
