%% Preflight for the Scryer Prolog Windows FFI implementation.
%% This test does not require ScryNeuro or Python.
:- use_module(library(ffi)).
:- use_module(library(format)).

assert_true(Goal, Label) :-
    ( call(Goal) -> true
    ; throw(error(assertion_failed(Label), test_windows_ffi_abi/0))
    ).

run_tests :-
    use_foreign_module("kernel32.dll", [
        'GetCurrentProcessId'([], uint32),
        'GetTickCount64'([], uint64),
        'GetProcessHeap'([], ptr),
        'GetCommandLineA'([], cstr),
        'lstrlenA'([cstr], sint32)
    ]),
    ffi:'GetCurrentProcessId'(Pid),
    assert_true(Pid > 0, uint32_positive),
    ffi:'GetTickCount64'(Ticks),
    assert_true(Ticks > 255, uint64_not_truncated),
    ffi:'GetProcessHeap'(Heap),
    assert_true(Heap > 255, pointer_not_truncated),
    ffi:'GetCommandLineA'(CommandLine),
    assert_true(CommandLine = [_|_], cstr_return_not_empty),
    ffi:'lstrlenA'("ScryNeuro", Length),
    assert_true(Length =:= 9, cstr_and_sint32),
    format("WINDOWS FFI ABI PASSED pid=~d ticks=~d heap=~d~n", [Pid, Ticks, Heap]).

main :-
    catch(
        run_tests,
        Error,
        ( format("WINDOWS FFI ABI FAILURE: ~q~n", [Error]), halt(1) )
    ),
    halt.

:- initialization(main).
