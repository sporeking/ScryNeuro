%% ===========================================================================
%% ScryNeuro: Scryer Prolog ↔ Python Bridge (Core)
%% ===========================================================================
%%
%% Usage:
%%   :- op(700, xfx, :=).  % Required before use_module if using := operator
%%   :- use_module('prolog/scryer_py').
%%   ?- py_init.
%%   ?- py_eval("1 + 2", X), py_to_int(X, V).
%%   V = 3.
%%
%% Plugin modules (load separately as needed):
%%   :- use_module('prolog/scryer_nn').  % nn_load/nn_predict
%%   :- use_module('prolog/scryer_llm'). % llm_load/llm_generate
%%   :- use_module('prolog/scryer_rl').  % rl_create/rl_train/rl_action etc.
%%
%% NOTE: The := operator must be defined before the module declaration
%%        so that the Prolog parser can handle it in the export list.
:- op(700, xfx, :=).
%% NOTE: The := operator must be defined before the module declaration
%%        so that the Prolog parser can handle it in the export list.
:- op(700, xfx, :=).

:- module(scryer_py, [
    % Lifecycle
    py_init/0,
    py_init/1,
    py_finalize/0,
    % Evaluation
    py_eval/2,
    py_exec/1,
    py_exec_lines/1,
    % Modules
    py_import/2,
    % Attribute access
    py_getattr/3,
    py_setattr/3,
    % Method calls
    py_call/3,
    py_call/4,
    py_call/5,
    py_call/6,
    py_calln/4,
    % Direct calls
    py_invoke/2,
    py_invoke/3,
    py_invoke/4,
    py_invoken/3,
    % Type conversion
    py_to_str/2,
    py_to_repr/2,
    py_to_int/2,
    py_to_float/2,
    py_to_bool/2,
    py_from_int/2,
    py_from_float/2,
    py_from_bool/2,
    py_from_str/2,
    % None
    py_none/1,
    py_is_none/1,
    % JSON
    py_to_json/2,
    py_from_json/2,
    % Collections
    py_list_new/1,
    py_list_append/2,
    py_list_get/3,
    py_list_len/2,
    py_list_from_handles/2,
    py_dict_new/1,
    py_dict_set/3,
    py_dict_get/3,
    % Memory
    py_free/1,
    py_handle_count/1,
    % Error
    py_last_error/1,
    % Resource management
    with_py/2,
    % Helpers (exported for use by plugin modules)
    load_options/2,
    state_to_json/2,
    json_to_value/2,
    % Operators
    (':=')/2
]).

:- use_module(library(ffi)).
:- use_module(library(lists)).
:- use_module(library(format)).

%% ---------------------------------------------------------------------------
%% FFI Declarations
%% ---------------------------------------------------------------------------
%% Loaded lazily by py_init/0. Tries ./libscryneuro.dylib first (macOS),
%% then falls back to ./libscryneuro.so (Linux).
%%
%% Scryer FFI syntax:
%%   use_foreign_module(Path, [
%%       'function_name'([ArgType1, ArgType2, ...], ReturnType)
%%   ]).
%%
%% Supported types: void, bool, uint8..uint64, sint8..sint64, f32, f64, ptr, cstr

:- dynamic(initialized/0).

load_ffi(LibPath) :-
    use_foreign_module(LibPath, [
        % Lifecycle
        'spy_init'([], sint32),
        'spy_finalize'([], void),
        % Evaluation
        'spy_eval'([cstr], ptr),
        'spy_exec'([cstr], sint32),
        % Modules
        'spy_import'([cstr], ptr),
        % Attribute access
        'spy_getattr'([ptr, cstr], ptr),
        'spy_setattr'([ptr, cstr, ptr], sint32),
    % Method calls
    % Direct calls (callable objects)
        'spy_call0'([ptr], ptr),
        'spy_call1'([ptr, ptr], ptr),
        'spy_call2'([ptr, ptr, ptr], ptr),
        'spy_call3'([ptr, ptr, ptr, ptr], ptr),
    % Method calls: obj.method(args...)
        'spy_invoke0'([ptr, cstr], ptr),
        'spy_invoke1'([ptr, cstr, ptr], ptr),
        'spy_invoke2'([ptr, cstr, ptr, ptr], ptr),
        'spy_invoke3'([ptr, cstr, ptr, ptr, ptr], ptr),
        'spy_invoken'([ptr, cstr, ptr], ptr),
    % Direct calls with list/tuple args
        'spy_calln'([ptr, ptr], ptr),
    % Type conversion: Python → C
        'spy_to_str'([ptr], cstr),
        'spy_to_repr'([ptr], cstr),
        'spy_to_int'([ptr], sint64),
        'spy_to_float'([ptr], f64),
        'spy_to_bool'([ptr], sint32),
    % Type conversion: C → Python
        'spy_from_int'([sint64], ptr),
        'spy_from_float'([f64], ptr),
        'spy_from_bool'([sint32], ptr),
        'spy_from_str'([cstr], ptr),
    % None
        'spy_none'([], ptr),
        'spy_is_none'([ptr], sint32),
    % JSON bridge
        'spy_to_json'([ptr], cstr),
        'spy_from_json'([cstr], ptr),
    % Collections
        'spy_list_new'([], ptr),
        'spy_list_append'([ptr, ptr], sint32),
        'spy_list_get'([ptr, sint64], ptr),
        'spy_list_len'([ptr], sint64),
        'spy_dict_new'([], ptr),
        'spy_dict_set'([ptr, cstr, ptr], sint32),
        'spy_dict_get'([ptr, cstr], ptr),
    % Memory management
        'spy_drop'([ptr], void),
        'spy_handle_count'([], sint64),
    % Error handling
        'spy_last_error'([], cstr),
        'spy_last_error_clear'([], void),
        'spy_cstr_free'([ptr], void)
    ]).

%% ---------------------------------------------------------------------------
%% Error Checking
%% ---------------------------------------------------------------------------

% Check if a handle is valid (non-zero).
check_handle(Handle, Context) :-
    ( Handle =:= 0 ->
        ffi:'spy_last_error'(Err),
        throw(error(python_error(Err), Context))
    ; true
    ).

% Check if a status code indicates success (0).
check_status(Status, Context) :-
    ( Status =:= 0 -> true
    ; ffi:'spy_last_error'(Err),
      throw(error(python_error(Err), Context))
    ).

%% ---------------------------------------------------------------------------
%% Lifecycle
%% ---------------------------------------------------------------------------

%% py_init/0: Initialize with auto-detected library path.
%% Tries libscryneuro.dylib first (macOS), falls back to libscryneuro.so (Linux).
py_init :-
    ( initialized -> true
    ; ( catch((open('./libscryneuro.dylib', read, S), close(S)), _, fail) ->
        ( load_ffi("./libscryneuro.dylib") -> true
        ; throw(error(python_error("Failed to load libscryneuro.dylib. If using Conda on macOS, try: DYLD_LIBRARY_PATH=\".:$PYLIB:$DYLD_LIBRARY_PATH\" scryer-prolog ..."), py_init/0))
        )
      ; catch((open('./libscryneuro.so', read, S), close(S)), _, fail) ->
        ( load_ffi("./libscryneuro.so") -> true
        ; throw(error(python_error("Failed to load libscryneuro.so. Check LD_LIBRARY_PATH."), py_init/0))
        )
      ; throw(error(python_error("Could not find libscryneuro.dylib or libscryneuro.so. Please build the project and copy the library to the current directory."), py_init/0))
      ),
      ffi:'spy_init'(Status),
      check_status(Status, py_init/0),
      assertz(initialized)
    ).

%% py_init/1: Initialize with a custom library path.
py_init(LibPath) :-
    ( initialized -> true
    ; load_ffi(LibPath),
      ffi:'spy_init'(Status),
      check_status(Status, py_init/1),
      assertz(initialized)
    ).

%% py_finalize/0: Clean up Python and release all handles.
py_finalize :-
    ( initialized ->
        ffi:'spy_finalize',
        retractall(initialized)
    ; true
    ).

%% ---------------------------------------------------------------------------
%% Evaluation
%% ---------------------------------------------------------------------------

%% py_eval(+Code, -Handle): Evaluate a Python expression.
%%   ?- py_eval("1 + 2", H).
py_eval(Code, Handle) :-
    ffi:'spy_eval'(Code, Handle),
    check_handle(Handle, py_eval/2).

%% py_exec(+Code): Execute Python statements.
%%   ?- py_exec("import numpy as np").
py_exec(Code) :-
    ffi:'spy_exec'(Code, Status),
    check_status(Status, py_exec/1).

%% py_exec_lines(+Lines): Execute multi-line Python code.
%% Takes a list of strings (one per line), joins them with newlines,
%% and passes to py_exec. This is needed because Scryer Prolog does
%% not support \n escape sequences in double-quoted strings.
%%   ?- py_exec_lines(["class Foo:", "    def bar(self):", "        return 42"]).
py_exec_lines(Lines) :-
    char_code(NL, 10),
    join_lines(Lines, NL, Code),
    py_exec(Code).

join_lines([], _, []).
join_lines([L], _, L).
join_lines([L|Ls], NL, Result) :-
    Ls = [_|_],
    join_lines(Ls, NL, Rest),
    append(L, [NL|Rest], Result).
%% ---------------------------------------------------------------------------
%% Modules
%% ---------------------------------------------------------------------------

%% py_import(+ModuleName, -Handle): Import a Python module.
%%   ?- py_import("numpy", NP).
py_import(ModuleName, Handle) :-
    ffi:'spy_import'(ModuleName, Handle),
    check_handle(Handle, py_import/2).

%% ---------------------------------------------------------------------------
%% Attribute Access
%% ---------------------------------------------------------------------------

%% py_getattr(+Obj, +AttrName, -Value): Get attribute from Python object.
%%   ?- py_import("math", M), py_getattr(M, "pi", Pi).
py_getattr(Obj, AttrName, Value) :-
    ffi:'spy_getattr'(Obj, AttrName, Value),
    check_handle(Value, py_getattr/3).

%% py_setattr(+Obj, +AttrName, +Value): Set attribute on Python object.
py_setattr(Obj, AttrName, Value) :-
    ffi:'spy_setattr'(Obj, AttrName, Value, Status),
    check_status(Status, py_setattr/3).

%% ---------------------------------------------------------------------------
%% Method Calls: obj.method(args...)
%% ---------------------------------------------------------------------------

%% py_call(+Obj, +Method, -Result): Call method with no arguments.
py_call(Obj, Method, Result) :-
    ffi:'spy_invoke0'(Obj, Method, Result),
    check_handle(Result, py_call/3).

%% py_call(+Obj, +Method, +Arg1, -Result): Call method with 1 argument.
py_call(Obj, Method, Arg1, Result) :-
    ffi:'spy_invoke1'(Obj, Method, Arg1, Result),
    check_handle(Result, py_call/4).

%% py_call(+Obj, +Method, +Arg1, +Arg2, -Result): Call method with 2 arguments.
py_call(Obj, Method, Arg1, Arg2, Result) :-
    ffi:'spy_invoke2'(Obj, Method, Arg1, Arg2, Result),
    check_handle(Result, py_call/5).

%% py_call(+Obj, +Method, +Arg1, +Arg2, +Arg3, -Result): Call method with 3 arguments.
py_call(Obj, Method, Arg1, Arg2, Arg3, Result) :-
    ffi:'spy_invoke3'(Obj, Method, Arg1, Arg2, Arg3, Result),
    check_handle(Result, py_call/6).

py_calln(Obj, Method, Args, Result) :-
    ( Args = [_|_] ->
        py_list_from_handles(Args, ArgsHandle),
        with_py(ArgsHandle, (
            ffi:'spy_invoken'(Obj, Method, ArgsHandle, Result),
            check_handle(Result, py_calln/4)
        ))
    ; Args = [] ->
        py_list_from_handles([], ArgsHandle),
        with_py(ArgsHandle, (
            ffi:'spy_invoken'(Obj, Method, ArgsHandle, Result),
            check_handle(Result, py_calln/4)
        ))
    ; ffi:'spy_invoken'(Obj, Method, Args, Result),
      check_handle(Result, py_calln/4)
    ).

%% ---------------------------------------------------------------------------
%% Direct Calls: callable(args...)
%% ---------------------------------------------------------------------------

%% py_invoke(+Callable, -Result): Call a callable with no arguments.
py_invoke(Callable, Result) :-
    ffi:'spy_call0'(Callable, Result),
    check_handle(Result, py_invoke/2).

%% py_invoke(+Callable, +Arg1, -Result): Call with 1 argument.
py_invoke(Callable, Arg1, Result) :-
    ffi:'spy_call1'(Callable, Arg1, Result),
    check_handle(Result, py_invoke/3).

%% py_invoke(+Callable, +Arg1, +Arg2, -Result): Call with 2 arguments.
py_invoke(Callable, Arg1, Arg2, Result) :-
    ffi:'spy_call2'(Callable, Arg1, Arg2, Result),
    check_handle(Result, py_invoke/4).

py_invoken(Callable, Args, Result) :-
    ( Args = [_|_] ->
        py_list_from_handles(Args, ArgsHandle),
        with_py(ArgsHandle, (
            ffi:'spy_calln'(Callable, ArgsHandle, Result),
            check_handle(Result, py_invoken/3)
        ))
    ; Args = [] ->
        py_list_from_handles([], ArgsHandle),
        with_py(ArgsHandle, (
            ffi:'spy_calln'(Callable, ArgsHandle, Result),
            check_handle(Result, py_invoken/3)
        ))
    ; ffi:'spy_calln'(Callable, Args, Result),
      check_handle(Result, py_invoken/3)
    ).

%% ---------------------------------------------------------------------------
%% Type Conversion
%% ---------------------------------------------------------------------------

%% py_to_str(+Handle, -String): Get str(obj).
py_to_str(Handle, String) :-
    ffi:'spy_to_str'(Handle, String).

%% py_to_repr(+Handle, -String): Get repr(obj).
py_to_repr(Handle, String) :-
    ffi:'spy_to_repr'(Handle, String).

%% py_to_int(+Handle, -Value): Extract integer.
py_to_int(Handle, Value) :-
    ffi:'spy_to_int'(Handle, Value).

%% py_to_float(+Handle, -Value): Extract float.
py_to_float(Handle, Value) :-
    ffi:'spy_to_float'(Handle, Value).

%% py_to_bool(+Handle, -Value): Extract boolean (true/false).
py_to_bool(Handle, Value) :-
    ffi:'spy_to_bool'(Handle, Code),
    ( Code =:= 1 -> Value = true
    ; Code =:= 0 -> Value = false
    ; ffi:'spy_last_error'(Err),
      throw(error(python_error(Err), py_to_bool/2))
    ).

%% py_from_int(+Value, -Handle): Create Python int.
py_from_int(Value, Handle) :-
    ffi:'spy_from_int'(Value, Handle),
    check_handle(Handle, py_from_int/2).

%% py_from_float(+Value, -Handle): Create Python float.
py_from_float(Value, Handle) :-
    ffi:'spy_from_float'(Value, Handle),
    check_handle(Handle, py_from_float/2).

%% py_from_bool(+Value, -Handle): Create Python bool.
%%   py_from_bool(true, H) or py_from_bool(false, H).
py_from_bool(true, Handle) :- !,
    ffi:'spy_from_bool'(1, Handle),
    check_handle(Handle, py_from_bool/2).
py_from_bool(false, Handle) :-
    ffi:'spy_from_bool'(0, Handle),
    check_handle(Handle, py_from_bool/2).

%% py_from_str(+String, -Handle): Create Python str.
py_from_str(String, Handle) :-
    ffi:'spy_from_str'(String, Handle),
    check_handle(Handle, py_from_str/2).

%% ---------------------------------------------------------------------------
%% None
%% ---------------------------------------------------------------------------

%% py_none(-Handle): Get a handle to Python None.
py_none(Handle) :-
    ffi:'spy_none'(Handle),
    check_handle(Handle, py_none/1).

%% py_is_none(+Handle): Succeeds if the handle points to None.
py_is_none(Handle) :-
    ffi:'spy_is_none'(Handle, Code),
    Code =:= 1.

%% ---------------------------------------------------------------------------
%% JSON Bridge
%% ---------------------------------------------------------------------------

%% py_to_json(+Handle, -JsonString): Serialize Python object to JSON.
py_to_json(Handle, Json) :-
    ffi:'spy_to_json'(Handle, Json).

%% py_from_json(+JsonString, -Handle): Deserialize JSON to Python object.
py_from_json(Json, Handle) :-
    ffi:'spy_from_json'(Json, Handle),
    check_handle(Handle, py_from_json/2).

%% ---------------------------------------------------------------------------
%% Collections
%% ---------------------------------------------------------------------------

%% py_list_new(-Handle): Create empty Python list.
py_list_new(Handle) :-
    ffi:'spy_list_new'(Handle),
    check_handle(Handle, py_list_new/1).

%% py_list_append(+List, +Item): Append item to list.
py_list_append(List, Item) :-
    ffi:'spy_list_append'(List, Item, Status),
    check_status(Status, py_list_append/2).

%% py_list_get(+List, +Index, -Item): Get item at index.
py_list_get(List, Index, Item) :-
    ffi:'spy_list_get'(List, Index, Item),
    check_handle(Item, py_list_get/3).

%% py_list_len(+List, -Len): Get list length.
py_list_len(List, Len) :-
    ffi:'spy_list_len'(List, Len).

py_list_from_handles(Handles, List) :-
    py_list_new(List),
    py_list_from_handles(Handles, List, List).

py_list_from_handles([], List, List).
py_list_from_handles([H | Rest], List, Out) :-
    py_list_append(List, H),
    py_list_from_handles(Rest, List, Out).

%% py_dict_new(-Handle): Create empty Python dict.
py_dict_new(Handle) :-
    ffi:'spy_dict_new'(Handle),
    check_handle(Handle, py_dict_new/1).

%% py_dict_set(+Dict, +Key, +Value): Set key-value pair (key is atom/string).
py_dict_set(Dict, Key, Value) :-
    ffi:'spy_dict_set'(Dict, Key, Value, Status),
    check_status(Status, py_dict_set/3).

%% py_dict_get(+Dict, +Key, -Value): Get value by key.
py_dict_get(Dict, Key, Value) :-
    ffi:'spy_dict_get'(Dict, Key, Value),
    check_handle(Value, py_dict_get/3).

%% ---------------------------------------------------------------------------
%% Memory Management
%% ---------------------------------------------------------------------------

%% py_free(+Handle): Release a Python object handle.
py_free(Handle) :-
    ffi:'spy_drop'(Handle).

%% py_handle_count(-Count): Number of live handles (diagnostic).
py_handle_count(Count) :-
    ffi:'spy_handle_count'(Count).

%% py_last_error(-Error): Get last error message (empty if none).
py_last_error(Error) :-
    ffi:'spy_last_error'(Error).

%% ---------------------------------------------------------------------------
%% Resource Management (RAII-style)
%% ---------------------------------------------------------------------------

%% with_py(+Handle, +Goal): Execute Goal, then free Handle regardless of outcome.
%%
%%   ?- py_eval("42", H), with_py(H, (py_to_int(H, V), write(V))).
%%
with_py(Handle, Goal) :-
    ( catch(Goal, E, (py_free(Handle), throw(E))) ->
        py_free(Handle)
    ; py_free(Handle),
      fail
    ).

%% ---------------------------------------------------------------------------
%% Syntactic Sugar: := operator
%% ---------------------------------------------------------------------------
%%
%% Var := py_eval(Expr)       → py_eval(Expr, Var)
%% Var := Module:Func         → py_call(Module, Func, Var)
%% Var := Module:Func(A)      → py_call(Module, Func, A, Var)
%% Var := Module:Func(A,B)    → py_call(Module, Func, A, B, Var)
%% Var := Module:Func(A,B,C)  → py_call(Module, Func, A, B, C, Var)
%% Var := Module:Func(Args..) → py_calln(Module, Func, ArgsList, Var)
%%
%% Examples:
%%   ?- NP := py_import("numpy"),
%%      Arr := NP:"array"([1,2,3]),
%%      Shape := Arr:"shape".


Var := py_eval(Expr) :- !,
    py_eval(Expr, Var).

Var := py_import(Module) :- !,
    py_import(Module, Var).

Var := py_from_json(Json) :- !,
    py_from_json(Json, Var).

Var := py_from_int(V) :- !,
    py_from_int(V, Var).

Var := py_from_float(V) :- !,
    py_from_float(V, Var).

Var := py_from_str(V) :- !,
    py_from_str(V, Var).

%% Method call sugar: Var := Obj:Method or Var := Obj:Method(Args...)
Var := Obj:Call :- !,
    ( Call = [_|_] ->
        %% Call is a string (char list) like "pi" - attribute access (getattr)
        py_getattr(Obj, Call, Var)
    ; atom(Call) ->
        %% Call is an atom like upper - no-arg method call: obj.method()
        atom_chars(Call, MethodStr),
        py_call(Obj, MethodStr, Var)
    ; %% Call is compound like replace(A,B) - method call with args
        Call =.. [Method | Args],
        atom_chars(Method, MethodStr),
        ( Args = [A, B, C] -> py_call(Obj, MethodStr, A, B, C, Var)
        ; Args = [A, B]    -> py_call(Obj, MethodStr, A, B, Var)
        ; Args = [A]       -> py_call(Obj, MethodStr, A, Var)
        ; Args = []        -> py_call(Obj, MethodStr, Var)
        ; py_calln(Obj, MethodStr, Args, Var)
        )
    ).

%% ---------------------------------------------------------------------------
%% Helpers (exported for plugin modules)
%% ---------------------------------------------------------------------------

%% state_to_json(+State, -Json): Convert a Prolog state (list of numbers) to JSON.
state_to_json(State, Json) :-
    state_to_json_chars(State, Inner),
    append("[", Inner, T1),
    append(T1, "]", Json).

state_to_json_chars([], []).
state_to_json_chars([X], Chars) :-
    number_chars(X, Chars).
state_to_json_chars([X|Xs], Chars) :-
    Xs = [_|_],
    number_chars(X, XChars),
    append(XChars, ",", WithComma),
    state_to_json_chars(Xs, Rest),
    append(WithComma, Rest, Chars).

%% json_to_value(+Json, -Value): Parse a simple JSON value.
%% Handles numbers, strings, and passes through complex JSON as char list.
json_to_value(Json, Value) :-
    ( Json = [] ->
        Value = []
    ; Json = [0'{|_] ->
        %% JSON object -- return as char list (the user can parse further)
        Value = Json
    ; Json = [0'[|_] ->
        %% JSON array -- return as char list
        Value = Json
    ; Json = [0'"| _] ->
        %% JSON string -- return as char list
        Value = Json
    ; %% Try to parse as number
      catch(
          (number_chars(N, Json), Value = N),
          _,
          Value = Json
      )
    ).

%% load_options(+DictHandle, +OptionList): Load key=value pairs into a Python dict.
load_options(_, []) :- !.
load_options(Dict, [Key=Value | Rest]) :- !,
    ( number(Value) ->
        ( float(Value) ->
            py_from_float(Value, PyVal)
        ; py_from_int(Value, PyVal)
        )
    ; atom(Value) ->
        atom_chars(Value, Chars),
        atom_chars(StrValue, Chars),
        py_from_str(StrValue, PyVal)
    ; (Value = [] ; Value = [V1|_], number(V1)) ->
        %% Convert Prolog list of numbers to Python list via JSON
        state_to_json(Value, ListJson),
        py_from_json(ListJson, PyVal)
    ; py_from_str(Value, PyVal)
    ),
    atom_chars(Key, KeyChars),
    atom_chars(KeyStr, KeyChars),
    py_dict_set(Dict, KeyStr, PyVal),
    py_free(PyVal),
    load_options(Dict, Rest).
load_options(Dict, [_ | Rest]) :-
    load_options(Dict, Rest).
