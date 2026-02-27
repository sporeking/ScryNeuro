%% ===========================================================================
%% ScryNeuro Basic Example
%% ===========================================================================
%%
%% Demonstrates basic Python interop from Scryer Prolog.
%%
%% Run:
%%   $ LD_LIBRARY_PATH=. scryer-prolog examples/basic.pl
%%
%% Prerequisites:
%%   - Build: cargo build --release
%%   - Copy: cp target/release/libscryneuro.so ./

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').

%% ---------------------------------------------------------------------------
%% Example 1: Basic arithmetic
%% ---------------------------------------------------------------------------

example_arithmetic :-
    py_init,
    format("=== Arithmetic ===~n", []),

    %% Evaluate a Python expression
    py_eval("2 ** 10", Result),
    py_to_int(Result, Value),
    format("2^10 = ~d~n", [Value]),
    py_free(Result),

    %% Using the := operator
    X := py_eval("sum(range(100))"),
    py_to_int(X, Sum),
    format("sum(0..99) = ~d~n", [Sum]),
    py_free(X).

%% ---------------------------------------------------------------------------
%% Example 2: Working with Python modules
%% ---------------------------------------------------------------------------

example_modules :-
    py_init,
    format("=== Modules ===~n", []),

    %% Import math module
    py_import("math", Math),

    %% Get pi (string = attribute access)
    Pi := Math:"pi",
    py_to_float(Pi, PiVal),
    format("math.pi = ~f~n", [PiVal]),

    %% Call math.sqrt(144) using compound term (atom = method call)
    Arg := py_from_int(144),
    py_call(Math, "sqrt", Arg, SqrtResult),
    py_to_float(SqrtResult, SqrtVal),
    format("sqrt(144) = ~f~n", [SqrtVal]),

    %% Clean up
    py_free(SqrtResult),
    py_free(Arg),
    py_free(Pi),
    py_free(Math).

%% ---------------------------------------------------------------------------
%% Example 3: Lists and dicts
%% ---------------------------------------------------------------------------

example_collections :-
    py_init,
    format("=== Collections ===~n", []),

    %% Create a Python list
    py_list_new(List),
    A := py_from_int(10),
    B := py_from_int(20),
    C := py_from_int(30),
    py_list_append(List, A),
    py_list_append(List, B),
    py_list_append(List, C),
    py_list_len(List, Len),
    format("List length: ~d~n", [Len]),

    %% Get an item
    py_list_get(List, 1, Item),
    py_to_int(Item, ItemVal),
    format("List[1] = ~d~n", [ItemVal]),

    %% Create a Python dict
    py_dict_new(Dict),
    Name := py_from_str("Alice"),
    Age := py_from_int(30),
    py_dict_set(Dict, "name", Name),
    py_dict_set(Dict, "age", Age),

    %% Read back
    py_dict_get(Dict, "name", NameBack),
    py_to_str(NameBack, NameStr),
    format("Dict['name'] = ~s~n", [NameStr]),

    %% Serialize to JSON
    py_to_json(Dict, Json),
    format("JSON: ~s~n", [Json]),

    %% Clean up
    py_free(NameBack), py_free(Age), py_free(Name),
    py_free(Dict), py_free(Item),
    py_free(C), py_free(B), py_free(A), py_free(List).

%% ---------------------------------------------------------------------------
%% Example 4: Error handling
%% ---------------------------------------------------------------------------

example_errors :-
    py_init,
    format("=== Error Handling ===~n", []),

    %% Catch Python errors
    ( catch(
        py_eval("1 / 0", _),
        error(python_error(Err), _),
        format("Caught Python error: ~s~n", [Err])
    ) -> true ; true ),

    %% Invalid handle
    ( catch(
        py_to_str(99999, _),
        error(python_error(Err2), _),
        format("Caught invalid handle error: ~s~n", [Err2])
    ) -> true ; true ).

%% ---------------------------------------------------------------------------
%% Example 5: Resource management with with_py/2
%% ---------------------------------------------------------------------------

example_raii :-
    py_init,
    format("=== RAII-style resource management ===~n", []),

    py_eval("[1, 2, 3, 4, 5]", ListH),
    with_py(ListH, (
        py_to_json(ListH, Json),
        format("List as JSON: ~s~n", [Json])
    )),
    %% ListH is automatically freed here

    py_handle_count(Count),
    format("Live handles after with_py: ~d~n", [Count]).

%% ---------------------------------------------------------------------------
%% Run all examples
%% ---------------------------------------------------------------------------

:- initialization((
    example_arithmetic,
    nl,
    example_modules,
    nl,
    example_collections,
    nl,
    example_errors,
    nl,
    example_raii,
    nl,
    format("=== All basic examples complete ===~n", []),
    py_finalize
)).
