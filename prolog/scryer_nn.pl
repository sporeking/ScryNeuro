%% ===========================================================================
%% ScryNeuro Neural Network Plugin
%% ===========================================================================
%%
%% High-level predicates for working with neural networks via the
%% scryer_nn_runtime Python module.
%%
%% Usage:
%%   :- use_module('prolog/scryer_nn').
%%   ?- nn_load(mnist, "models/mnist.pt", [input_shape=[1,28,28]]).
%%   ?- nn_predict(mnist, ImageTensor, Prediction).

:- module(scryer_nn, [
    nn_load/3,
    nn_load/4,
    nn_predict/3,
    nn_predict/4
]).

:- use_module('scryer_py').
:- use_module(library(format)).

:- dynamic(nn_registry/2).   %% nn_registry(Name, PyHandle)

%% nn_load(+Name, +Path, +Options): Load a neural network model.
%%   Name: atom identifying the model
%%   Path: string path to model file
%%   Options: list of key=value pairs
nn_load(Name, Path, Options) :-
    nn_load(Name, Path, Options, _Handle).

nn_load(Name, Path, Options, Handle) :-
    ( nn_registry(Name, _) ->
        throw(error(model_already_loaded(Name), nn_load/4))
    ; true
    ),
    py_import("scryer_nn_runtime", Runtime),
    %% Build kwargs dict from options list
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_from_str(Path, PyPath),
    py_call(Runtime, "load_model", PyPath, Kwargs, Handle),
    assertz(nn_registry(Name, Handle)),
    py_free(Kwargs),
    py_free(PyPath),
    py_free(Runtime).

%% nn_predict(+Name, +Input, -Output): Run inference.
nn_predict(Name, Input, Output) :-
    nn_predict(Name, Input, Output, []).

nn_predict(Name, Input, Output, Options) :-
    ( nn_registry(Name, ModelHandle) -> true
    ; throw(error(model_not_loaded(Name), nn_predict/4))
    ),
    py_import("scryer_nn_runtime", Runtime),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "predict", ModelHandle, Input, Output0),
    ( Kwargs \= 0 -> py_free(Kwargs) ; true ),
    py_free(Runtime),
    Output = Output0.
