%% ===========================================================================
%% ScryNeuro LLM Plugin
%% ===========================================================================
%%
%% High-level predicates for interacting with Large Language Models via the
%% scryer_llm_runtime Python module.
%%
%% Usage:
%%   :- use_module('prolog/scryer_llm').
%%   ?- llm_load(gpt, "gpt-4", [provider=openai]).
%%   ?- llm_generate(gpt, "What is 2+2?", Response).

:- module(scryer_llm, [
    llm_load/3,
    llm_load/4,
    llm_generate/3,
    llm_generate/4
]).

:- use_module('scryer_py').
:- use_module(library(format)).

:- dynamic(llm_registry/2).  %% llm_registry(Name, PyHandle)

%% llm_load(+Name, +ModelId, +Options): Load an LLM.
%%   Name: atom identifying the LLM
%%   ModelId: string model identifier (e.g., "gpt-4", "llama-3")
%%   Options: list of key=value pairs (e.g., [provider=openai, temperature=0.7])
llm_load(Name, ModelId, Options) :-
    llm_load(Name, ModelId, Options, _Handle).

llm_load(Name, ModelId, Options, Handle) :-
    ( llm_registry(Name, _) ->
        throw(error(llm_already_loaded(Name), llm_load/4))
    ; true
    ),
    py_import("scryer_llm_runtime", Runtime),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_from_str(ModelId, PyModelId),
    py_call(Runtime, "load_llm", PyModelId, Kwargs, Handle),
    assertz(llm_registry(Name, Handle)),
    py_free(Kwargs),
    py_free(PyModelId),
    py_free(Runtime).

%% llm_generate(+Name, +Prompt, -Response): Generate text with an LLM.
llm_generate(Name, Prompt, Response) :-
    llm_generate(Name, Prompt, Response, []).

llm_generate(Name, Prompt, Response, Options) :-
    ( llm_registry(Name, LLMHandle) -> true
    ; throw(error(llm_not_loaded(Name), llm_generate/4))
    ),
    py_import("scryer_llm_runtime", Runtime),
    py_from_str(Prompt, PyPrompt),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "generate", LLMHandle, PyPrompt, RespHandle),
    py_to_str(RespHandle, Response),
    py_free(RespHandle),
    py_free(PyPrompt),
    py_free(Kwargs),
    py_free(Runtime).
