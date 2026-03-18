%% ===========================================================================
%% ScryNeuro Agent Plugin
%% ===========================================================================
%%
%% Extensible LLM-agent framework plugin built on top of scryer_py.
%%
%% Usage:
%%   :- use_module('prolog/scryer_agent').
%%   ?- agent_create(demo, "mock-v1", [provider=mock]).
%%   ?- agent_register_tool(demo, add, 'scryer_agent_tools:add_numbers').
%%   ?- agent_run(demo, "tool:add {\"a\": 2, \"b\": 3}", Out).

:- module(scryer_agent, [
    agent_create/3,
    agent_create/4,
    agent_register_tool/3,
    agent_register_tool/4,
    agent_register_builtin_tools/2,
    agent_load_skill/2,
    agent_load_skill/3,
    agent_load_plugin/2,
    agent_load_plugin/3,
    agent_save_session/2,
    agent_load_session/2,
    agent_load_session/3,
    agent_step/3,
    agent_step/4,
    agent_run/3,
    agent_run/4,
    agent_trace/2,
    agent_list/1,
    agent_unload/1
]).

:- use_module('scryer_py').

:- dynamic(agent_registry/2).  %% agent_registry(Name, PyHandle)

%% to_py_str(+Value, -Handle): Convert atom/string to Python str handle.
to_py_str(Value, Handle) :-
    ( atom(Value) ->
        atom_chars(Value, Chars),
        py_from_str(Chars, Handle)
    ; py_from_str(Value, Handle)
    ).

%% agent_create(+Name, +ModelId, +Options)
agent_create(Name, ModelId, Options) :-
    agent_create(Name, ModelId, Options, _).

%% agent_create(+Name, +ModelId, +Options, -Handle)
%%   Name: atom identifier
%%   ModelId: model identifier string (e.g. "gpt-4", "mock-v1")
%%   Options: provider and runtime options
agent_create(Name, ModelId, Options, Handle) :-
    ( agent_registry(Name, _) ->
        throw(error(agent_already_exists(Name), agent_create/4))
    ; true
    ),
    py_import("scryer_agent_runtime", Runtime),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    atom_chars(Name, NameChars),
    py_from_str(NameChars, PyName),
    to_py_str(ModelId, PyModelId),
    py_call(Runtime, "agent_create", PyName, PyModelId, Kwargs, Handle),
    assertz(agent_registry(Name, Handle)),
    py_free(PyModelId),
    py_free(PyName),
    py_free(Kwargs),
    py_free(Runtime).

%% agent_register_tool(+AgentName, +ToolName, +Entrypoint)
agent_register_tool(AgentName, ToolName, Entrypoint) :-
    agent_register_tool(AgentName, ToolName, Entrypoint, []).

%% agent_register_tool(+AgentName, +ToolName, +Entrypoint, +Options)
%%   Entrypoint format: 'python_module:function_name'
%%   Example: 'scryer_agent_tools:add_numbers'
agent_register_tool(AgentName, ToolName, Entrypoint, Options) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_register_tool/4))
    ),
    py_import("scryer_agent_runtime", Runtime),
    atom_chars(ToolName, ToolChars),
    py_from_str(ToolChars, PyToolName),
    to_py_str(Entrypoint, PyEntrypoint),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_calln(Runtime, "agent_register_tool", [AgentHandle, PyToolName, PyEntrypoint, Kwargs], ResultH),
    py_free(ResultH),
    py_free(Kwargs),
    py_free(PyEntrypoint),
    py_free(PyToolName),
    py_free(Runtime).

%% agent_register_builtin_tools(+AgentName, +ToolNames)
%% Tool names: web_fetch, shell_exec, read_file, write_file,
%%             list_dir, grep_text, add, multiply, reverse.
agent_register_builtin_tools(AgentName, ToolNames) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_register_builtin_tools/2))
    ),
    py_import("scryer_agent_runtime", Runtime),
    py_list_from_handles([], ToolListH),
    add_tool_name_handles(ToolNames, ToolListH),
    py_call(Runtime, "agent_register_builtin_tools", AgentHandle, ToolListH, ResultH),
    py_free(ResultH),
    py_free(ToolListH),
    py_free(Runtime).

add_tool_name_handles([], _).
add_tool_name_handles([Name | Rest], ToolListH) :- !,
    atom_chars(Name, NameChars),
    py_from_str(NameChars, NameH),
    py_list_append(ToolListH, NameH),
    py_free(NameH),
    add_tool_name_handles(Rest, ToolListH).
add_tool_name_handles(Name, ToolListH) :-
    atom_chars(Name, NameChars),
    py_from_str(NameChars, NameH),
    py_list_append(ToolListH, NameH),
    py_free(NameH).

%% agent_load_skill(+AgentName, +SkillName)
agent_load_skill(AgentName, SkillName) :-
    agent_load_skill(AgentName, SkillName, []).

%% agent_load_skill(+AgentName, +SkillName, +Options)
%% Options may include: [skills_dir="skills"]
agent_load_skill(AgentName, SkillName, Options) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_load_skill/3))
    ),
    py_import("scryer_agent_runtime", Runtime),
    atom_chars(SkillName, SkillChars),
    py_from_str(SkillChars, PySkillName),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "agent_load_skill", AgentHandle, PySkillName, Kwargs, ResultH),
    py_free(ResultH),
    py_free(Kwargs),
    py_free(PySkillName),
    py_free(Runtime).

%% agent_load_plugin(+AgentName, +PluginEntrypoint)
%% Entrypoint format: 'python_module:function_name'
agent_load_plugin(AgentName, PluginEntrypoint) :-
    agent_load_plugin(AgentName, PluginEntrypoint, []).

%% agent_load_plugin(+AgentName, +PluginEntrypoint, +Options)
agent_load_plugin(AgentName, PluginEntrypoint, Options) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_load_plugin/3))
    ),
    py_import("scryer_agent_runtime", Runtime),
    to_py_str(PluginEntrypoint, PyEntrypoint),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "agent_load_plugin", AgentHandle, PyEntrypoint, Kwargs, ResultH),
    py_free(ResultH),
    py_free(Kwargs),
    py_free(PyEntrypoint),
    py_free(Runtime).

%% agent_save_session(+AgentName, +Path)
agent_save_session(AgentName, Path) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_save_session/2))
    ),
    py_import("scryer_agent_runtime", Runtime),
    to_py_str(Path, PyPath),
    py_call(Runtime, "agent_save_session", AgentHandle, PyPath, ResultH),
    py_free(ResultH),
    py_free(PyPath),
    py_free(Runtime).

%% agent_load_session(+Name, +Path)
agent_load_session(Name, Path) :-
    agent_load_session(Name, Path, []).

%% agent_load_session(+Name, +Path, +Options)
%% Options: provider=<atom/string>, model_id=<string>
agent_load_session(Name, Path, Options) :-
    ( agent_registry(Name, _) ->
        throw(error(agent_already_exists(Name), agent_load_session/3))
    ; true
    ),
    py_import("scryer_agent_runtime", Runtime),
    atom_chars(Name, NameChars),
    py_from_str(NameChars, PyName),
    to_py_str(Path, PyPath),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "agent_load_session", PyName, PyPath, Kwargs, Handle),
    assertz(agent_registry(Name, Handle)),
    py_free(Kwargs),
    py_free(PyPath),
    py_free(PyName),
    py_free(Runtime).

%% agent_step(+AgentName, +Input, -OutputJson)
agent_step(AgentName, Input, OutputJson) :-
    agent_step(AgentName, Input, OutputJson, []).

%% agent_step(+AgentName, +Input, -OutputJson, +Options)
%% Performs one agent turn. If the agent decides to call a tool,
%% output may indicate done=false.
agent_step(AgentName, Input, OutputJson, Options) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_step/4))
    ),
    py_import("scryer_agent_runtime", Runtime),
    to_py_str(Input, PyInput),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "agent_step", AgentHandle, PyInput, Kwargs, OutH),
    py_to_json(OutH, OutputJson),
    py_free(OutH),
    py_free(Kwargs),
    py_free(PyInput),
    py_free(Runtime).

%% agent_run(+AgentName, +Input, -OutputJson)
agent_run(AgentName, Input, OutputJson) :-
    agent_run(AgentName, Input, OutputJson, []).

%% agent_run(+AgentName, +Input, -OutputJson, +Options)
%% Runs iterative loop until done=true or max_steps reached.
agent_run(AgentName, Input, OutputJson, Options) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_run/4))
    ),
    py_import("scryer_agent_runtime", Runtime),
    to_py_str(Input, PyInput),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "agent_run", AgentHandle, PyInput, Kwargs, OutH),
    py_to_json(OutH, OutputJson),
    py_free(OutH),
    py_free(Kwargs),
    py_free(PyInput),
    py_free(Runtime).

%% agent_trace(+AgentName, -TraceJson)
agent_trace(AgentName, TraceJson) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_trace/2))
    ),
    py_import("scryer_agent_runtime", Runtime),
    py_call(Runtime, "agent_trace", AgentHandle, TraceH),
    py_to_json(TraceH, TraceJson),
    py_free(TraceH),
    py_free(Runtime).

%% agent_list(-AgentsJson)
%% Returns JSON array of registered agent names.
agent_list(AgentsJson) :-
    py_import("scryer_agent_runtime", Runtime),
    py_call(Runtime, "agent_list", ListH),
    py_to_json(ListH, AgentsJson),
    py_free(ListH),
    py_free(Runtime).

%% agent_unload(+AgentName)
%% Unloads runtime state and frees the Prolog-held handle.
agent_unload(AgentName) :-
    ( agent_registry(AgentName, AgentHandle) -> true
    ; throw(error(agent_not_found(AgentName), agent_unload/1))
    ),
    py_import("scryer_agent_runtime", Runtime),
    py_call(Runtime, "agent_unload", AgentHandle, ResultH),
    py_free(ResultH),
    py_free(Runtime),
    py_free(AgentHandle),
    retractall(agent_registry(AgentName, _)).
