%% ===========================================================================
%% ScryNeuro Agent User-Facing API (Facade)
%% ===========================================================================

:- module(scryer_agent_api, [
    agent_providers/1,
    agent_profiles/1,
    agent_profile/2,
    agent_models/2,
    agent_tools/1,
    agent_skills/1,
    agent_new_from_profile/2,
    agent_new_from_profile/3,
    agent_new/4,
    agent_enable_tools/2,
    agent_enable_skills/2,
    agent_run/3,
    agent_trace/2,
    agent_close/1
]).

:- use_module('scryer_agent', [
    agent_list_profiles/1,
    agent_get_profile/2,
    agent_create/3,
    agent_create_from_profile/3,
    agent_register_builtin_tools/2,
    agent_discover_skills/2,
    agent_load_skill/3,
    agent_unload/1
]).
:- use_module('scryer_tool_predicates').
:- use_module(library(lists)).

agent_providers([openai, anthropic, huggingface, ollama, mock]).

agent_profiles(ProfilesJson) :-
    agent_list_profiles(ProfilesJson).

agent_profile(ProfileName, ProfileJson) :-
    agent_get_profile(ProfileName, ProfileJson).

agent_models(openai, ["auto", "gpt-4o-mini", "gpt-4.1", "glm-5"]).
agent_models(anthropic, ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"]).
agent_models(huggingface, ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]).
agent_models(ollama, ["qwen2.5:7b", "llama3.1:8b"]).
agent_models(mock, ["mock-v2", "mock-v3"]).

agent_tools(ToolsJson) :-
    tool_list_available(ToolsJson).

agent_skills(SkillsJson) :-
    agent_discover_skills(SkillsJson, [skills_dir="python/skills"]).

agent_new(Name, Provider, Model, Options) :-
    append([provider=Provider], Options, FullOptions),
    agent_create(Name, Model, FullOptions).

agent_new_from_profile(Name, ProfileName) :-
    agent_new_from_profile(Name, ProfileName, []).

agent_new_from_profile(Name, ProfileName, Options) :-
    agent_create_from_profile(Name, ProfileName, Options).

agent_enable_tools(Name, ToolList) :-
    agent_register_builtin_tools(Name, ToolList).

agent_enable_skills(_, []).
agent_enable_skills(Name, [S | Rest]) :-
    agent_load_skill(Name, S, [skills_dir="python/skills"]),
    agent_enable_skills(Name, Rest).

agent_run(Name, Task, OutputJson) :-
    scryer_agent:agent_run(Name, Task, OutputJson, [
        max_steps=5,
        max_auto_tools=4,
        temperature=0.0,
        max_tokens=900
    ]).

agent_trace(Name, TraceJson) :-
    scryer_agent:agent_trace(Name, TraceJson).

agent_close(Name) :-
    agent_unload(Name).
