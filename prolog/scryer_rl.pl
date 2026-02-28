%% ===========================================================================
%% ScryNeuro Reinforcement Learning Plugin
%% ===========================================================================
%%
%% High-level predicates for working with RL agents via the
%% scryer_rl_runtime Python module (Tianshou v2.0).
%%
%% Usage:
%%   :- use_module('prolog/scryer_rl').
%%   ?- rl_create(my_agent, "CartPole-v1", dqn, [hidden_sizes=[64,64]]).
%%   ?- rl_train(my_agent, [max_epochs=5, epoch_num_steps=1000]).
%%   ?- rl_action(my_agent, [0.5, 0.2, 0.3, 0.1], Action).
%%   ?- rl_evaluate(my_agent, 10, Metrics).
%%   ?- rl_save(my_agent, "checkpoints/my_agent.pt").
%%   ?- rl_load(loaded_agent, "checkpoints/my_agent.pt",
%%           [env_id="CartPole-v1", algorithm=dqn]).

:- module(scryer_rl, [
    rl_create/4,
    rl_load/3,
    rl_load/4,
    rl_save/2,
    rl_action/3,
    rl_action/4,
    rl_train/2,
    rl_train/3,
    rl_evaluate/3,
    rl_info/2
]).

:- use_module('scryer_py').
:- use_module(library(format)).

:- dynamic(rl_registry/2).   %% rl_registry(Name, PyHandle)

%% rl_create(+Name, +EnvId, +Algorithm, +Options): Create an RL agent.
%%   Name: atom identifying the agent
%%   EnvId: string Gymnasium environment ID (e.g., "CartPole-v1")
%%   Algorithm: atom for RL algorithm (dqn, ppo, a2c, sac, td3, ddpg, pg, discrete_sac)
%%   Options: list of key=value pairs (e.g., [hidden_sizes=[64,64], lr=0.001])
rl_create(Name, EnvId, Algorithm, Options) :-
    ( rl_registry(Name, _) ->
        throw(error(rl_agent_already_exists(Name), rl_create/4))
    ; true
    ),
    py_import("scryer_rl_runtime", Runtime),
    py_dict_new(Kwargs),
    %% Inject name and algorithm into kwargs
    atom_chars(Name, NameChars),
    py_from_str(NameChars, PyName),
    py_dict_set(Kwargs, "name", PyName),
    py_free(PyName),
    atom_chars(Algorithm, AlgoChars),
    py_from_str(AlgoChars, PyAlgo),
    py_dict_set(Kwargs, "algorithm", PyAlgo),
    py_free(PyAlgo),
    load_options(Kwargs, Options),
    py_from_str(EnvId, PyEnvId),
    py_call(Runtime, "rl_create", PyEnvId, Kwargs, Handle),
    assertz(rl_registry(Name, Handle)),
    py_free(Kwargs),
    py_free(PyEnvId),
    py_free(Runtime).

%% rl_load(+Name, +Path, +Options): Load an RL agent from checkpoint.
%%   Name: atom identifying the agent
%%   Path: string path to checkpoint file
%%   Options: list of key=value pairs (must include env_id and algorithm)
rl_load(Name, Path, Options) :-
    rl_load(Name, Path, Options, _Handle).

rl_load(Name, Path, Options, Handle) :-
    ( rl_registry(Name, _) ->
        throw(error(rl_agent_already_exists(Name), rl_load/4))
    ; true
    ),
    py_import("scryer_rl_runtime", Runtime),
    py_dict_new(Kwargs),
    %% Inject name into kwargs
    atom_chars(Name, NameChars),
    py_from_str(NameChars, PyName),
    py_dict_set(Kwargs, "name", PyName),
    py_free(PyName),
    load_options(Kwargs, Options),
    py_from_str(Path, PyPath),
    py_call(Runtime, "rl_load", PyPath, Kwargs, Handle),
    assertz(rl_registry(Name, Handle)),
    py_free(Kwargs),
    py_free(PyPath),
    py_free(Runtime).

%% rl_save(+Name, +Path): Save an RL agent to a checkpoint file.
rl_save(Name, Path) :-
    ( rl_registry(Name, AgentHandle) -> true
    ; throw(error(rl_agent_not_loaded(Name), rl_save/2))
    ),
    py_import("scryer_rl_runtime", Runtime),
    py_from_str(Path, PyPath),
    py_call(Runtime, "rl_save", AgentHandle, PyPath, ResultH),
    py_free(PyPath),
    py_free(ResultH),
    py_free(Runtime).

%% rl_action(+Name, +State, -Action): Select an action for a given state.
%%   State: a list of numbers (observation)
%%   Action: the selected action (integer for discrete, list for continuous)
rl_action(Name, State, Action) :-
    rl_action(Name, State, Action, []).

rl_action(Name, State, Action, Options) :-
    ( rl_registry(Name, AgentHandle) -> true
    ; throw(error(rl_agent_not_loaded(Name), rl_action/4))
    ),
    py_import("scryer_rl_runtime", Runtime),
    %% Convert Prolog state list to Python list via JSON
    state_to_json(State, StateJson),
    py_from_json(StateJson, PyState),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "rl_action", AgentHandle, PyState, Kwargs, ActionHandle),
    %% Convert action back to Prolog
    py_to_json(ActionHandle, ActionJson),
    json_to_value(ActionJson, Action),
    py_free(ActionHandle),
    py_free(PyState),
    py_free(Kwargs),
    py_free(Runtime).

%% rl_train(+Name, +Options): Train an RL agent.
%%   Options: [max_epochs=5, epoch_num_steps=1000, batch_size=64, ...]
rl_train(Name, Options) :-
    rl_train(Name, Options, _Metrics).

rl_train(Name, Options, Metrics) :-
    ( rl_registry(Name, AgentHandle) -> true
    ; throw(error(rl_agent_not_loaded(Name), rl_train/3))
    ),
    py_import("scryer_rl_runtime", Runtime),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "rl_train", AgentHandle, Kwargs, MetricsHandle),
    py_to_json(MetricsHandle, MetricsJson),
    json_to_value(MetricsJson, Metrics),
    py_free(MetricsHandle),
    py_free(Kwargs),
    py_free(Runtime).

%% rl_evaluate(+Name, +NumEpisodes, -Metrics): Evaluate an RL agent.
%%   NumEpisodes: integer number of evaluation episodes
%%   Metrics: dict with mean_reward, std_reward, n_episodes, etc.
rl_evaluate(Name, NumEpisodes, Metrics) :-
    ( rl_registry(Name, AgentHandle) -> true
    ; throw(error(rl_agent_not_loaded(Name), rl_evaluate/3))
    ),
    py_import("scryer_rl_runtime", Runtime),
    py_from_int(NumEpisodes, PyN),
    py_call(Runtime, "rl_evaluate", AgentHandle, PyN, MetricsHandle),
    py_to_json(MetricsHandle, MetricsJson),
    json_to_value(MetricsJson, Metrics),
    py_free(MetricsHandle),
    py_free(PyN),
    py_free(Runtime).

%% rl_info(+Name, -Info): Get information about a registered RL agent.
%%   Info: dict with name, env_id, algorithm, device, obs_shape, etc.
rl_info(Name, Info) :-
    ( rl_registry(Name, AgentHandle) -> true
    ; throw(error(rl_agent_not_loaded(Name), rl_info/2))
    ),
    py_import("scryer_rl_runtime", Runtime),
    py_call(Runtime, "rl_info", AgentHandle, InfoHandle),
    py_to_json(InfoHandle, InfoJson),
    json_to_value(InfoJson, Info),
    py_free(InfoHandle),
    py_free(Runtime).
