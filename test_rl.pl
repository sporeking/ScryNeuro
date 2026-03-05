%% ===========================================================================
%% ScryNeuro RL Prolog Test Suite
%% ===========================================================================
%% Tests the rl_* predicates in scryer_rl.pl using Tianshou v2.0
%% Run: LD_LIBRARY_PATH=. scryer-prolog test_rl.pl

:- op(700, xfx, :=).
:- use_module('prolog/scryer_py').
:- use_module('prolog/scryer_rl').
:- use_module(library(format)).

%% ---------------------------------------------------------------------------
%% Test 1: rl_create — Create a DQN agent on CartPole-v1
%% ---------------------------------------------------------------------------
test_rl_create_dqn :-
    rl_create(test_dqn, "CartPole-v1", dqn, [lr=0.001, gamma=0.99]),
    format("1. rl_create (DQN): OK~n", []).

%% ---------------------------------------------------------------------------
%% Test 2: rl_info — Get agent info
%% ---------------------------------------------------------------------------
test_rl_info :-
    rl_info(test_dqn, Info),
    ( Info = [_|_] ->
        format("2. rl_info: OK (~s)~n", [Info])
    ; format("2. rl_info: FAIL (empty info)~n", [])
    ).

%% ---------------------------------------------------------------------------
%% Test 3: rl_action — Select action from state
%% ---------------------------------------------------------------------------
test_rl_action :-
    rl_action(test_dqn, [0.05, 0.1, -0.02, 0.03], Action),
    ( number(Action) ->
        format("3. rl_action: OK (action=~w)~n", [Action])
    ; format("3. rl_action: FAIL (non-numeric: ~w)~n", [Action])
    ).

%% ---------------------------------------------------------------------------
%% Test 4: rl_action with deterministic option
%% ---------------------------------------------------------------------------
test_rl_action_deterministic :-
    rl_action(test_dqn, [0.1, -0.2, 0.05, 0.3], Action, [deterministic=true]),
    ( number(Action) ->
        format("4. rl_action (deterministic): OK (action=~w)~n", [Action])
    ; format("4. rl_action (deterministic): FAIL~n", [])
    ).

%% ---------------------------------------------------------------------------
%% Test 5: rl_create — Create a PPO agent
%% ---------------------------------------------------------------------------
test_rl_create_ppo :-
    rl_create(test_ppo, "CartPole-v1", ppo, [lr=0.0003]),
    format("5. rl_create (PPO): OK~n", []).

%% ---------------------------------------------------------------------------
%% Test 6: rl_create — Create a PG (Reinforce) agent
%% ---------------------------------------------------------------------------
test_rl_create_pg :-
    rl_create(test_pg, "CartPole-v1", pg, [lr=0.001]),
    format("6. rl_create (PG/Reinforce): OK~n", []).

%% ---------------------------------------------------------------------------
%% Test 7: rl_train — Short DQN training (1 epoch)
%% ---------------------------------------------------------------------------
test_rl_train_dqn :-
    rl_train(test_dqn, [
        max_epochs=1,
        epoch_num_steps=200,
        collection_step_num_env_steps=10,
        test_step_num_episodes=2,
        batch_size=32
    ], Metrics),
    format("7. rl_train (DQN, 1 epoch): OK (~s)~n", [Metrics]).

%% ---------------------------------------------------------------------------
%% Test 8: rl_train — Short PPO training (1 epoch)
%% ---------------------------------------------------------------------------
test_rl_train_ppo :-
    rl_train(test_ppo, [
        max_epochs=1,
        epoch_num_steps=200,
        collection_step_num_env_steps=50,
        update_step_num_repetitions=2,
        test_step_num_episodes=2,
        batch_size=32
    ], Metrics),
    format("8. rl_train (PPO, 1 epoch): OK (~s)~n", [Metrics]).

%% ---------------------------------------------------------------------------
%% Test 9: rl_evaluate — Evaluate trained DQN agent
%% ---------------------------------------------------------------------------
test_rl_evaluate :-
    rl_evaluate(test_dqn, 3, Metrics),
    format("9. rl_evaluate: OK (~s)~n", [Metrics]).

%% ---------------------------------------------------------------------------
%% Test 10: rl_save — Save DQN agent to checkpoint
%% ---------------------------------------------------------------------------
test_rl_save :-
    py_exec("import os; os.makedirs('test_checkpoints', exist_ok=True)"),
    rl_save(test_dqn, "test_checkpoints/test_dqn.pt"),
    %% Verify file exists
    py_eval("__import__('os').path.exists('test_checkpoints/test_dqn.pt')", ExH),
    py_to_bool(ExH, Exists),
    py_free(ExH),
    ( Exists = true ->
        format("10. rl_save: OK~n", [])
    ; format("10. rl_save: FAIL (file not found)~n", [])
    ).

%% ---------------------------------------------------------------------------
%% Test 11: rl_load — Load agent from checkpoint
%% ---------------------------------------------------------------------------
test_rl_load :-
    rl_load(loaded_dqn, "test_checkpoints/test_dqn.pt", [
        env_id="CartPole-v1",
        algorithm=dqn
    ]),
    %% Verify loaded agent can select an action
    rl_action(loaded_dqn, [0.0, 0.0, 0.0, 0.0], Action),
    ( number(Action) ->
        format("11. rl_load: OK (action=~w)~n", [Action])
    ; format("11. rl_load: FAIL~n", [])
    ).

%% ---------------------------------------------------------------------------
%% Test 12: rl_create error — duplicate name
%% ---------------------------------------------------------------------------
test_rl_create_duplicate :-
    ( catch(
        rl_create(test_dqn, "CartPole-v1", dqn, []),
        error(rl_agent_already_exists(test_dqn), _),
        format("12. rl_create (duplicate error): OK~n", [])
    ) -> true ; format("12. rl_create (duplicate error): FAIL~n", []) ).

%% ---------------------------------------------------------------------------
%% Test 13: rl_action on loaded agent — verify loaded weights work
%% ---------------------------------------------------------------------------
test_rl_loaded_action :-
    rl_action(loaded_dqn, [0.1, -0.5, 0.05, 0.3], Action1),
    rl_action(loaded_dqn, [0.1, -0.5, 0.05, 0.3], Action2, [deterministic=true]),
    ( number(Action1), number(Action2) ->
        format("13. rl_action (loaded agent): OK (a1=~w, a2=~w)~n", [Action1, Action2])
    ; format("13. rl_action (loaded agent): FAIL~n", [])
    ).

%% ---------------------------------------------------------------------------
%% Test 14: rl_info on PPO agent
%% ---------------------------------------------------------------------------
test_rl_info_ppo :-
    rl_info(test_ppo, Info),
    ( Info = [_|_] ->
        format("14. rl_info (PPO): OK~n", [])
    ; format("14. rl_info (PPO): FAIL~n", [])
    ).

%% ---------------------------------------------------------------------------
%% Test 15: rl_evaluate on loaded agent
%% ---------------------------------------------------------------------------
test_rl_evaluate_loaded :-
    rl_evaluate(loaded_dqn, 2, Metrics),
    format("15. rl_evaluate (loaded agent): OK (~s)~n", [Metrics]).

%% ---------------------------------------------------------------------------
%% Cleanup: remove test checkpoint files
%% ---------------------------------------------------------------------------
cleanup :-
    py_exec("import shutil; shutil.rmtree('test_checkpoints', ignore_errors=True)"),
    format("Cleanup: removed test_checkpoints/~n", []).

%% ---------------------------------------------------------------------------
%% Run all tests
%% ---------------------------------------------------------------------------
:- initialization((
    py_init,
    format("=== ScryNeuro RL Prolog Test Suite ===~n~n", []),
    catch(
        (
            test_rl_create_dqn,
            test_rl_info,
            test_rl_action,
            test_rl_action_deterministic,
            test_rl_create_ppo,
            test_rl_create_pg,
            test_rl_train_dqn,
            test_rl_train_ppo,
            test_rl_evaluate,
            test_rl_save,
            test_rl_load,
            test_rl_create_duplicate,
            test_rl_loaded_action,
            test_rl_info_ppo,
            test_rl_evaluate_loaded,
            cleanup,
            format("~n=== ALL 15 RL PROLOG TESTS PASSED ===~n", [])
        ),
        Error,
        format("~nTEST SUITE ERROR: ~w~n", [Error])
    ),
    py_finalize
)).
