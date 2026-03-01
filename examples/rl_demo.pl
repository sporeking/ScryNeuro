%% ===========================================================================
%% ScryNeuro Reinforcement Learning Demo
%% ===========================================================================
%%
%% Demonstrates RL agent integration using Tianshou v2.0 via ScryNeuro's
%% high-level Prolog predicates.
%%
%% Prerequisites:
%%   - Build: cargo build --release
%%   - Copy: cp target/release/libscryneuro.so ./     # Linux
%%   -       cp target/release/libscryneuro.dylib ./  # macOS
%%   - Python deps: pip install torch gymnasium tianshou
%%     (Tianshou v2.0: pip install git+https://github.com/thu-ml/tianshou.git)
%%
%% Run:
%%   $ LD_LIBRARY_PATH=. scryer-prolog examples/rl_demo.pl

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_rl').

%% ===========================================================================
%% Example 1: CartPole DQN — Create, Train, Evaluate, Save
%% ===========================================================================
%%
%% The classic CartPole-v1 environment:
%%   - Observation: [cart_pos, cart_vel, pole_angle, pole_vel] (4 floats)
%%   - Action: 0 (push left) or 1 (push right)
%%   - Goal: keep the pole balanced (max reward ~500)
%%
%% DQN (Deep Q-Network) is a natural choice for discrete action spaces.

demo_cartpole_dqn :-
    format("=== CartPole-v1 DQN Demo ===~n", []),
    format("Creating DQN agent for CartPole-v1...~n", []),

    %% Create a DQN agent with custom network sizes
    rl_create(cartpole_dqn, "CartPole-v1", dqn, [
        lr=0.001,
        gamma=0.99,
        eps_training=0.1,
        eps_inference=0.05,
        n_train_envs=8,
        n_test_envs=4,
        buffer_size=20000
    ]),
    format("Agent created successfully.~n", []),

    %% Get agent info
    rl_info(cartpole_dqn, Info),
    format("Agent info: ~s~n", [Info]),

    %% Train for a few epochs (short demo)
    format("~nTraining for 50 epochs...~n", []),
    rl_train(cartpole_dqn, [
        max_epochs=50,
        epoch_num_steps=1000,
        collection_step_num_env_steps=10,
        test_step_num_episodes=5,
        batch_size=64
    ], TrainMetrics),
    format("Training metrics: ~s~n", [TrainMetrics]),

    %% Evaluate the trained agent
    format("~nEvaluating over 10 episodes...~n", []),
    rl_evaluate(cartpole_dqn, 10, EvalMetrics),
    format("Evaluation metrics: ~s~n", [EvalMetrics]),

    %% Use the agent to select an action for a single state
    format("~nSelecting action for sample state...~n", []),
    rl_action(cartpole_dqn, [0.05, 0.1, -0.02, 0.03], Action),
    format("Selected action: ~w~n", [Action]),

    %% Save checkpoint
    format("~nSaving agent to checkpoints/cartpole_dqn.pt...~n", []),
    py_exec("import os; os.makedirs('checkpoints', exist_ok=True)"),
    rl_save(cartpole_dqn, "checkpoints/cartpole_dqn.pt"),
    format("Agent saved.~n", []),

    format("~n=== CartPole DQN Demo Complete ===~n", []).


%% ===========================================================================
%% Example 2: CartPole PPO — On-Policy Algorithm
%% ===========================================================================
%%
%% PPO (Proximal Policy Optimization) is a popular on-policy algorithm
%% that works well for many environments.

demo_cartpole_ppo :-
    format("~n=== CartPole-v1 PPO Demo ===~n", []),
    format("Creating PPO agent for CartPole-v1...~n", []),

    rl_create(cartpole_ppo, "CartPole-v1", ppo, [
        lr=0.0003,
        gamma=0.99,
        eps_clip=0.2,
        n_train_envs=8,
        n_test_envs=4
    ]),
    format("PPO agent created.~n", []),

    %% Short training
    format("Training PPO for 50 epochs...~n", []),
    rl_train(cartpole_ppo, [
        max_epochs=50,
        epoch_num_steps=1000,
        collection_step_num_env_steps=200,
        update_step_num_repetitions=4,
        test_step_num_episodes=5,
        batch_size=64
    ], Metrics),
    format("PPO training metrics: ~s~n", [Metrics]),

    %% Evaluate
    rl_evaluate(cartpole_ppo, 10, EvalMetrics),
    format("PPO evaluation: ~s~n", [EvalMetrics]),

    format("=== CartPole PPO Demo Complete ===~n", []).


%% ===========================================================================
%% Example 3: Load a Saved Agent
%% ===========================================================================
%%
%% Demonstrates loading a previously saved checkpoint.

demo_load_agent :-
    format("~n=== Load Agent Demo ===~n", []),

    %% Check if checkpoint exists
    py_eval("__import__('os').path.exists('checkpoints/cartpole_dqn.pt')", ExistsH),
    py_to_bool(ExistsH, Exists),
    py_free(ExistsH),

    ( Exists = true ->
        format("Loading saved DQN agent...~n", []),
        rl_load(loaded_dqn, "checkpoints/cartpole_dqn.pt", [
            env_id="CartPole-v1",
            algorithm=dqn
        ]),
        format("Agent loaded successfully.~n", []),

        %% Test the loaded agent
        rl_evaluate(loaded_dqn, 5, Metrics),
        format("Loaded agent evaluation: ~s~n", [Metrics]),

        %% Select an action
        rl_action(loaded_dqn, [0.0, 0.0, 0.0, 0.0], Action),
        format("Loaded agent action: ~w~n", [Action])
    ;
        format("No checkpoint found, skipping load demo.~n", [])
    ),
    format("=== Load Agent Demo Complete ===~n", []).


%% ===========================================================================
%% Example 4: Neuro-Symbolic RL — Logic + Learning
%% ===========================================================================
%%
%% This demonstrates a key neuro-symbolic pattern:
%%   - The RL agent provides raw action selection (neural)
%%   - Prolog provides safety constraints and reasoning (symbolic)
%%
%% Example: An RL agent navigating a grid where certain cells are forbidden.

:- dynamic(forbidden_cell/1).
forbidden_cell(left).    %% Don't go left off the edge
forbidden_cell(down).    %% Don't go down off the edge

:- dynamic(action_label/2).
action_label(0, push_left).
action_label(1, push_right).

%% Safe action selection: use RL but override with safety rules
safe_action(AgentName, State, SafeAction) :-
    %% Get the neural agent's preferred action
    rl_action(AgentName, State, RawAction),
    action_label(RawAction, ActionName),
    ( forbidden_cell(ActionName) ->
        %% If forbidden, choose the opposite action
        format("  RL suggested ~w (forbidden), overriding...~n", [ActionName]),
        opposite_action(RawAction, SafeAction)
    ;
        SafeAction = RawAction
    ).

opposite_action(0, 1).  %% left -> right
opposite_action(1, 0).  %% right -> left

demo_neurosymbolic :-
    format("~n=== Neuro-Symbolic RL Demo ===~n", []),
    format("Using RL agent with Prolog safety constraints.~n~n", []),

    %% Use the cartpole_dqn agent created earlier
    ( catch(
        (
            safe_action(cartpole_dqn, [0.1, -0.5, 0.05, 0.3], Action1),
            action_label(Action1, Name1),
            format("State 1 -> safe action: ~d (~w)~n", [Action1, Name1]),

            safe_action(cartpole_dqn, [-0.3, 0.2, -0.1, -0.4], Action2),
            action_label(Action2, Name2),
            format("State 2 -> safe action: ~d (~w)~n", [Action2, Name2])
        ),
        Error,
        format("Neuro-symbolic demo error: ~w~n", [Error])
    ) -> true ; true),

    format("=== Neuro-Symbolic RL Demo Complete ===~n", []).


%% ===========================================================================
%% Run all demos
%% ===========================================================================

:- initialization((
    py_init,
    catch(
        (
            demo_cartpole_dqn, nl,
            demo_cartpole_ppo, nl,
            demo_load_agent, nl,
            demo_neurosymbolic, nl,
            format("~n=== All RL demos complete ===~n", [])
        ),
        Error,
        format("RL demo error: ~w~n", [Error])
    ),
    py_finalize
)).
