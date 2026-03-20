%% ===========================================================================
%% ScryNeuro Agent Mock Regression Test
%% ===========================================================================

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_agent').

test_mock_agent :-
    agent_discover_skills(Discovered, [skills_dir="python/skills"]),
    format("Discovered skills: ~s~n", [Discovered]),

    agent_create(mock_agent, "mock-v2", [provider=mock, enable_experiment_log=true, experiment_log_dir="logs", experiment_run_id="test_agent_mock_run"]),
    agent_register_builtin_tools(mock_agent, [list_dir, shell_exec]),
    agent_load_skill(mock_agent, 'shell-safety-exec', [skills_dir="python/skills"]),
    agent_load_plugin(mock_agent, 'scryer_agent_plugins:memory_compress_plugin', [max_messages=12, keep_tail=6]),
    agent_set_skill_policy(mock_agent, [mode=manual, max_skills=1, min_score=1]),
    agent_list_skills(mock_agent, Skills),
    format("Mock skill catalog: ~s~n", [Skills]),
    agent_disable_skill(mock_agent, 'shell-safety-exec'),
    agent_enable_skill(mock_agent, 'shell-safety-exec'),

    agent_step(mock_agent, "tool:list_dir {\"path\": \".\"}", Out1),
    format("Mock tool step: ~s~n", [Out1]),

    agent_step(mock_agent, "tool:list_dir {\"path\": \".\"}", Out2),
    format("Mock list step: ~s~n", [Out2]),

    agent_run(mock_agent, "Summarize what tools are available.", Out3, [max_steps=3]),
    format("Mock run: ~s~n", [Out3]),

    agent_save_session(mock_agent, "checkpoints/test_agent_mock.json"),
    agent_unload(mock_agent).

:- initialization((
    py_init,
    catch(
        (
            test_mock_agent,
            format("=== Agent mock regression test passed ===~n", [])
        ),
        Error,
        print_py_error(Error)
    ),
    py_finalize
)).
