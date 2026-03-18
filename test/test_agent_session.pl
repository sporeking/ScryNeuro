%% ===========================================================================
%% ScryNeuro Agent Session Persistence Test
%% ===========================================================================

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_agent').

test_session_roundtrip :-
    agent_create(sess_agent, "mock-v2", [provider=mock]),
    agent_register_builtin_tools(sess_agent, [add, list_dir]),
    agent_load_skill(sess_agent, research, [skills_dir="python/skills"]),
    agent_load_plugin(sess_agent, 'scryer_agent_plugins:memory_compress_plugin', [max_messages=8, keep_tail=4]),
    agent_step(sess_agent, "tool:add {\"a\": 9, \"b\": 10}", Out1),
    format("Initial step: ~s~n", [Out1]),

    agent_save_session(sess_agent, "checkpoints/test_agent_session.json"),
    agent_unload(sess_agent),

    agent_load_session(restored_sess_agent, "checkpoints/test_agent_session.json", [provider=mock, model_id="mock-v2"]),
    agent_step(restored_sess_agent, "tool:add {\"a\": 2, \"b\": 3}", Out2),
    format("Restored step: ~s~n", [Out2]),
    agent_unload(restored_sess_agent).

:- initialization((
    py_init,
    catch(
        (
            test_session_roundtrip,
            format("=== Agent session persistence test passed ===~n", [])
        ),
        Error,
        print_py_error(Error)
    ),
    py_finalize
)).
