%% ===========================================================================
%% ScryNeuro Agent Framework Demo (Practical)
%% ===========================================================================
%%
%% Demonstrates a practical, extensible agent setup:
%%   - Built-in tools registration (web_fetch/shell/read/write/list/grep)
%%   - Skills loading from Anthropic-style directories under ./python/skills
%%   - Behavior plugin loading (memory compression hook)
%%   - Real step/run execution in mock mode (offline deterministic)
%%
%% Run:
%%   $ LD_LIBRARY_PATH=. scryer-prolog test/test_agent_demo.pl

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_agent').

run_example(Name, Goal) :-
    ( catch(Goal, E, (format("[ERROR] ~s failed:~n", [Name]), print_py_error(E), fail)) ->
        format("[OK] ~s~n", [Name])
    ; true
    ).

demo_setup_agent :-
    format("=== Setup Agent ===~n", []),
    agent_create(practical_agent, "mock-v2", [provider=mock]),
    agent_register_builtin_tools(
        practical_agent,
        [web_fetch, shell_exec, read_file, write_file, list_dir, grep_text, add, multiply, reverse]
    ),
    agent_load_skill(practical_agent, 'research-web-markdown', [skills_dir="python/skills"]),
    agent_load_skill(practical_agent, 'shell-safety-exec', [skills_dir="python/skills"]),
    agent_load_plugin(practical_agent, 'scryer_agent_plugins:memory_compress_plugin', [max_messages=10, keep_tail=6]),
    agent_list(Agents),
    format("Agents: ~s~n", [Agents]).

demo_direct_tool_calls :-
    format("~n=== Direct Tool Calls via Agent Step ===~n", []),
    agent_step(practical_agent, "tool:add {\"a\": 7, \"b\": 8}", Out1),
    format("Add result: ~s~n", [Out1]),
    agent_step(practical_agent, "tool:shell_exec {\"command\": \"pwd\"}", Out2),
    format("Shell result: ~s~n", [Out2]),
    agent_step(practical_agent, "tool:list_dir {\"path\": \".\"}", Out3),
    format("List dir result: ~s~n", [Out3]).

demo_agent_run :-
    format("~n=== Agent Run ===~n", []),
    agent_run(
        practical_agent,
        "Summarize current working setup and mention which tools are available.",
        RunOut,
        [max_steps=3]
    ),
    format("Run output: ~s~n", [RunOut]).

demo_trace :-
    format("~n=== Agent Trace ===~n", []),
    agent_trace(practical_agent, Trace),
    format("Trace: ~s~n", [Trace]).

demo_session_persistence :-
    format("~n=== Session Save/Load ===~n", []),
    agent_save_session(practical_agent, "checkpoints/agent_demo_session.json"),
    format("Saved session to checkpoints/agent_demo_session.json~n", []),
    agent_load_session(restored_agent, "checkpoints/agent_demo_session.json", [provider=mock, model_id="mock-restored"]),
    agent_step(restored_agent, "tool:add {\"a\": 1, \"b\": 2}", RestoredOut),
    format("Restored agent output: ~s~n", [RestoredOut]),
    agent_unload(restored_agent).

cleanup :-
    catch(agent_unload(practical_agent), _, true).

:- initialization((
    py_init,
    run_example("demo_setup_agent", demo_setup_agent),
    run_example("demo_direct_tool_calls", demo_direct_tool_calls),
    run_example("demo_agent_run", demo_agent_run),
    run_example("demo_trace", demo_trace),
    run_example("demo_session_persistence", demo_session_persistence),
    cleanup,
    py_finalize
)).
