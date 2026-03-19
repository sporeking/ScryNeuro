%% ===========================================================================
%% ScryNeuro Real LLM Agent Example (User-Facing API)
%% ===========================================================================
%%
%% User-visible flow:
%%   1) Discover providers/models/tools/skills
%%   2) Create agent with chosen model
%%   3) Enable tools and skills
%%   4) Run concise task loop
%%
%% Run:
%%   LD_LIBRARY_PATH=. scryer-prolog examples/real_llm_agent.pl

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_agent_api').

run_example(Name, Goal) :-
    catch(Goal, E, (format("[ERROR] ~s failed:~n", [Name]), print_py_error(E), fail)),
    format("[OK] ~s~n", [Name]).

setup_and_run_real_agent :-
    py_eval("__import__('importlib.util', fromlist=['util']).find_spec('openai') is not None", OpenAIInstalledH),
    py_to_bool(OpenAIInstalledH, HasOpenAI),
    py_free(OpenAIInstalledH),
    ( HasOpenAI == true -> true
    ; throw(error(missing_python_dependency('openai', 'pip install openai'), setup_and_run_real_agent/0))
    ),

    agent_providers(Providers),
    format("Providers: ~w~n", [Providers]),

    agent_profiles(Profiles),
    format("Profiles: ~s~n", [Profiles]),

    agent_profile("default", DefaultProfile),
    format("default_openai profile: ~s~n", [DefaultProfile]),

    agent_models(openai, OpenAIModels),
    format("OpenAI models: ~w~n", [OpenAIModels]),

    agent_tools(ToolsJson),
    format("Available tools: ~s~n", [ToolsJson]),

    agent_skills(SkillsJson),
    format("Available skills: ~s~n", [SkillsJson]),

    agent_new_from_profile(research_agent, "default", [
        enable_experiment_log=true,
        experiment_log_dir="logs",
        experiment_run_id="real_llm_agent_demo"
    ]),
    agent_enable_tools(research_agent, [web_fetch, read_file, write_file, shell_exec, list_dir, grep_text]),
    agent_enable_skills(research_agent, ['research-web-markdown', 'shell-safety-exec']),

    Task = "Gather one AI-related web source and produce a concise markdown report at reports/latest_news.md.",
    agent_run(research_agent, Task, Out),
    format("Agent output: ~s~n", [Out]),

    agent_trace(research_agent, Trace),
    format("Agent trace: ~s~n", [Trace]),

    agent_close(research_agent).

:- initialization((
    py_init,
    ( catch(
          setup_and_run_real_agent,
          error(missing_python_dependency(Dep, InstallHint), _),
          ( format("[SKIP] Missing Python dependency '~w'. Install with: ~w~n", [Dep, InstallHint]), true )
      ) ->
        format("=== Real LLM user-facing example passed ===~n", [])
    ; format("=== Real LLM user-facing example failed ===~n", [])
    ),
    py_finalize
)).
