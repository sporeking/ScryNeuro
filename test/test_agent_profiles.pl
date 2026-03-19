%% ===========================================================================
%% ScryNeuro Agent Profile Configuration Test
%% ===========================================================================

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_agent_api').

test_profiles_flow :-
    agent_profiles(Profiles),
    format("Profiles: ~s~n", [Profiles]),

    agent_profile("default_mock", Profile),
    format("default_mock: ~s~n", [Profile]),

    agent_new_from_profile(profile_agent, "default_mock", [
        enable_experiment_log=true,
        experiment_run_id="test_profile_agent_run"
    ]),
    agent_enable_tools(profile_agent, [add]),
    agent_run(profile_agent, "tool:add {\"a\": 1, \"b\": 41}", Out),
    format("Profile agent output: ~s~n", [Out]),
    agent_close(profile_agent).

:- initialization((
    py_init,
    catch(
        (
            test_profiles_flow,
            format("=== Agent profile config test passed ===~n", [])
        ),
        Error,
        print_py_error(Error)
    ),
    py_finalize
)).
