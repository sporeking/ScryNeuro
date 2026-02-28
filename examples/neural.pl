%% ===========================================================================
%% ScryNeuro Neural Predicate Examples
%% ===========================================================================
%%
%% Demonstrates neuro-symbolic AI patterns with ScryNeuro.
%%
%% Prerequisites:
%%   - Build: cargo build --release
%%   - Copy: cp target/release/libscryneuro.so ./
%%   - Python deps: pip install torch numpy
%%   - For LLM examples: pip install openai
%%
%% Run:
%%   $ LD_LIBRARY_PATH=. scryer-prolog examples/neural.pl

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_nn').
:- use_module('../prolog/scryer_llm').

%% ===========================================================================
%% Example 1: MNIST Digit Classification (Neural Predicate)
%% ===========================================================================
%%
%% A neural predicate that classifies handwritten digits.
%% This demonstrates the core neuro-symbolic pattern:
%%   logical rules + neural perception.
%%
%%   digit(Image, Class) :-
%%       nn_predict(mnist, Image, Output),
%%       argmax(Output, Class).

%% Simulated: define a simple neural network in Python
setup_mnist_demo :-
    py_init,
    %% Define a mock classifier in Python for demonstration
    py_exec_lines([
        "import sys, random",
        "class MockMNIST:",
        "    def __call__(self, x):",
        "        probs = [random.random() for _ in range(10)]",
        "        total = sum(probs)",
        "        return [p / total for p in probs]",
        "",
        "_mock_mnist = MockMNIST()"
    ]).

%% Neural predicate: classify a digit image
digit(Image, Class) :-
    py_import("__main__", Main),
    py_getattr(Main, "_mock_mnist", Model),
    py_invoke(Model, Image, Output),
    %% Convert output to list and find argmax
    py_to_json(Output, JsonStr),
    py_from_json(JsonStr, Probs),
    argmax_py(Probs, Class),
    py_free(Output),
    py_free(Model),
    py_free(Main).

%% Find argmax using Python
argmax_py(ProbsHandle, Class) :-
    py_eval("lambda probs: max(range(len(probs)), key=lambda i: probs[i])", ArgmaxFn),
    py_invoke(ArgmaxFn, ProbsHandle, ClassHandle),
    py_to_int(ClassHandle, Class),
    py_free(ClassHandle),
    py_free(ArgmaxFn).

demo_mnist :-
    setup_mnist_demo,
    format("=== MNIST Demo (Mock) ===~n", []),

    %% Create a mock input tensor (28x28 flattened)
    py_eval("[0.0] * 784", MockImage),

    %% Classify
    digit(MockImage, Class),
    format("Predicted digit class: ~d~n", [Class]),

    py_free(MockImage).

%% ===========================================================================
%% Example 2: Neuro-Symbolic Reasoning
%% ===========================================================================
%%
%% Combine neural perception with logical reasoning.
%% Pattern: Neural networks provide base facts, Prolog reasons over them.
%%
%%   addition(Image1, Image2, Sum) :-
%%       digit(Image1, D1),
%%       digit(Image2, D2),
%%       Sum is D1 + D2.

addition(Image1, Image2, Sum) :-
    digit(Image1, D1),
    digit(Image2, D2),
    Sum is D1 + D2.

demo_addition :-
    format("=== Neuro-Symbolic Addition ===~n", []),
    py_eval("[0.0] * 784", Img1),
    py_eval("[0.0] * 784", Img2),
    addition(Img1, Img2, Sum),
    format("digit(img1) + digit(img2) = ~d~n", [Sum]),
    py_free(Img1),
    py_free(Img2).

%% ===========================================================================
%% Example 3: LLM as a Knowledge Source
%% ===========================================================================
%%
%% Use an LLM to answer questions that can't be encoded as rules.
%% The Prolog program provides structure; the LLM provides knowledge.
%%
%%   answer(Question, Answer) :-
%%       llm_generate(gpt, Question, Answer).

%% NOTE: This requires an OpenAI API key in the environment.
%% Skip if not available.

demo_llm :-
    format("=== LLM Integration (requires API key) ===~n", []),
    ( catch(
        (
            llm_load(demo_llm, "gpt-4", [provider=openai]),
            llm_generate(demo_llm, "What is the capital of France? Reply in one word.", Response),
            format("LLM response: ~s~n", [Response])
        ),
        _Error,
        format("Skipped: LLM not available (no API key or network)~n", [])
    ) -> true ; true ),
    nl.

%% ===========================================================================
%% Example 4: Deep RL Agent as a Predicate
%% ===========================================================================
%%
%% A reinforcement learning agent exposed as a Prolog predicate.
%% The agent chooses actions given states.
%%
%%   agent_action(State, Action) :-
%%       nn_predict(rl_agent, State, QValues),
%%       argmax(QValues, Action).

setup_rl_demo :-
    py_exec_lines([
        "import random",
        "class MockRLAgent:",
        "    def __call__(self, state):",
        "        return [random.random() for _ in range(4)]",
        "",
        "_mock_rl_agent = MockRLAgent()"
    ]).

:- dynamic(action_name/2).
action_name(0, up).
action_name(1, down).
action_name(2, left).
action_name(3, right).

agent_action(State, ActionName) :-
    py_import("__main__", Main),
    py_getattr(Main, "_mock_rl_agent", Agent),
    py_invoke(Agent, State, QValues),
    py_to_json(QValues, JsonStr),
    py_from_json(JsonStr, QList),
    argmax_py(QList, ActionIdx),
    action_name(ActionIdx, ActionName),
    py_free(QValues),
    py_free(QList),
    py_free(Agent),
    py_free(Main).

demo_rl :-
    setup_rl_demo,
    format("=== RL Agent Demo (Mock) ===~n", []),

    %% Create a mock state
    py_eval("[1.0, 0.5, -0.3, 0.8]", State),
    agent_action(State, Action),
    format("Agent chose action: ~w~n", [Action]),
    py_free(State).

%% ===========================================================================
%% Example 5: Probabilistic Neural Predicate (DeepProbLog-style)
%% ===========================================================================
%%
%% Neural network output as probabilities for logical facts.
%% Pattern from DeepProbLog: nn(Network, [Input], Output, Domain).
%%
%% We implement a simplified version where the neural network
%% assigns probabilities to domain values.

%% Simplified probabilistic query:
%% "What is the most likely classification?"
nn_classify(ModelHandle, Input, BestClass, Confidence) :-
    py_invoke(ModelHandle, Input, Output),
    py_to_json(Output, JsonStr),
    py_from_json(JsonStr, ProbList),
    %% Find max probability
    py_eval("lambda probs: (max(range(len(probs)), key=lambda i: probs[i]), max(probs))", MaxFn),
    py_invoke(MaxFn, ProbList, ResultTuple),
    py_eval("lambda t: t[0]", GetIdx),
    py_eval("lambda t: t[1]", GetProb),
    py_invoke(GetIdx, ResultTuple, IdxH),
    py_invoke(GetProb, ResultTuple, ProbH),
    py_to_int(IdxH, BestClass),
    py_to_float(ProbH, Confidence),
    py_free(IdxH), py_free(ProbH),
    py_free(GetIdx), py_free(GetProb),
    py_free(MaxFn), py_free(ResultTuple),
    py_free(ProbList), py_free(Output).

demo_probabilistic :-
    format("=== Probabilistic Neural Predicate ===~n", []),
    py_import("__main__", Main),
    py_getattr(Main, "_mock_mnist", Model),

    py_eval("[0.0] * 784", Input),
    nn_classify(Model, Input, Class, Conf),
    format("Best class: ~d (confidence: ~f)~n", [Class, Conf]),

    py_free(Input),
    py_free(Model),
    py_free(Main).

%% ===========================================================================
%% Run all demos
%% ===========================================================================

:- initialization((
    py_init,
    demo_mnist, nl,
    demo_addition, nl,
    demo_llm, nl,
    demo_rl, nl,
    demo_probabilistic, nl,
    format("=== All neural examples complete ===~n", []),
    py_finalize
)).
