"""
ScryNeuro RL Runtime
====================

Reinforcement Learning runtime module for the ScryNeuro Prolog-Python bridge.
Wraps Tianshou v2.0 (https://github.com/thu-ml/tianshou) to provide RL agent
management: creation, training, inference, evaluation, and persistence.

This module mirrors the pattern of scryer_py_runtime.py's ModelRegistry
and LLMManager, providing a unified RLRegistry for managing RL agents.

Tianshou v2.0 key architectural changes (vs v0.5.x):
  - Policy and Algorithm are separated. Policy handles action computation;
    Algorithm handles training (optimizer, loss, update).
  - OptimizerFactory pattern replaces raw torch.optim instances.
  - algorithm.run_training(TrainerParams) replaces old trainer.run().
  - Collectors take algorithm, not policy.
  - policy.compute_action(obs) for single-observation inference.

Usage from Prolog (via scryer_py.pl):
    ?- rl_create(my_agent, "CartPole-v1", dqn, [hidden_sizes=[64,64]]).
    ?- rl_train(my_agent, [max_epochs=10, epoch_num_steps=1000]).
    ?- rl_action(my_agent, State, Action).
    ?- rl_evaluate(my_agent, 10, Metrics).
    ?- rl_save(my_agent, "checkpoints/my_agent.pt").
    ?- rl_load(my_agent, "checkpoints/my_agent.pt", [env_id="CartPole-v1", algorithm=dqn]).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("scryneuro.rl")


# ============================================================================
# Supported Algorithms
# ============================================================================

SUPPORTED_ALGORITHMS = {
    "dqn",
    "ppo",
    "a2c",
    "sac",
    "td3",
    "ddpg",
    "pg",  # REINFORCE / vanilla policy gradient
    "discrete_sac",
}


# ============================================================================
# RL Agent Entry
# ============================================================================


@dataclass
class RLEntry:
    """A registered RL agent with its associated components."""

    name: str
    env_id: str
    algorithm_name: str
    policy: Any  # Tianshou v2.0 Policy (lightweight, action computation only)
    algorithm: Any  # Tianshou v2.0 Algorithm (training logic + optimizer)
    env: Any  # Gymnasium env (single, for action inference / space info)
    train_envs: Any  # Tianshou VectorEnv for training
    test_envs: Any  # Tianshou VectorEnv for testing
    train_collector: Any  # Tianshou Collector
    test_collector: Any  # Tianshou Collector
    obs_space: Any  # gym.spaces.Space
    act_space: Any  # gym.spaces.Space
    device: str
    metadata: dict = field(default_factory=dict)


# ============================================================================
# Network Builders (Tianshou v2.0 API)
# ============================================================================


def _build_networks(
    algorithm: str,
    obs_space: gym.spaces.Space,
    act_space: gym.spaces.Space,
    device: str,
    hidden_sizes: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build neural networks appropriate for the given algorithm.

    Returns a dict with keys like 'net', 'actor', 'critic', etc.
    Uses Tianshou v2.0 network classes with keyword-only arguments.
    """
    from tianshou.utils.net.common import Net

    if hidden_sizes is None:
        hidden_sizes = [128, 128]

    state_shape = obs_space.shape or (obs_space.n,)  # type: ignore[union-attr]
    is_discrete = isinstance(act_space, gym.spaces.Discrete)
    action_shape = int(act_space.n) if is_discrete else act_space.shape[0]  # type: ignore[union-attr]

    result: dict[str, Any] = {}

    if algorithm in ("dqn",):
        # DQN: single Q-network
        net = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
        ).to(device)
        result["net"] = net

    elif algorithm in ("pg", "a2c", "ppo"):
        # Actor-Critic for discrete or continuous
        if is_discrete:
            from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic

            net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
            actor = DiscreteActor(preprocess_net=net_a, action_shape=action_shape).to(
                device
            )

            net_c = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
            critic = DiscreteCritic(preprocess_net=net_c).to(device)
        else:
            from tianshou.utils.net.continuous import (
                ContinuousActorProbabilistic,
                ContinuousCritic,
            )

            net_a = Net(
                state_shape=state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh
            )
            actor = ContinuousActorProbabilistic(
                preprocess_net=net_a, action_shape=action_shape, unbounded=True
            ).to(device)

            net_c = Net(
                state_shape=state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh
            )
            critic = ContinuousCritic(preprocess_net=net_c).to(device)

        result["actor"] = actor
        result["critic"] = critic

    elif algorithm in ("sac", "discrete_sac"):
        if algorithm == "discrete_sac" or is_discrete:
            from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic

            # Discrete SAC: actor + twin critics
            net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
            actor = DiscreteActor(preprocess_net=net_a, action_shape=action_shape).to(
                device
            )

            net_c1 = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
            critic1 = DiscreteCritic(preprocess_net=net_c1).to(device)

            net_c2 = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
            critic2 = DiscreteCritic(preprocess_net=net_c2).to(device)

            result["actor"] = actor
            result["critic1"] = critic1
            result["critic2"] = critic2
        else:
            from tianshou.utils.net.continuous import (
                ContinuousActorProbabilistic,
                ContinuousCritic,
            )

            # Continuous SAC: probabilistic actor + twin critics
            net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
            actor = ContinuousActorProbabilistic(
                preprocess_net=net_a,
                action_shape=action_shape,
                unbounded=True,
            ).to(device)

            net_c1 = Net(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
            )
            critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)

            net_c2 = Net(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
            )
            critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)

            result["actor"] = actor
            result["critic1"] = critic1
            result["critic2"] = critic2

    elif algorithm in ("ddpg", "td3"):
        from tianshou.utils.net.continuous import (
            ContinuousActorDeterministic,
            ContinuousCritic,
        )

        # DDPG / TD3: continuous only
        if is_discrete:
            raise ValueError(
                f"Algorithm '{algorithm}' does not support discrete action spaces"
            )

        net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
        actor = ContinuousActorDeterministic(
            preprocess_net=net_a,
            action_shape=action_shape,
            action_space=act_space,
        ).to(device)

        net_c1 = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
        )
        critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)

        result["actor"] = actor
        result["critic1"] = critic1

        if algorithm == "td3":
            net_c2 = Net(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
            )
            critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)
            result["critic2"] = critic2

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return result


# ============================================================================
# Policy & Algorithm Builders (Tianshou v2.0 API)
# ============================================================================


def _build_policy_and_algorithm(
    algorithm: str,
    networks: dict[str, Any],
    obs_space: gym.spaces.Space,
    act_space: gym.spaces.Space,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Build a Tianshou v2.0 Policy and Algorithm from pre-built networks.

    In v2.0, Policy is lightweight (action computation only) and Algorithm
    wraps Policy + optimizer + training logic.

    Returns:
        (policy, algorithm_obj) tuple.
    """
    from tianshou.algorithm.optim import AdamOptimizerFactory

    is_discrete = isinstance(act_space, gym.spaces.Discrete)
    gamma = kwargs.get("gamma", 0.99)
    lr = kwargs.get("lr", 1e-3)
    optim = AdamOptimizerFactory(lr=lr)

    if algorithm == "dqn":
        from tianshou.algorithm import DQN
        from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy

        policy = DiscreteQLearningPolicy(
            model=networks["net"],
            action_space=act_space,
            eps_training=kwargs.get("eps_training", 0.1),
            eps_inference=kwargs.get("eps_inference", 0.05),
        )
        algo = DQN(
            policy=policy,
            optim=optim,
            gamma=gamma,
            n_step_return_horizon=kwargs.get("n_step_return_horizon", 3),
            target_update_freq=kwargs.get("target_update_freq", 320),
        )
        return policy, algo

    elif algorithm == "pg":
        from tianshou.algorithm import Reinforce
        from tianshou.algorithm.modelfree.reinforce import (
            DiscreteActorPolicy,
            ProbabilisticActorPolicy,
        )

        if is_discrete:
            policy = DiscreteActorPolicy(
                actor=networks["actor"],
                action_space=act_space,
            )
        else:
            policy = ProbabilisticActorPolicy(
                actor=networks["actor"],
                dist_fn=_get_dist_fn(),
                action_scaling=True,
                action_bound_method="clip",
                action_space=act_space,
            )
        algo = Reinforce(
            policy=policy,
            optim=optim,
            gamma=gamma,
        )
        return policy, algo

    elif algorithm == "a2c":
        from tianshou.algorithm import A2C
        from tianshou.algorithm.modelfree.reinforce import (
            DiscreteActorPolicy,
            ProbabilisticActorPolicy,
        )

        if is_discrete:
            policy = DiscreteActorPolicy(
                actor=networks["actor"],
                action_space=act_space,
            )
        else:
            policy = ProbabilisticActorPolicy(
                actor=networks["actor"],
                dist_fn=_get_dist_fn(),
                action_scaling=True,
                action_bound_method="clip",
                action_space=act_space,
            )
        algo = A2C(
            policy=policy,
            critic=networks["critic"],
            optim=optim,
            gamma=gamma,
            vf_coef=kwargs.get("vf_coef", 0.5),
            ent_coef=kwargs.get("ent_coef", 0.01),
        )
        return policy, algo

    elif algorithm == "ppo":
        from tianshou.algorithm import PPO
        from tianshou.algorithm.modelfree.reinforce import (
            DiscreteActorPolicy,
            ProbabilisticActorPolicy,
        )

        if is_discrete:
            policy = DiscreteActorPolicy(
                actor=networks["actor"],
                action_space=act_space,
            )
        else:
            policy = ProbabilisticActorPolicy(
                actor=networks["actor"],
                dist_fn=_get_dist_fn(),
                action_scaling=True,
                action_bound_method="clip",
                action_space=act_space,
            )
        algo = PPO(
            policy=policy,
            critic=networks["critic"],
            optim=optim,
            gamma=gamma,
            eps_clip=kwargs.get("eps_clip", 0.2),
            vf_coef=kwargs.get("vf_coef", 0.25),
            ent_coef=kwargs.get("ent_coef", 0.0),
            max_grad_norm=kwargs.get("max_grad_norm", 0.5),
            value_clip=kwargs.get("value_clip", True),
            advantage_normalization=kwargs.get("advantage_normalization", False),
            recompute_advantage=kwargs.get("recompute_advantage", True),
        )
        return policy, algo

    elif algorithm in ("sac", "discrete_sac"):
        if algorithm == "discrete_sac" or is_discrete:
            from tianshou.algorithm import DiscreteSAC
            from tianshou.algorithm.modelfree.discrete_sac import DiscreteSACPolicy

            policy = DiscreteSACPolicy(
                actor=networks["actor"],
                action_space=act_space,
            )
            algo = DiscreteSAC(
                policy=policy,
                policy_optim=optim,
                critic=networks["critic1"],
                critic_optim=AdamOptimizerFactory(lr=lr),
                critic2=networks["critic2"],
                critic2_optim=AdamOptimizerFactory(lr=lr),
                tau=kwargs.get("tau", 0.005),
                gamma=gamma,
                alpha=kwargs.get("alpha", 0.05),
            )
            return policy, algo
        else:
            from tianshou.algorithm import SAC
            from tianshou.algorithm.modelfree.sac import SACPolicy

            policy = SACPolicy(
                actor=networks["actor"],
                action_space=act_space,
            )
            algo = SAC(
                policy=policy,
                policy_optim=optim,
                critic=networks["critic1"],
                critic_optim=AdamOptimizerFactory(lr=lr),
                critic2=networks["critic2"],
                critic2_optim=AdamOptimizerFactory(lr=lr),
                tau=kwargs.get("tau", 0.005),
                gamma=gamma,
            )
            return policy, algo

    elif algorithm == "ddpg":
        from tianshou.algorithm import DDPG
        from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy

        policy = ContinuousDeterministicPolicy(
            actor=networks["actor"],
            action_space=act_space,
            exploration_noise=_get_exploration_noise(act_space, kwargs),
        )
        algo = DDPG(
            policy=policy,
            policy_optim=optim,
            critic=networks["critic1"],
            critic_optim=AdamOptimizerFactory(lr=lr),
            tau=kwargs.get("tau", 0.005),
            gamma=gamma,
        )
        return policy, algo

    elif algorithm == "td3":
        from tianshou.algorithm import TD3
        from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy

        policy = ContinuousDeterministicPolicy(
            actor=networks["actor"],
            action_space=act_space,
            exploration_noise=_get_exploration_noise(act_space, kwargs),
        )
        algo = TD3(
            policy=policy,
            policy_optim=optim,
            critic=networks["critic1"],
            critic_optim=AdamOptimizerFactory(lr=lr),
            critic2=networks["critic2"],
            critic2_optim=AdamOptimizerFactory(lr=lr),
            tau=kwargs.get("tau", 0.005),
            gamma=gamma,
        )
        return policy, algo

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def _get_dist_fn() -> Any:
    """Get the distribution function for on-policy continuous algorithms.

    For Tianshou v2.0, continuous policies use ProbabilisticActorPolicy
    which expects a dist_fn that takes a (loc, scale) tuple and returns
    a torch.distributions.Distribution.
    """
    from torch.distributions import Distribution, Independent, Normal

    def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    return dist_fn


def _get_exploration_noise(act_space: gym.spaces.Space, kwargs: dict) -> Any:
    """Get exploration noise for off-policy continuous algorithms."""
    from tianshou.exploration import GaussianNoise

    sigma = kwargs.get("exploration_noise_sigma", 0.1)
    return GaussianNoise(sigma=sigma)


# ============================================================================
# Environment Builders
# ============================================================================


def _make_envs(
    env_id: str,
    n_train: int = 8,
    n_test: int = 4,
) -> tuple[Any, Any, Any]:
    """Create training and testing vectorized environments.

    Returns:
        (single_env, train_envs, test_envs)
    """
    from tianshou.env import DummyVectorEnv

    single_env = gym.make(env_id)

    train_envs = DummyVectorEnv([lambda: gym.make(env_id) for _ in range(n_train)])
    test_envs = DummyVectorEnv([lambda: gym.make(env_id) for _ in range(n_test)])

    return single_env, train_envs, test_envs


# ============================================================================
# RL Registry
# ============================================================================


class RLRegistry:
    """Global registry for RL agents.

    Manages the full lifecycle of Tianshou v2.0-based RL agents:
    creation, training, inference, evaluation, and persistence.
    """

    def __init__(self) -> None:
        self._agents: dict[str, RLEntry] = {}

    def create(
        self,
        name: str,
        env_id: str,
        algorithm: str,
        device: str = "auto",
        **kwargs: Any,
    ) -> RLEntry:
        """Create a new RL agent.

        Args:
            name: Unique identifier for the agent.
            env_id: Gymnasium environment ID (e.g., "CartPole-v1").
            algorithm: RL algorithm name (e.g., "dqn", "ppo", "sac").
            device: Compute device ("auto", "cpu", "cuda").
            **kwargs: Algorithm-specific options:
                - hidden_sizes: list[int] -- network hidden layer sizes (default [128,128])
                - lr: float -- learning rate (default 1e-3)
                - gamma: float -- discount factor (default 0.99)
                - n_train_envs: int -- number of training envs (default 8)
                - n_test_envs: int -- number of test envs (default 4)
                - buffer_size: int -- replay buffer size (default 20000)
                - Plus algorithm-specific params (eps_clip, tau, alpha, etc.)

        Returns:
            The registered RLEntry.
        """
        if name in self._agents:
            raise ValueError(f"RL agent '{name}' already exists")

        algorithm = algorithm.lower()
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {sorted(SUPPORTED_ALGORITHMS)}"
            )

        # Resolve device
        if device == "auto":
            from scryer_py_runtime import get_device

            device = get_device("auto")

        # Create environments
        n_train = kwargs.pop("n_train_envs", 8)
        n_test = kwargs.pop("n_test_envs", 4)
        single_env, train_envs, test_envs = _make_envs(env_id, n_train, n_test)

        obs_space = single_env.observation_space
        act_space = single_env.action_space

        # Build networks
        hidden_sizes_raw = kwargs.pop("hidden_sizes", None)
        if isinstance(hidden_sizes_raw, str):
            hidden_sizes = json.loads(hidden_sizes_raw)
        elif isinstance(hidden_sizes_raw, list):
            hidden_sizes = hidden_sizes_raw
        else:
            hidden_sizes = None

        networks = _build_networks(
            algorithm,
            obs_space,
            act_space,
            device,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

        # Build policy and algorithm (v2.0: separated)
        policy, algo = _build_policy_and_algorithm(
            algorithm, networks, obs_space, act_space, **kwargs
        )

        # Create collectors (v2.0: collectors take algorithm, not policy)
        from tianshou.data import Collector, CollectStats, VectorReplayBuffer

        buffer_size = kwargs.pop("buffer_size", 20000)

        train_buffer = VectorReplayBuffer(total_size=buffer_size, buffer_num=n_train)

        train_collector = Collector[CollectStats](
            algo, train_envs, train_buffer, exploration_noise=True
        )
        test_collector = Collector[CollectStats](algo, test_envs)

        entry = RLEntry(
            name=name,
            env_id=env_id,
            algorithm_name=algorithm,
            policy=policy,
            algorithm=algo,
            env=single_env,
            train_envs=train_envs,
            test_envs=test_envs,
            train_collector=train_collector,
            test_collector=test_collector,
            obs_space=obs_space,
            act_space=act_space,
            device=device,
            metadata=kwargs,
        )
        self._agents[name] = entry
        logger.info(
            f"Created RL agent '{name}' (env={env_id}, algo={algorithm}, device={device})"
        )
        return entry

    def load(
        self,
        name: str,
        path: str,
        env_id: str,
        algorithm: str,
        device: str = "auto",
        **kwargs: Any,
    ) -> RLEntry:
        """Load an RL agent from a checkpoint.

        First creates a fresh agent with the same architecture, then loads
        the saved state_dict.

        Args:
            name: Unique identifier.
            path: Path to the checkpoint file.
            env_id: Gymnasium environment ID (must match the saved agent).
            algorithm: Algorithm name (must match the saved agent).
            device: Compute device.
            **kwargs: Same options as create().

        Returns:
            The registered RLEntry.
        """
        entry = self.create(name, env_id, algorithm, device=device, **kwargs)
        checkpoint = torch.load(path, map_location=entry.device, weights_only=False)
        entry.algorithm.load_state_dict(checkpoint)
        logger.info(f"Loaded RL agent '{name}' from {path}")
        return entry

    def save(self, name: str, path: str) -> None:
        """Save an RL agent's algorithm state_dict to disk.

        Args:
            name: Agent identifier.
            path: Output file path.
        """
        entry = self.get(name)
        torch.save(entry.algorithm.state_dict(), path)
        logger.info(f"Saved RL agent '{name}' to {path}")

    def get(self, name: str) -> RLEntry:
        """Get a registered RL agent by name."""
        if name not in self._agents:
            raise KeyError(
                f"RL agent '{name}' not found. Available: {list(self._agents)}"
            )
        return self._agents[name]

    def action(
        self,
        name: str,
        state: Any,
        deterministic: bool = False,
    ) -> Any:
        """Select an action given an observation.

        Uses Tianshou v2.0's policy.compute_action() for single-observation
        inference.

        Args:
            name: Agent identifier.
            state: Observation -- can be a list, numpy array, or tensor.
            deterministic: If True, use greedy/deterministic action.

        Returns:
            The selected action (as a Python scalar or list).
        """
        entry = self.get(name)
        policy = entry.policy

        # Normalize state to numpy
        if isinstance(state, list):
            obs = np.array(state, dtype=np.float32)
        elif isinstance(state, torch.Tensor):
            obs = state.cpu().numpy()
        else:
            obs = np.asarray(state, dtype=np.float32)

        # v2.0: policy.compute_action(obs) for single observation inference
        act = policy.compute_action(obs)

        # Convert to Python native type
        if isinstance(act, np.ndarray):
            return act.tolist()
        elif isinstance(act, (np.integer, np.floating)):
            return act.item()
        return int(act) if isinstance(act, (int, float)) else act

    def train(
        self,
        name: str,
        **kwargs: Any,
    ) -> dict:
        """Train an RL agent.

        This is synchronous -- blocks until training completes.
        Uses Tianshou v2.0's algorithm.run_training(TrainerParams).

        Args:
            name: Agent identifier.
            **kwargs: Training options:
                - max_epochs: int -- number of training epochs (default 5)
                - epoch_num_steps: int -- steps per epoch (default 1000)
                - collection_step_num_env_steps: int -- env steps per collect (default 10)
                - test_step_num_episodes: int -- episodes per test (default 10)
                - batch_size: int -- training batch size (default 64)
                - update_step_num_gradient_steps_per_sample: float (off-policy, default 1/10)
                - update_step_num_repetitions: int (on-policy, default 2)

        Returns:
            dict with training statistics.
        """
        entry = self.get(name)
        algo_name = entry.algorithm_name

        max_epochs = int(kwargs.get("max_epochs", 5))
        epoch_num_steps = int(kwargs.get("epoch_num_steps", 1000))
        collection_step_num_env_steps = int(
            kwargs.get("collection_step_num_env_steps", 10)
        )
        test_step_num_episodes = int(kwargs.get("test_step_num_episodes", 10))
        batch_size = int(kwargs.get("batch_size", 64))

        is_on_policy = algo_name in ("pg", "a2c", "ppo")

        if is_on_policy:
            from tianshou.trainer import OnPolicyTrainerParams

            update_step_num_repetitions = int(
                kwargs.get("update_step_num_repetitions", 2)
            )
            trainer_params = OnPolicyTrainerParams(
                training_collector=entry.train_collector,
                test_collector=entry.test_collector,
                max_epochs=max_epochs,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collection_step_num_env_steps,
                update_step_num_repetitions=update_step_num_repetitions,
                test_step_num_episodes=test_step_num_episodes,
                batch_size=batch_size,
            )
        else:
            from tianshou.trainer import OffPolicyTrainerParams

            update_step_num_gradient_steps_per_sample = float(
                kwargs.get("update_step_num_gradient_steps_per_sample", 1 / 10)
            )
            trainer_params = OffPolicyTrainerParams(
                training_collector=entry.train_collector,
                test_collector=entry.test_collector,
                max_epochs=max_epochs,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collection_step_num_env_steps,
                update_step_num_gradient_steps_per_sample=update_step_num_gradient_steps_per_sample,
                test_step_num_episodes=test_step_num_episodes,
                batch_size=batch_size,
            )

        # Run training (v2.0: algorithm.run_training)
        result = entry.algorithm.run_training(trainer_params)

        # Extract stats from the training result
        stats = _extract_training_stats(result)

        logger.info(
            f"Training complete for '{name}': "
            f"best_reward={stats.get('best_reward', 'N/A')}"
        )
        return stats

    def evaluate(self, name: str, n_episodes: int = 10) -> dict:
        """Evaluate an RL agent.

        Args:
            name: Agent identifier.
            n_episodes: Number of episodes to evaluate.

        Returns:
            dict with evaluation metrics (mean_reward, std_reward, n_episodes).
        """
        entry = self.get(name)

        result = entry.test_collector.collect(n_episode=n_episodes, reset_before_collect=True)

        metrics = _extract_eval_stats(result, n_episodes)

        logger.info(
            f"Evaluation for '{name}': "
            f"reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}"
        )
        return metrics

    def info(self, name: str) -> dict:
        """Get information about a registered RL agent.

        Args:
            name: Agent identifier.

        Returns:
            dict with agent metadata.
        """
        entry = self.get(name)
        is_discrete = isinstance(entry.act_space, gym.spaces.Discrete)

        return {
            "name": entry.name,
            "env_id": entry.env_id,
            "algorithm": entry.algorithm_name,
            "device": entry.device,
            "obs_shape": list(entry.obs_space.shape)
            if hasattr(entry.obs_space, "shape")
            else [],
            "action_type": "discrete" if is_discrete else "continuous",
            "action_size": int(entry.act_space.n)
            if is_discrete
            else list(entry.act_space.shape),  # type: ignore[union-attr]
        }

    def unload(self, name: str) -> None:
        """Unload an RL agent and free resources."""
        if name in self._agents:
            entry = self._agents[name]
            # Close environments
            try:
                entry.env.close()
                entry.train_envs.close()
                entry.test_envs.close()
            except Exception:
                pass
            del self._agents[name]
            logger.info(f"Unloaded RL agent '{name}'")

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())


# ============================================================================
# Stat Extraction Helpers
# ============================================================================


def _extract_training_stats(result: Any) -> dict:
    """Extract training statistics from Tianshou v2.0 training result.

    The result object structure may vary; we extract what's available.
    """
    stats: dict[str, Any] = {}

    # Try various attribute patterns for v2.0 result objects
    if hasattr(result, "best_reward"):
        stats["best_reward"] = float(result.best_reward)
    if hasattr(result, "best_reward_std"):
        stats["best_reward_std"] = float(result.best_reward_std)
    if hasattr(result, "duration"):
        stats["duration"] = str(result.duration)

    # If result is dict-like (fallback)
    if isinstance(result, dict):
        stats["best_reward"] = float(result.get("best_reward", 0.0))
        stats["best_reward_std"] = float(result.get("best_reward_std", 0.0))
        stats["duration"] = str(result.get("duration", ""))

    return stats


def _extract_eval_stats(result: Any, n_episodes: int) -> dict:
    """Extract evaluation statistics from Tianshou v2.0 CollectStats."""
    metrics: dict[str, Any] = {
        "n_episodes": n_episodes,
    }

    # v2.0 CollectStats has returns_stat / lens_stat attributes
    if hasattr(result, "returns_stat"):
        returns_stat = result.returns_stat
        metrics["mean_reward"] = float(returns_stat.mean)
        metrics["std_reward"] = float(returns_stat.std)
    elif hasattr(result, "rews"):
        returns = result["rews"] if isinstance(result, dict) else result.rews
        metrics["mean_reward"] = float(np.mean(returns))
        metrics["std_reward"] = float(np.std(returns))
    else:
        metrics["mean_reward"] = 0.0
        metrics["std_reward"] = 0.0

    if hasattr(result, "lens_stat"):
        lens_stat = result.lens_stat
        metrics["mean_length"] = float(lens_stat.mean)
    elif hasattr(result, "lens"):
        lens = result["lens"] if isinstance(result, dict) else result.lens
        metrics["mean_length"] = float(np.mean(lens))

    return metrics


# ============================================================================
# Global Instance & Module-level API (called from Rust FFI via spy_call)
# ============================================================================

_rl_registry = RLRegistry()


def _get_rl_registry() -> RLRegistry:
    """Return the global RLRegistry instance."""
    return _rl_registry


def rl_create(env_id: str, kwargs: Optional[dict] = None) -> Any:
    """Create an RL agent.

    Called from Prolog: py_call(Runtime, "rl_create", EnvIdHandle, KwargsHandle, Result).

    Args:
        env_id: Gymnasium environment ID (e.g., "CartPole-v1").
        kwargs: Dict with options: name, algorithm, device, hidden_sizes, etc.

    Returns:
        The RLEntry object (stored as handle on Prolog side).
    """
    if kwargs is None:
        kwargs = {}
    name = kwargs.pop("name", env_id)
    algorithm = kwargs.pop("algorithm", "dqn")
    device = kwargs.pop("device", "auto")
    registry = _get_rl_registry()
    entry = registry.create(name, env_id, algorithm, device=device, **kwargs)
    return entry


def rl_load(path: str, kwargs: Optional[dict] = None) -> Any:
    """Load an RL agent from a checkpoint.

    Args:
        path: Path to checkpoint file.
        kwargs: Dict with options: name, env_id, algorithm, device, etc.

    Returns:
        The RLEntry object.
    """
    if kwargs is None:
        kwargs = {}
    env_id = kwargs.pop("env_id", "")
    if not env_id:
        raise ValueError("rl_load requires 'env_id' in options")
    name = kwargs.pop("name", env_id)
    algorithm = kwargs.pop("algorithm", "dqn")
    device = kwargs.pop("device", "auto")
    registry = _get_rl_registry()
    entry = registry.load(name, path, env_id, algorithm, device=device, **kwargs)
    return entry


def rl_save(rl_entry: Any, path: str) -> None:
    """Save an RL agent to a checkpoint.

    Args:
        rl_entry: RLEntry object from rl_create/rl_load.
        path: Output file path.
    """
    registry = _get_rl_registry()
    registry.save(rl_entry.name, path)


def rl_action(rl_entry: Any, state: Any, kwargs: Optional[dict] = None) -> Any:
    """Select an action for a given state.

    Args:
        rl_entry: RLEntry object.
        state: Observation data (list, array, or tensor).
        kwargs: Options (e.g., deterministic=true).

    Returns:
        The selected action.
    """
    if kwargs is None:
        kwargs = {}
    deterministic = kwargs.get("deterministic", False)
    if isinstance(deterministic, str):
        deterministic = deterministic.lower() in ("true", "1", "yes")
    registry = _get_rl_registry()
    return registry.action(rl_entry.name, state, deterministic=deterministic)


def rl_train(rl_entry: Any, kwargs: Optional[dict] = None) -> dict:
    """Train an RL agent.

    Args:
        rl_entry: RLEntry object.
        kwargs: Training options (n_epoch, step_per_epoch, etc.).

    Returns:
        Dict with training statistics.
    """
    if kwargs is None:
        kwargs = {}
    registry = _get_rl_registry()
    return registry.train(rl_entry.name, **kwargs)


def rl_evaluate(rl_entry: Any, n_episodes: int = 10) -> dict:
    """Evaluate an RL agent.

    Args:
        rl_entry: RLEntry object.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with evaluation metrics.
    """
    registry = _get_rl_registry()
    return registry.evaluate(rl_entry.name, n_episodes=n_episodes)


def rl_info(rl_entry: Any) -> dict:
    """Get info about an RL agent.

    Args:
        rl_entry: RLEntry object.

    Returns:
        Dict with agent metadata.
    """
    registry = _get_rl_registry()
    return registry.info(rl_entry.name)
