"""
ScryNeuro RL Runtime Test Suite
================================

Tests the scryer_rl_runtime.py module directly (Python-level, no Rust FFI).
Uses CartPole-v1 as a simple discrete environment.

Run:  python test_rl.py          (from project root)
      python python/test_rl.py   (alternative)
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

# Ensure python/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0
_total = 0


def run_test(name: str, fn):
    """Run a test function, print OK/FAIL."""
    global _pass, _fail, _total
    _total += 1
    try:
        fn()
        _pass += 1
        print(f"  {_total:2d}. {name}: OK")
    except Exception as e:
        _fail += 1
        print(f"  {_total:2d}. {name}: FAIL -- {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import():
    """Basic import sanity check."""
    from scryer_rl_runtime import RLRegistry, RLEntry, SUPPORTED_ALGORITHMS

    assert isinstance(SUPPORTED_ALGORITHMS, set)
    assert "dqn" in SUPPORTED_ALGORITHMS
    assert "ppo" in SUPPORTED_ALGORITHMS


def test_create_dqn():
    """Create a DQN agent on CartPole-v1."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    entry = reg.create(
        "test_dqn",
        "CartPole-v1",
        "dqn",
        device="cpu",
        hidden_sizes=[64, 64],
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=1000,
    )
    assert entry.name == "test_dqn"
    assert entry.env_id == "CartPole-v1"
    assert entry.algorithm_name == "dqn"
    assert entry.device == "cpu"
    assert entry.policy is not None
    assert entry.algorithm is not None
    assert entry.train_collector is not None
    assert entry.test_collector is not None
    reg.unload("test_dqn")


def test_create_ppo():
    """Create a PPO agent on CartPole-v1."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    entry = reg.create(
        "test_ppo",
        "CartPole-v1",
        "ppo",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
    )
    assert entry.name == "test_ppo"
    assert entry.algorithm_name == "ppo"
    assert entry.policy is not None
    assert entry.algorithm is not None
    reg.unload("test_ppo")


def test_create_pg():
    """Create a Reinforce (PG) agent on CartPole-v1."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    entry = reg.create(
        "test_pg",
        "CartPole-v1",
        "pg",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
    )
    assert entry.name == "test_pg"
    assert entry.algorithm_name == "pg"
    reg.unload("test_pg")


def test_create_a2c():
    """Create an A2C agent on CartPole-v1."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    entry = reg.create(
        "test_a2c",
        "CartPole-v1",
        "a2c",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
    )
    assert entry.name == "test_a2c"
    assert entry.algorithm_name == "a2c"
    reg.unload("test_a2c")


def test_duplicate_name():
    """Creating an agent with a duplicate name should raise."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "dup",
        "CartPole-v1",
        "dqn",
        device="cpu",
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )
    try:
        reg.create(
            "dup",
            "CartPole-v1",
            "dqn",
            device="cpu",
            n_train_envs=2,
            n_test_envs=2,
            buffer_size=500,
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass  # expected
    finally:
        reg.unload("dup")


def test_unsupported_algorithm():
    """Creating an agent with an unsupported algorithm should raise."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    try:
        reg.create("bad", "CartPole-v1", "rainbow_unicorn", device="cpu")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass  # expected


def test_action():
    """Select an action given an observation."""
    import numpy as np
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "act_test",
        "CartPole-v1",
        "dqn",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )
    # CartPole obs: [cart_pos, cart_vel, pole_angle, pole_vel]
    obs = [0.0, 0.0, 0.0, 0.0]
    action = reg.action("act_test", obs)
    # CartPole is discrete: action should be 0 or 1
    assert action in (0, 1), f"Unexpected action: {action}"

    # Also test with numpy array
    obs_np = np.array([0.1, -0.2, 0.05, 0.1], dtype=np.float32)
    action2 = reg.action("act_test", obs_np)
    assert action2 in (0, 1), f"Unexpected action: {action2}"

    reg.unload("act_test")


def test_action_deterministic():
    """Deterministic action should be consistent."""
    import numpy as np
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "det_test",
        "CartPole-v1",
        "dqn",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )
    obs = [0.0, 0.0, 0.0, 0.0]
    actions = [reg.action("det_test", obs, deterministic=True) for _ in range(5)]
    # Deterministic: all actions should be the same
    assert len(set(actions)) == 1, f"Deterministic actions differ: {actions}"
    reg.unload("det_test")


def test_info():
    """Get agent info dict."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "info_test",
        "CartPole-v1",
        "dqn",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )
    info = reg.info("info_test")
    assert isinstance(info, dict)
    assert info["name"] == "info_test"
    assert info["env_id"] == "CartPole-v1"
    assert info["algorithm"] == "dqn"
    assert info["device"] == "cpu"
    assert info["action_type"] == "discrete"
    assert info["action_size"] == 2  # CartPole: left or right
    assert info["obs_shape"] == [4]  # CartPole: 4-dim observation
    reg.unload("info_test")


def test_list_agents():
    """List registered agents."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    assert reg.list_agents() == []
    reg.create(
        "a1",
        "CartPole-v1",
        "dqn",
        device="cpu",
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )
    reg.create("a2", "CartPole-v1", "ppo", device="cpu", n_train_envs=2, n_test_envs=2)
    agents = reg.list_agents()
    assert "a1" in agents
    assert "a2" in agents
    assert len(agents) == 2
    reg.unload("a1")
    reg.unload("a2")


def test_train_dqn_short():
    """Train DQN for 1 epoch (smoke test)."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "train_dqn",
        "CartPole-v1",
        "dqn",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=2000,
    )
    stats = reg.train(
        "train_dqn",
        max_epochs=1,
        epoch_num_steps=200,
        collection_step_num_env_steps=10,
        test_step_num_episodes=2,
        batch_size=32,
    )
    assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"
    reg.unload("train_dqn")


def test_train_ppo_short():
    """Train PPO for 1 epoch (smoke test)."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "train_ppo",
        "CartPole-v1",
        "ppo",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
    )
    stats = reg.train(
        "train_ppo",
        max_epochs=1,
        epoch_num_steps=200,
        collection_step_num_env_steps=10,
        test_step_num_episodes=2,
        batch_size=32,
        update_step_num_repetitions=2,
    )
    assert isinstance(stats, dict)
    reg.unload("train_ppo")


def test_evaluate():
    """Evaluate an agent over a few episodes."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "eval_test",
        "CartPole-v1",
        "dqn",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )
    metrics = reg.evaluate("eval_test", n_episodes=3)
    assert isinstance(metrics, dict)
    assert "mean_reward" in metrics
    assert "std_reward" in metrics
    assert "n_episodes" in metrics
    assert metrics["n_episodes"] == 3
    assert isinstance(metrics["mean_reward"], float)
    reg.unload("eval_test")


def test_save_load():
    """Save an agent, load it back, verify action consistency."""
    import numpy as np
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "save_test",
        "CartPole-v1",
        "dqn",
        device="cpu",
        hidden_sizes=[32, 32],
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )

    obs = [0.1, -0.2, 0.03, 0.15]
    action_before = reg.action("save_test", obs, deterministic=True)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        # Save
        reg.save("save_test", ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file not created"
        assert os.path.getsize(ckpt_path) > 0, "Checkpoint file is empty"

        reg.unload("save_test")

        # Load into a new agent
        entry = reg.load(
            "save_loaded",
            ckpt_path,
            "CartPole-v1",
            "dqn",
            device="cpu",
            hidden_sizes=[32, 32],
            n_train_envs=2,
            n_test_envs=2,
            buffer_size=500,
        )
        assert entry.name == "save_loaded"

        # Verify same deterministic action
        action_after = reg.action("save_loaded", obs, deterministic=True)
        assert action_before == action_after, (
            f"Action mismatch after load: {action_before} vs {action_after}"
        )
        reg.unload("save_loaded")
    finally:
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)


def test_unload():
    """Unload removes the agent from registry."""
    from scryer_rl_runtime import RLRegistry

    reg = RLRegistry()
    reg.create(
        "unload_test",
        "CartPole-v1",
        "dqn",
        device="cpu",
        n_train_envs=2,
        n_test_envs=2,
        buffer_size=500,
    )
    assert "unload_test" in reg.list_agents()
    reg.unload("unload_test")
    assert "unload_test" not in reg.list_agents()

    # Getting a non-existent agent should raise
    try:
        reg.get("unload_test")
        raise AssertionError("Should have raised KeyError")
    except KeyError:
        pass  # expected


def test_py_runtime_wrappers():
    """Test the module-level wrappers in scryer_rl_runtime.py."""
    from scryer_rl_runtime import rl_create, rl_action, rl_info, rl_save, rl_evaluate

    entry = rl_create(
        "CartPole-v1",
        {
            "name": "wrapper_test",
            "algorithm": "dqn",
            "device": "cpu",
            "hidden_sizes": [32, 32],
            "n_train_envs": 2,
            "n_test_envs": 2,
            "buffer_size": 500,
        },
    )
    assert entry.name == "wrapper_test"

    # rl_action
    action = rl_action(entry, [0.0, 0.0, 0.0, 0.0])
    assert action in (0, 1)

    # rl_info
    info = rl_info(entry)
    assert info["name"] == "wrapper_test"
    assert info["env_id"] == "CartPole-v1"

    # rl_evaluate
    metrics = rl_evaluate(entry, 2)
    assert isinstance(metrics["mean_reward"], float)

    # rl_save + cleanup
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt = f.name
    try:
        rl_save(entry, ckpt)
        assert os.path.getsize(ckpt) > 0
    finally:
        os.unlink(ckpt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ScryNeuro RL Runtime Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Import sanity", test_import),
        ("Create DQN agent", test_create_dqn),
        ("Create PPO agent", test_create_ppo),
        ("Create PG (Reinforce) agent", test_create_pg),
        ("Create A2C agent", test_create_a2c),
        ("Duplicate name error", test_duplicate_name),
        ("Unsupported algorithm error", test_unsupported_algorithm),
        ("Action selection", test_action),
        ("Deterministic action", test_action_deterministic),
        ("Agent info", test_info),
        ("List agents", test_list_agents),
        ("Train DQN (1 epoch)", test_train_dqn_short),
        ("Train PPO (1 epoch)", test_train_ppo_short),
        ("Evaluate agent", test_evaluate),
        ("Save / Load checkpoint", test_save_load),
        ("Unload agent", test_unload),
        ("scryer_rl_runtime wrappers", test_py_runtime_wrappers),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print()
    print("-" * 60)
    print(f"Results: {_pass} passed, {_fail} failed, {_total} total")
    if _fail == 0:
        print("=== ALL RL TESTS PASSED ===")
    else:
        print(f"*** {_fail} TEST(S) FAILED ***")
    print("-" * 60)

    sys.exit(1 if _fail > 0 else 0)
