"""Action CLI delegation and role isolation without training or playing games."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO

from dominion_rl.env.action_env import action_reward


def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/action.py"
    spec = importlib.util.spec_from_file_location("action_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_delegates_to_shared_ppo(monkeypatch, tmp_path):
    captured = {}
    output = tmp_path / "run"

    def train(path, **kwargs):
        captured.update(output=path, **kwargs)
        return path / "model.zip"

    monkeypatch.setattr("dominion_rl.ppo.train_ppo", train)
    monkeypatch.setattr(sys, "argv", ["action.py", "train", "--timesteps", "256",
                                    "--checkpoint-freq", "0", "--output", str(output)])
    cli().main()
    assert captured["output"] == output
    assert captured["policy_role"] == "action"
    assert captured["game_config"].version == "V3"
    assert captured["reward_config"] == action_reward()
    assert captured["ppo_kwargs"]["gamma"] == 1.0
    assert captured["total_timesteps"] == 256
    assert captured["checkpoint_freq"] == 0
    assert not output.exists()


def test_evaluation_rejects_purchasing_role_before_output(monkeypatch, tmp_path):
    output = tmp_path / "results" / "games.csv"
    monkeypatch.setattr(MaskablePPO, "load", lambda *a, **k: SimpleNamespace(policy_role="purchase"))
    monkeypatch.setattr(sys, "argv", ["action.py", "evaluate", "--model", str(tmp_path / "purchase.zip"),
                                    "--games", "1", "--output", str(output)])
    with pytest.raises(ValueError, match="Action model"):
        cli().main()
    assert not output.parent.exists()


def test_evaluation_delegates_both_seats_without_training(monkeypatch, tmp_path):
    from test_action_ppo import native_stub

    output = tmp_path / "games.csv"
    model = native_stub()
    calls = []
    monkeypatch.setattr(MaskablePPO, "load", lambda *a, **k: model)

    def play(loaded, env, seed):
        assert loaded is model
        calls.append((env.agent_player, seed))
        return dict(agent_player=env.agent_player, evaluation_seed=seed, outcome="tie")

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not start training")

    monkeypatch.setattr("dominion_rl.action_ppo.play_action_episode", play)
    monkeypatch.setattr("dominion_rl.ppo.train_ppo", forbidden)
    monkeypatch.setattr(sys, "argv", ["action.py", "evaluate", "--model", str(tmp_path / "action.zip"),
                                    "--games", "1", "--seed", "20000", "--agent-player", "0", "1",
                                    "--output", str(output)])
    cli().main()
    assert calls == [(0, 20000), (1, 20000)]
    assert len(output.read_text().splitlines()) == 3
