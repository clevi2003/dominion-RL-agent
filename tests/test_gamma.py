"""Small gamma configuration/persistence checks; run locally with the rl extra."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

pytest.importorskip("sb3_contrib", reason="Gamma integration requires the rl extra")
from sb3_contrib import MaskablePPO
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import END_BUYING
from dominion_rl.ppo import model_training_gamma, play_episode, train_ppo


@pytest.mark.parametrize("gamma", [0.99, 1.0])
@pytest.mark.parametrize("mode", ["terminal_only", "potential_based"])
def test_matching_gamma_survives_every_saved_model(tmp_path, gamma, mode):
    path = train_ppo(tmp_path / "run", game_config=GameConfig(max_turns=2),
        reward_config=RewardConfig(mode=mode, gamma=gamma), seed=42, total_timesteps=16,
        checkpoint_freq=16, ppo_kwargs=dict(gamma=gamma, n_steps=16, batch_size=8,
            n_epochs=1, policy_kwargs={"net_arch": [16, 16]}))
    metadata = yaml.safe_load((path.parent / "run.yaml").read_text())
    assert metadata["training_gamma"] == metadata["ppo"]["gamma"] == metadata["reward"]["gamma"] == gamma
    for archive in (path, *sorted((path.parent / "checkpoints").glob("*.zip"))):
        loaded = MaskablePPO.load(archive, device="cpu")
        assert loaded.gamma == model_training_gamma(loaded, metadata) == gamma
        # Legacy snapshots lack the new top-level label; native SB3 gamma remains authoritative.
        legacy_snapshot = {k: v for k, v in metadata.items() if k != "training_gamma"}
        assert model_training_gamma(loaded, legacy_snapshot) == gamma
        assert model_training_gamma(loaded) == gamma  # No adjacent YAML needed.


@pytest.mark.parametrize("field", ["training_gamma", "ppo", "reward"])
def test_stale_gamma_metadata_is_rejected(field):
    metadata = {field: 0.99 if field == "training_gamma" else {"gamma": 0.99}}
    with pytest.raises(ValueError, match="gamma disagree"):
        model_training_gamma(SimpleNamespace(gamma=1.0), metadata)


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_evaluation_defaults_to_native_model_gamma_and_labels_overrides(monkeypatch, gamma):
    env = DominionEnv()  # Environment reward config still defaults to 0.99.
    step = env.step

    def finish_on_second_decision(action):
        if env.game.state.players[0].turns_taken == 1:
            env.game.state.supply["province"] = 0
        return step(action)

    monkeypatch.setattr(env, "step", finish_on_second_decision)
    model = SimpleNamespace(gamma=gamma, observation_space=env.observation_space,
        action_space=env.action_space, set_random_seed=lambda seed: None,
        predict=lambda obs, **kwargs: (np.array(END_BUYING), None))
    try:
        result = play_episode(model, env, 42)
        assert result["outcome"] == "loss" and result["steps"] == 2
        assert result["terminated"] and not result["truncated"]
        assert result["episode_return"] == -1
        assert result["training_gamma"] == result["evaluation_return_gamma"] == gamma
        assert result["discounted_episode_return"] == pytest.approx(-gamma)
        common = play_episode(model, env, 42, return_gamma=1.0)
        assert common["training_gamma"] == gamma
        assert common["evaluation_return_gamma"] == 1.0
        assert common["discounted_episode_return"] == -1
        for key in ("outcome", "agent_vp", "opponent_vp", "steps", "episode_return"):
            assert common[key] == result[key]
    finally:
        env.close()


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_train_cli_gamma_overrides_both_yaml_discounts(monkeypatch, tmp_path, gamma):
    script = Path(__file__).resolve().parents[1] / "scripts/train_ppo.py"
    spec = importlib.util.spec_from_file_location("train_ppo_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorded = {}

    def capture(output, **kwargs):
        recorded.update(kwargs)
        return Path(output) / "model.zip"

    monkeypatch.setattr(module, "train_ppo", capture)
    monkeypatch.chdir(script.parent.parent)
    monkeypatch.setattr(sys, "argv", [str(script), "--gamma", str(gamma),
        "--mode", "potential_based", "--output", str(tmp_path / "run")])
    module.main()
    assert recorded["reward_config"].gamma == recorded["ppo_kwargs"]["gamma"] == gamma
    assert recorded["reward_config"].mode == "potential_based"
