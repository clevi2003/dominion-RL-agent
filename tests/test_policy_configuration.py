"""Activation/preprocessing persistence, compatibility and raw evaluation metrics."""

from copy import deepcopy
import shutil

import numpy as np
import pytest

pytest.importorskip("sb3_contrib", reason="Policy configuration requires the rl extra")
from sb3_contrib import MaskablePPO
import torch
import yaml

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import END_BUYING
from dominion_rl.env.observations import CARD_IDS, FEATURE_INDEX, encode_observation
from dominion_rl.env.preprocessing import PolicyObservation, preprocessing_config
from dominion_rl.ppo import model_preprocessing, play_episode, train_ppo


@pytest.mark.parametrize("activation", ["tanh", "relu"])
@pytest.mark.parametrize("scaling", ["raw", "fixed"])
def test_activation_and_preprocessing_survive_checkpoints_and_relocation(tmp_path, activation, scaling):
    kwargs = dict(n_steps=16, batch_size=8, n_epochs=1, observation_scaling=scaling,
                  policy_kwargs={"net_arch": [16, 16], "activation_fn": activation})
    path = train_ppo(tmp_path / "run", game_config=GameConfig(max_turns=8),
                     reward_config=RewardConfig(), ppo_kwargs=kwargs, total_timesteps=32,
                     checkpoint_freq=16, seed=42)
    metadata = yaml.safe_load((path.parent / "run.yaml").read_text())
    assert metadata["activation"] == activation
    assert metadata["ppo"]["policy_kwargs"]["activation_fn"] == activation
    assert metadata["observation_preprocessing"] == preprocessing_config(scaling)
    expected_class = torch.nn.Tanh if activation == "tanh" else torch.nn.ReLU
    for checkpoint in (path, path.parent / "checkpoints/ppo_0_steps.zip",
                       path.parent / "checkpoints/ppo_16_steps.zip"):
        loaded = MaskablePPO.load(checkpoint, device="cpu")
        assert loaded.policy.activation_fn is expected_class
        assert isinstance(loaded.policy.mlp_extractor.policy_net[1], expected_class)
        assert isinstance(loaded.policy.mlp_extractor.value_net[1], expected_class)
        assert model_preprocessing(loaded) == preprocessing_config(scaling)
    moved = tmp_path / "moved.zip"
    shutil.copyfile(path, moved)
    loaded = MaskablePPO.load(moved, device="cpu")
    env = DominionEnv(GameConfig(max_turns=8))
    try:
        first = play_episode(loaded, env, 100)
        assert first == play_episode(loaded, env, 100)
        with pytest.raises(ValueError, match="Model requires"):
            play_episode(loaded, env, 100, expected_scaling="raw" if scaling == "fixed" else "fixed")
        with pytest.raises(ValueError, match="raw DominionEnv"):
            play_episode(loaded, PolicyObservation(env, scaling), 100)
        corrupted = deepcopy(loaded.observation_preprocessing)
        corrupted["divisors"][34] = 999
        loaded.observation_preprocessing = corrupted
        with pytest.raises(ValueError, match="preprocessing"):
            model_preprocessing(loaded)
    finally:
        env.close()


def test_legacy_checkpoint_without_preprocessing_remains_raw_tanh(tmp_path):
    env = DominionEnv(GameConfig(max_turns=2))
    model = MaskablePPO("MlpPolicy", env, n_steps=2, batch_size=2, seed=4, device="cpu")
    try:
        model.save(tmp_path / "legacy.zip")
        loaded = MaskablePPO.load(tmp_path / "legacy.zip", device="cpu")
        assert not hasattr(loaded, "observation_preprocessing")
        assert loaded.policy.activation_fn is torch.nn.Tanh
        assert model_preprocessing(loaded) == preprocessing_config("raw")
        assert play_episode(loaded, env, 5)["outcome"] == "truncated"
    finally:
        model.get_env().close()


@pytest.mark.parametrize("seat", [0, 1])
def test_scaled_inference_keeps_raw_purchase_counts_and_turns(seat):
    results = []
    for mode in ("raw", "fixed"):
        env = DominionEnv(agent_player=seat)
        contract = preprocessing_config(mode)
        divisors = np.asarray(contract["divisors"], dtype=np.float32)
        first_province = []

        class ScriptedPolicy:
            observation_space = env.observation_space
            action_space = env.action_space
            observation_preprocessing = contract

            def set_random_seed(self, seed):
                pass

            def predict(self, obs, deterministic, action_masks):
                raw = encode_observation(env.game.state, seat)
                np.testing.assert_array_equal(obs, raw / divisors)
                legal = [card for card, allowed in zip(CARD_IDS, action_masks) if allowed]
                card = big_money_purchase(legal, int(raw[FEATURE_INDEX["provinces_remaining"]]))
                if card == "province" and not first_province:
                    first_province.append(int(raw[FEATURE_INDEX["agent_turns_taken"]]) + 1)
                return np.array(END_BUYING if card is None else CARD_IDS.index(card)), None

        try:
            result = play_episode(ScriptedPolicy(), env, 7)
            assert result["first_province_turn"] == first_province[0]
            raw = encode_observation(env.game.state, seat)
            assert result["purchases_province"] == int(raw[FEATURE_INDEX["owned_province"]])
            assert result["agent_vp"] == int(raw[FEATURE_INDEX["agent_vp"]])
            results.append(result)
        finally:
            env.close()
    assert results[0] == results[1]


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_fixed_scaled_training_reproduces_weights(tmp_path, activation):
    states = []
    for name in ("first", "second"):
        path = train_ppo(tmp_path / name, game_config=GameConfig(max_turns=8),
            reward_config=RewardConfig(), seed=9, total_timesteps=16, checkpoint_freq=0,
            ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1, observation_scaling="fixed",
                            policy_kwargs={"net_arch": [16, 16], "activation_fn": activation}))
        states.append(MaskablePPO.load(path, device="cpu").policy.state_dict())
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
