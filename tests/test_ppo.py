"""Short integration checks; install .[dev,rl] to run these tests."""

import numpy as np
import pytest

pytest.importorskip("sb3_contrib", reason="PPO integration requires the rl extra")
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import END_BUYING
from dominion_rl.env.observations import FEATURE_INDEX, encode_observation
from dominion_rl.ppo import play_episode, train_ppo
from dominion_rl.env.preprocessing import PolicyObservation


SMALL_PPO = dict(n_steps=16, batch_size=8, n_epochs=1,
                 gamma=0.99, policy_kwargs={"net_arch": [16, 16]})


def test_initialization_and_masks_block_an_illegal_preferred_purchase():
    env = DominionEnv()
    model = MaskablePPO("MlpPolicy", env, device="cpu", seed=3, **SMALL_PPO)
    try:
        obs, _ = env.reset(seed=3)
        assert obs.shape == (41,) and obs.dtype == np.float32
        mask = env.action_masks()
        assert mask.shape == (7,) and mask.dtype == np.bool_
        assert not mask[5]  # A starting hand cannot buy a Province.
        # Make the illegal action overwhelmingly preferred without masking.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.bias[5] = 1000
        assert int(model.predict(obs, deterministic=True)[0].item()) == 5
        with pytest.raises(ValueError, match="Illegal action"):
            env.step(5)
        for _ in range(20):
            action, _ = model.predict(obs, action_masks=mask)
            assert mask[int(action.item())]
        env.step(0)  # Copper consumes the only Buy.
        np.testing.assert_array_equal(env.action_masks(), [False] * 6 + [True])
        obs = env.step(END_BUYING)[0]
        env.game.state.supply["silver"] = 0
        assert not env.action_masks()[1]
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        assert env.action_masks()[int(action.item())]
    finally:
        model.get_env().close()


@pytest.mark.parametrize("mode", ["terminal_only", "potential_based"])
@pytest.mark.parametrize("opponent", ["big-money", "mixed"])
def test_training_checkpoints_reload_and_complete_masked_games(tmp_path, mode, opponent):
    path = train_ppo(
        tmp_path / mode, game_config=GameConfig(max_turns=8),
        reward_config=RewardConfig(mode=mode), ppo_kwargs=SMALL_PPO,
        seed=42, total_timesteps=32, checkpoint_freq=16, opponent=opponent,
    )
    assert path.is_file()
    assert (path.parent / "checkpoints" / "ppo_0_steps.zip").is_file()
    assert (path.parent / "checkpoints" / "ppo_16_steps.zip").is_file()
    assert (path.parent / "checkpoints" / "ppo_32_steps.zip").is_file()
    assert (path.parent / "progress.csv").is_file()
    assert (path.parent / "episodes.monitor.csv").is_file()
    metadata = yaml.safe_load((path.parent / "run.yaml").read_text())
    assert metadata["reward"]["mode"] == mode
    assert metadata["actual_timesteps"] == 32
    exposure = metadata["by_opponent"]
    assert sum(row["decision_steps"] for row in exposure.values()) == 32
    assert sum(row["completed_episodes"] for row in exposure.values()) == metadata["completed_episodes"]
    assert sum(row["started_episodes"] for row in exposure.values()) == metadata["started_episodes"]
    assert sum(row["completed_episode_decision_steps"] for row in exposure.values()) + metadata["partial_episode_steps"] == 32
    assert sum(row["step_fraction"] for row in exposure.values()) == pytest.approx(1)
    assert metadata["opponent_distribution"] == (
        {"big-money": 0.5, "random": 0.5} if opponent == "mixed"
        else {"big-money": 1.0, "random": 0.0}
    )
    assert metadata["completed_episodes"] > 0
    assert metadata["training_seconds"] > 0
    assert metadata["steps_per_second"] > 0
    assert 0 <= metadata["truncation_rate"] <= 1
    assert metadata["ppo"]["gamma"] == metadata["reward"]["gamma"] == 0.99
    model = MaskablePPO.load(path, device="cpu")
    assert model.num_timesteps == 32
    before = {key: value.clone() for key, value in model.policy.state_dict().items()}
    env = DominionEnv()  # Complete games with the ordinary 200-turn cap.
    try:
        first = play_episode(model, env, seed=100)
        assert first == play_episode(model, env, seed=100)
        assert first["terminated"] != first["truncated"]
        assert first["end_reason"] in ("province_empty", "three_piles_empty", "turn_limit")
        assert first["steps"] <= 400
        short_env = DominionEnv(GameConfig(max_turns=1))
        result = play_episode(model, short_env, seed=101, deterministic=False)
        short_env.close()
        assert result["outcome"] == "truncated" and result["winner"] is None
        assert result["episode_return"] == 0
        for key, value in model.policy.state_dict().items():
            assert torch.equal(before[key], value)  # Evaluation never updates weights.
    finally:
        env.close()


def test_seeded_cpu_training_reproduces_parameters(tmp_path):
    states = []
    for name in ("first", "second"):
        path = train_ppo(
            tmp_path / name, game_config=GameConfig(max_turns=8),
            reward_config=RewardConfig(mode="potential_based"), ppo_kwargs=SMALL_PPO,
            seed=7, total_timesteps=16, checkpoint_freq=0,
        )
        states.append(MaskablePPO.load(path, device="cpu").policy.state_dict())
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


@pytest.mark.parametrize("reward_gamma,ppo_gamma", [(0.9, 0.99), (0.99, 0.9), (0.99, 1.0), (1.0, 0.99)])
def test_discount_mismatch_is_rejected_before_creating_outputs(tmp_path, reward_gamma, ppo_gamma):
    with pytest.raises(ValueError, match="gamma must match reward gamma"):
        train_ppo(
            tmp_path / "invalid", game_config=GameConfig(),
            reward_config=RewardConfig(gamma=reward_gamma),
            ppo_kwargs={**SMALL_PPO, "gamma": ppo_gamma},
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("gamma", [0.99, 1.0])
@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("scaling", ["raw", "fixed"])
def test_sb3_bootstraps_only_truncations_from_the_final_observation(monkeypatch, terminated, scaling, gamma):
    env = DominionEnv(GameConfig(max_turns=1), RewardConfig(mode="potential_based", gamma=gamma))
    reset = env.reset

    def reset_near_end(**kwargs):
        obs, info = reset(**kwargs)
        if terminated:
            # Force a normal end condition at cleanup, coinciding with the cap.
            env.game.state.supply["province"] = 0
            obs = encode_observation(env.game.state)
            info = env._info(obs)
        return obs, info

    monkeypatch.setattr(env, "reset", reset_near_end)
    policy_env = PolicyObservation(env, scaling)
    model = MaskablePPO("MlpPolicy", Monitor(policy_env), device="cpu", seed=0,
                        **{**SMALL_PPO, "n_steps": 2, "batch_size": 2, "gamma": gamma})
    forward = model.policy.forward

    def end_buying(obs, **kwargs):
        actions, values, log_probs = forward(obs, **kwargs)
        return torch.full_like(actions, END_BUYING), values, log_probs

    value_observations = []

    def constant_value(obs):
        value_observations.append(obs.detach().cpu().numpy().copy())
        return torch.full((len(obs), 1), 2.0, device=obs.device)

    monkeypatch.setattr(model.policy, "forward", end_buying)
    monkeypatch.setattr(model.policy, "predict_values", constant_value)

    class CaptureTransitions(BaseCallback):
        def __init__(self):
            super().__init__()
            self.transitions = []

        def _on_step(self):
            self.transitions.append((
                float(self.locals["rewards"][0]), self.locals["infos"][0].copy(),
                self.locals["new_obs"][0].copy(),
            ))
            return True

    callback = CaptureTransitions()
    try:
        model.learn(total_timesteps=2, callback=callback)
        assert len(callback.transitions) == 2
        for index, (raw_reward, info, reset_obs) in enumerate(callback.transitions):
            assert info["TimeLimit.truncated"] == (not terminated)
            final_obs = info["terminal_observation"]
            assert final_obs[FEATURE_INDEX["agent_turns_taken"]] == pytest.approx(
                1 / policy_env.divisors[FEATURE_INDEX["agent_turns_taken"]])
            assert reset_obs[FEATURE_INDEX["agent_turns_taken"]] == 0
            expected = raw_reward + (0 if terminated else gamma * 2)
            assert model.rollout_buffer.rewards[index, 0] == pytest.approx(expected)
            if not terminated:
                assert raw_reward == 0 and info["winner"] is None
                np.testing.assert_array_equal(value_observations[index][0], final_obs)
        # The final call evaluates the auto-reset observation; timeout calls are extra.
        assert len(value_observations) == (1 if terminated else 3)
    finally:
        model.get_env().close()


@pytest.mark.parametrize("seat", [0, 1])
def test_evaluation_purchase_metrics_and_relative_winner(seat):
    env = DominionEnv(GameConfig(max_turns=200), agent_player=seat)

    class ScriptedPolicy:
        observation_space = env.observation_space
        action_space = env.action_space

        def set_random_seed(self, seed):
            pass

        def predict(self, obs, deterministic, action_masks):
            # A transparent legal policy buys Silver then the usual Big Money cards.
            from dominion_rl.agents.big_money import big_money_purchase
            from dominion_rl.env.observations import CARD_IDS
            legal = [card for card, allowed in zip(CARD_IDS, action_masks) if allowed]
            card = big_money_purchase(legal, int(obs[FEATURE_INDEX["provinces_remaining"]]))
            action = END_BUYING if card is None else CARD_IDS.index(card)
            return np.array(action), None

    try:
        result = play_episode(ScriptedPolicy(), env, seed=7)
        assert result["agent_player"] == seat
        assert result["evaluation_seed"] == 7
        assert result["vp_differential"] == result["agent_vp"] - result["opponent_vp"]
        assert result["turn_count"] == result["agent_turns"] + result["opponent_turns"]
        for card in ("copper", "silver", "gold", "estate", "duchy", "province"):
            starting = {"copper": 7, "estate": 3}.get(card, 0)
            assert result[f"purchases_{card}"] == (
                encode_observation(env.game.state, seat)[FEATURE_INDEX[f"owned_{card}"]] - starting
            )
        assert result["first_province_turn"] is not None
        assert result["province_opportunities"] == result["purchases_province"]
        assert result["province_declines"] == 0
        assert result["copper_despite_silver_or_gold"] == 0
        assert result["discounted_episode_return"] == pytest.approx(
            env.reward_config.gamma ** (result["steps"] - 1) * result["episode_return"]
        )
        assert result["first_province_turn"] <= result["agent_turns"]
        assert result["first_duchy_turn"] is None or result["first_duchy_turn"] <= result["agent_turns"]
        expected = "tie" if result["winner"] is None else "win" if result["winner"] == seat else "loss"
        assert result["outcome"] == expected
    finally:
        env.close()


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_evaluation_cli_records_multiple_checkpoints_opponents_and_seats(tmp_path, gamma):
    import csv
    import subprocess
    import sys
    from pathlib import Path

    model_path = train_ppo(
        tmp_path / "trained", game_config=GameConfig(max_turns=8),
        reward_config=RewardConfig(mode="potential_based", gamma=gamma),
        ppo_kwargs={**SMALL_PPO, "gamma": gamma, "observation_scaling": "fixed",
                    "policy_kwargs": {"net_arch": [16, 16], "activation_fn": "relu"}},
        seed=19, total_timesteps=16, checkpoint_freq=0,
    )
    output = tmp_path / "games.csv"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_ppo.py"
    subprocess.run([
        sys.executable, str(script), "--model",
        str(model_path.parent / "checkpoints" / "ppo_0_steps.zip"), str(model_path),
        "--games", "2", "--seed", "20000", "--opponent", "big-money", "random",
        "--agent-player", "0", "1", "--max-turns", "2", "--output", str(output),
    ], check=True, capture_output=True, text=True)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {row["checkpoint_step"] for row in rows} == {"0", "16"}
    assert {row["reward_condition"] for row in rows} == {"potential_based"}
    assert {row["training_seed"] for row in rows} == {"19"}
    assert {float(row["training_gamma"]) for row in rows} == {gamma}
    assert {float(row["evaluation_return_gamma"]) for row in rows} == {gamma}
    assert {row["activation"] for row in rows} == {"relu"}
    assert {row["observation_scaling"] for row in rows} == {"fixed"}
    assert {row["training_opponent"] for row in rows} == {"big-money"}
    assert {row["evaluation_seed"] for row in rows} == {"20000", "20001"}
    assert {row["opponent"] for row in rows} == {"big-money", "random"}
    assert {row["agent_player"] for row in rows} == {"0", "1"}
    for row in rows:
        assert int(row["agent_vp"]) - int(row["opponent_vp"]) == int(row["vp_differential"])
        assert row["first_province_turn"] == ""
        assert row["outcome"] == "truncated"
        assert int(row["turn_count"]) == 2
    snapshot = yaml.safe_load(output.with_suffix(".yaml").read_text())
    assert snapshot["game"]["max_turns"] == 2
    assert snapshot["evaluation_return_gamma_source"] == "training_gamma"
    assert snapshot["models"][0]["training_gamma"] == gamma
    assert snapshot["models"][0]["evaluation_return_gamma"] == gamma
    assert len(snapshot["models"]) == 2
    assert len(snapshot["models"][0]["sha256"]) == 64
    assert snapshot["models"][0]["observation_preprocessing"]["mode"] == "fixed"
    # The CLI must reject both explicit incompatible input units and a stale
    # adjacent snapshot, rather than silently overriding the saved model.
    base_command = [sys.executable, str(script), "--model", str(model_path),
                    "--games", "1", "--max-turns", "2"]
    wrong_mode = subprocess.run(base_command + ["--observation-scaling", "raw", "--output",
        str(tmp_path / "wrong_mode.csv")], capture_output=True, text=True)
    assert wrong_mode.returncode != 0 and "Model requires fixed" in wrong_mode.stderr
    override_output = tmp_path / "common_discount.csv"
    subprocess.run(base_command + ["--return-gamma", "1.0", "--output", str(override_output)],
                   check=True, capture_output=True, text=True)
    with override_output.open() as stream:
        override_row = next(csv.DictReader(stream))
    assert float(override_row["training_gamma"]) == gamma
    assert float(override_row["evaluation_return_gamma"]) == 1.0
    run_path = model_path.parent / "run.yaml"
    training = yaml.safe_load(run_path.read_text())
    training["training_gamma"] = 0.99 if gamma == 1.0 else 1.0
    run_path.write_text(yaml.safe_dump(training))
    stale_gamma = subprocess.run(base_command + ["--output", str(tmp_path / "stale_gamma.csv")],
                                capture_output=True, text=True)
    assert stale_gamma.returncode != 0 and "gamma disagree" in stale_gamma.stderr
    training["training_gamma"] = gamma
    training["activation"] = "tanh"
    run_path.write_text(yaml.safe_dump(training))
    stale_snapshot = subprocess.run(base_command + ["--output", str(tmp_path / "stale.csv")],
                                    capture_output=True, text=True)
    assert stale_snapshot.returncode != 0 and "activation disagree" in stale_snapshot.stderr


def test_evaluation_counts_declined_provinces_and_avoidable_copper(monkeypatch):
    env = DominionEnv(GameConfig(max_turns=2))
    reset = env.reset

    def reset_with_eight_coins(**kwargs):
        reset(**kwargs)
        env.game.state.players[0].coins = 8
        obs = encode_observation(env.game.state)
        return obs, env._info(obs)

    monkeypatch.setattr(env, "reset", reset_with_eight_coins)

    class CopperPolicy:
        observation_space = env.observation_space
        action_space = env.action_space

        def set_random_seed(self, seed):
            pass

        def predict(self, obs, deterministic, action_masks):
            return np.array(0 if action_masks[0] else END_BUYING), None

    try:
        result = play_episode(CopperPolicy(), env, seed=3)
        assert result["province_opportunities"] == result["province_declines"] == 1
        assert result["purchases_province"] == 0
        assert result["copper_despite_silver_or_gold"] == 1
        assert result["copper_despite_non_copper"] == 1
        assert result["tie_adjusted_score"] is None
        assert result["discounted_episode_return"] == 0
    finally:
        env.close()
