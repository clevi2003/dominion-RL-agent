"""Episode-level sampling, fixed policy dispatch and observation boundaries."""

import numpy as np
import pytest

from dominion_rl.config import GameConfig
from dominion_rl.env import DominionEnv
import dominion_rl.env.dominion_env as env_module
from dominion_rl.env.dominion_env import END_BUYING


def test_mixed_selection_matches_seeded_fair_coin_and_reproduces():
    first, second = DominionEnv(opponent="mixed"), DominionEnv(opponent="mixed")
    rng = np.random.default_rng(42)
    selections = []
    try:
        for index in range(100):
            rng.integers(0, 2**63 - 1, size=2)  # Existing engine/policy streams.
            expected = "random" if rng.random() < 0.5 else "big-money"
            seed = 42 if index == 0 else None
            obs, info = first.reset(seed=seed)
            repeated, repeated_info = second.reset(seed=seed)
            assert info["episode_opponent"] == repeated_info["episode_opponent"] == expected
            np.testing.assert_array_equal(obs, repeated)
            selections.append(expected)
        assert set(selections) == {"random", "big-money"}
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("seat", [0, 1])
def test_mixed_opponent_is_fixed_throughout_each_episode(monkeypatch, seat):
    calls = []

    def random_policy(legal, rng):
        calls.append("random")
        return None

    def money_policy(legal, provinces):
        calls.append("big-money")
        return None

    monkeypatch.setattr(env_module, "random_purchase", random_policy)
    monkeypatch.setattr(env_module, "big_money_purchase", money_policy)
    env = DominionEnv(GameConfig(max_turns=8), opponent="mixed", agent_player=seat)
    try:
        for seed in range(10):
            calls.clear()
            _, info = env.reset(seed=seed)
            selected = info["episode_opponent"]
            while True:
                _, _, terminated, truncated, info = env.step(END_BUYING)
                assert info["episode_opponent"] == selected
                if terminated or truncated:
                    break
            assert calls and set(calls) == {selected}
        assert sum(env.opponent_step_counts.values()) == 40
    finally:
        env.close()


def test_opponent_identity_does_not_add_an_observation_feature():
    mixed = DominionEnv(opponent="mixed")
    try:
        obs, info = mixed.reset(seed=42)
        fixed = DominionEnv(opponent=info["episode_opponent"])
        try:
            expected, _ = fixed.reset(seed=42)
            assert obs.shape == (41,) and obs.dtype == np.float32
            np.testing.assert_array_equal(obs, expected)
            np.testing.assert_array_equal(mixed.action_masks(), fixed.action_masks())
        finally:
            fixed.close()
    finally:
        mixed.close()
