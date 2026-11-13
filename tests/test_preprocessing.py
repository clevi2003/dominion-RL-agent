"""Policy input units must not change raw rewards or game summaries."""

import numpy as np
import pytest

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import END_BUYING
from dominion_rl.env.observations import FEATURE_INDEX, FEATURE_NAMES
from dominion_rl.env.preprocessing import FIXED_DIVISORS, PolicyObservation


@pytest.mark.parametrize("mode", ["raw", "fixed"])
def test_policy_observation_values_bounds_and_no_mutation(mode):
    raw = np.zeros(41, dtype=np.float32)
    values = {
        "coins": (80, 8), "buys": (1, 1), "hand_estate": (5, 1),
        "in_play_gold": (5, 1), "owned_copper": (30, .5),
        "opponent_owned_estate": (7, .5), "supply_copper": (23, .5),
        "supply_estate": (4, .5), "deck_size": (80, .5), "discard_size": (80, .5),
        "agent_vp": (24, .5), "opponent_vp": (24, .5),
        "turn_number": (100, .5), "agent_turns_taken": (50, .5),
        "opponent_turns_taken": (50, .5), "provinces_remaining": (4, .5),
        "empty_supply_piles": (3, .5),
    }
    expected = raw.copy()
    for name, (value, scaled) in values.items():
        raw[FEATURE_INDEX[name]] = value
        expected[FEATURE_INDEX[name]] = scaled if mode == "fixed" else value
    original = raw.copy()
    env = PolicyObservation(DominionEnv(), mode)
    try:
        actual = env.observation(raw)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(raw, original)
        assert len(FIXED_DIVISORS) == len(FEATURE_NAMES) == 41
        assert all(divisor > 0 for divisor in FIXED_DIVISORS)
        assert actual.dtype == np.float32 and actual.shape == (41,)
        assert env.observation_space.contains(actual)
        assert np.all(env.observation_space.low == 0)
        assert np.all(np.isinf(env.observation_space.high))
    finally:
        env.close()


def test_fixed_wrapper_preserves_raw_vp_shaping_masks_and_terminal_info():
    config = RewardConfig(mode="potential_based")
    raw = DominionEnv(GameConfig(max_turns=2), config)
    wrapped = PolicyObservation(DominionEnv(GameConfig(max_turns=2), config), "fixed")
    try:
        before, _ = raw.reset(seed=3)
        scaled, _ = wrapped.reset(seed=3)
        np.testing.assert_array_equal(scaled, before / np.asarray(FIXED_DIVISORS, dtype=np.float32))
        for action in (3, END_BUYING):
            np.testing.assert_array_equal(wrapped.get_wrapper_attr("action_masks")(), raw.action_masks())
            obs, reward, terminated, truncated, info = raw.step(action)
            actual, actual_reward, actual_term, actual_trunc, actual_info = wrapped.step(action)
            np.testing.assert_array_equal(actual, obs / wrapped.divisors)
            assert actual_reward == reward
            assert (actual_term, actual_trunc) == (terminated, truncated)
            assert actual_info["scores"] == info["scores"] == (4, 3)
            assert actual_info["turns_taken"] == info["turns_taken"]
            if action == 3:
                assert reward == pytest.approx(.99 / 48)
            else:
                assert truncated and not terminated
                assert reward == pytest.approx(-.01 / 48)
    finally:
        raw.close()
        wrapped.close()
