from dataclasses import replace

import numpy as np
import pytest

from dominion_rl.config import RewardConfig
from dominion_rl.env.observations import FEATURE_INDEX, FEATURE_NAMES
from dominion_rl.env.rewards import calculate_reward, potential, terminal_reward


def observation(agent_vp, opponent_vp):
    obs = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    obs[FEATURE_INDEX["agent_vp"]] = agent_vp
    obs[FEATURE_INDEX["opponent_vp"]] = opponent_vp
    return obs


@pytest.mark.parametrize("winner,expected", [(0, 1), (1, -1), (None, 0)])
def test_terminal_only_uses_official_outcome(winner, expected):
    config = RewardConfig()
    # VP difference deliberately disagrees with winner: the engine decides tiebreaks.
    assert calculate_reward(
        observation(9, 3), observation(9, 3), config,
        terminated=True, truncated=False, winner=winner,
    ) == expected


@pytest.mark.parametrize("truncated", [False, True])
def test_no_outcome_without_true_termination(truncated):
    assert terminal_reward(0, RewardConfig(), terminated=False, truncated=truncated) == 0
    assert calculate_reward(
        observation(3, 3), observation(12, 0), RewardConfig(),
        terminated=False, truncated=truncated, winner=None,
    ) == 0


def test_reward_values_respect_configuration():
    config = RewardConfig(win_reward=2, tie_reward=0.5, loss_reward=-2)
    assert terminal_reward(0, config, terminated=True, truncated=False) == 2
    assert terminal_reward(None, config, terminated=True, truncated=False) == 0.5
    assert terminal_reward(1, config, terminated=True, truncated=False) == -2


def test_potential_is_opponent_relative_normalized_and_not_bounded():
    config = RewardConfig()
    assert potential(observation(3, 9), config) == -6 / 48
    assert potential(observation(100, 0), config) == 100 / 48
    assert potential(observation(100, 0), config, terminated=True) == 0


def test_shaping_uses_configured_gamma_beta_and_normalization():
    config = RewardConfig(mode="potential_based", gamma=0.5, shaping_strength=2, vp_normalization=12)
    reward = calculate_reward(
        observation(6, 3), observation(9, 3), config,
        terminated=False, truncated=False, winner=None,
    )
    assert reward == pytest.approx(2 * (0.5 * (6 / 12) - (3 / 12)))


def test_true_terminal_zero_potential_even_with_nonzero_final_vp_difference():
    config = RewardConfig(mode="potential_based")
    reward = calculate_reward(
        observation(9, 3), observation(15, 3), config,
        terminated=True, truncated=False, winner=0,
    )
    assert reward == pytest.approx(1 - 6 / 48)


def test_truncation_keeps_final_nonterminal_potential_and_no_outcome():
    config = RewardConfig(mode="potential_based")
    reward = calculate_reward(
        observation(3, 9), observation(3, 15), config,
        terminated=False, truncated=True, winner=None,
    )
    assert reward == pytest.approx(0.99 * (-12 / 48) - (-6 / 48))


def test_zero_beta_equals_terminal_only_and_shaping_does_not_mutate_observations():
    config = RewardConfig(mode="potential_based", shaping_strength=0)
    before, after = observation(3, 3), observation(9, 3)
    old_before, old_after = before.copy(), after.copy()
    for terminated in (False, True):
        flags = dict(terminated=terminated, truncated=False, winner=0 if terminated else None)
        assert calculate_reward(before, after, config, **flags) == calculate_reward(
            before, after, replace(config, mode="terminal_only"), **flags,
        )
    np.testing.assert_array_equal(before, old_before)
    np.testing.assert_array_equal(after, old_after)


def test_discounted_shaping_telescopes_at_terminal_endpoint():
    config = RewardConfig(mode="potential_based", gamma=0.8, shaping_strength=2)
    observations = [observation(9, 3), observation(10, 3), observation(10, 6), observation(20, 6)]
    shaped = [calculate_reward(
        observations[index], observations[index + 1], config,
        terminated=index == 2, truncated=False, winner=0 if index == 2 else None,
    ) for index in range(3)]
    discounted = sum(config.gamma**index * reward for index, reward in enumerate(shaped))
    assert discounted == pytest.approx(config.gamma**2 - 2 * (6 / 48))
