"""Terminal outcomes and potential-based shaping, once per purchasing decision."""

import numpy as np

from dominion_rl.config import RewardConfig
from dominion_rl.env.observations import FEATURE_INDEX


def potential(observation: np.ndarray, config: RewardConfig, *, terminated: bool = False) -> float:
    """Use inferable VP; zero only at a true terminal state, not a truncation."""
    if terminated:
        return 0.0
    return (
        float(observation[FEATURE_INDEX["agent_vp"]])
        - float(observation[FEATURE_INDEX["opponent_vp"]])
    ) / config.vp_normalization


def terminal_reward(
    winner: int | None, config: RewardConfig, *, terminated: bool, truncated: bool
) -> float:
    """Score the engine's official outcome from player 0's perspective."""
    if not terminated or truncated:
        return 0.0
    if winner is None:
        return float(config.tie_reward)
    return float(config.win_reward if winner == 0 else config.loss_reward)


def calculate_reward(
    previous_observation: np.ndarray,
    observation: np.ndarray,
    config: RewardConfig,
    *,
    terminated: bool,
    truncated: bool,
    winner: int | None,
) -> float:
    """Apply r + beta * (gamma * Phi(next) - Phi(previous)).

    Truncation has no outcome reward, retains Phi(next), and requires later
    learners to bootstrap from this final observation. No reset occurs here.
    The learner's per-decision discount must match config.gamma.
    """
    reward = terminal_reward(winner, config, terminated=terminated, truncated=truncated)
    if config.mode == "potential_based":
        reward += config.shaping_strength * (
            config.gamma * potential(observation, config, terminated=terminated)
            - potential(previous_observation, config)
        )
    return float(reward)
