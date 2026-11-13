"""Terminal outcomes and potential-based shaping, once per purchasing decision."""

from functools import lru_cache
from math import comb, isfinite

import numpy as np

from dominion_rl.config import RewardConfig
from dominion_rl.env.observations import FEATURE_INDEX, V1_FEATURE_INDEX, V2_FEATURE_INDEX, V3_FEATURE_INDEX
from dominion_rl.game.cards import CARD_REGISTRY, TreasureCard, V0_CARD_IDS, V1_CARD_IDS, V2_CARD_IDS, V3_CARD_IDS


@lru_cache(maxsize=16384, typed=True)
def five_card_p8(copper: int, silver: int, gold: int, victory: int) -> float:
    """Exact P(coins >= 8) in a uniform five-card ownership sample.

    This hypothetical composition probability is not the conditional probability
    of the actual next draw. No zones, draw order or random sampling are used.
    The legacy argument name victory means all zero-Treasure cards, including
    Actions. Legal V0/V1/V2/V3 ownerships have at least ten cards; undersized inputs
    are rejected.
    The bounded cache only memoizes this deterministic count-to-probability map.
    """
    zero_coin = victory
    counts = (copper, silver, gold, zero_coin)
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("Card counts must be nonnegative integers")
    total = sum(counts)
    if total < 5:
        raise ValueError("A five-card sample requires at least five owned cards")
    favorable = 0
    # At most 56 count tuples (i+j+k+l=5), with exact integer weights.
    for i in range(min(copper, 5)+1):
        for j in range(min(silver, 5-i)+1):
            for k in range(min(gold, 5-i-j)+1):
                l = 5-i-j-k
                if l <= zero_coin and i+2*j+3*k >= 8:
                    favorable += comb(copper, i)*comb(silver, j)*comb(gold, k)*comb(zero_coin, l)
    return favorable / comb(total, 5)


def ownership_p8(observation: np.ndarray, *, opponent: bool = False) -> float:
    """Read inferable RAW ownership; Actions dilute Treasure just like Victory."""
    if observation.shape == (41,):
        ids, index = V0_CARD_IDS, FEATURE_INDEX
    elif observation.shape == (51,):
        ids, index = V1_CARD_IDS, V1_FEATURE_INDEX
    elif observation.shape == (56,):
        ids, index = V2_CARD_IDS, V2_FEATURE_INDEX
    elif observation.shape == (66,):
        ids, index = V3_CARD_IDS, V3_FEATURE_INDEX
    else:
        raise ValueError("P8 requires a known V0/V1/V2/V3 observation schema")
    prefix = "opponent_owned_" if opponent else "owned_"
    counts = [float(observation[index[prefix+card]]) for card in ids]
    if any(not isfinite(n) or n < 0 or not n.is_integer() for n in counts):
        raise ValueError("P8 requires raw nonnegative integer ownership counts")
    owned = dict(zip(ids, map(int, counts)))
    zero_coin = sum(n for card, n in owned.items() if not isinstance(CARD_REGISTRY[card], TreasureCard))
    return five_card_p8(owned["copper"], owned["silver"], owned["gold"], zero_coin)


def potential(observation: np.ndarray, config: RewardConfig, *, terminated: bool = False) -> float:
    """Use public VP/ownership; zero at true terminals, retain at truncations."""
    if terminated:
        return 0.0
    vp = (
        float(observation[FEATURE_INDEX["agent_vp"]])
        - float(observation[FEATURE_INDEX["opponent_vp"]])
    ) / config.vp_normalization
    if config.potential == "vp_economy":
        return vp + config.economy_coefficient * (
            ownership_p8(observation) - ownership_p8(observation, opponent=True))
    return vp


def terminal_reward(
    winner: int | None, config: RewardConfig, *, terminated: bool, truncated: bool,
    observation: np.ndarray | None = None,
) -> float:
    """Official outcome, or unbounded final VP difference / 48, agent-relative.

    Margin uses actual final public VP from the raw observation, independently
    of the potential normalization and official turn-count tiebreak. The default
    win_loss branch retains the original operations and return values.
    """
    if not terminated or truncated:
        return 0.0
    if config.terminal_objective == "vp_margin":
        if observation is None:
            raise ValueError("vp_margin requires an actual final observation")
        agent = float(observation[FEATURE_INDEX["agent_vp"]])
        opponent = float(observation[FEATURE_INDEX["opponent_vp"]])
        if not isfinite(agent) or not isfinite(opponent):
            raise ValueError("vp_margin requires finite final VP totals")
        return (agent - opponent) / 48.0
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
    reward = terminal_reward(winner, config, terminated=terminated, truncated=truncated, observation=observation)
    if config.mode == "potential_based":
        reward += config.shaping_strength * (
            config.gamma * potential(observation, config, terminated=terminated)
            - potential(previous_observation, config)
        )
    return float(reward)
