"""Fixed policy-input units; raw game observations and reward units stay intact."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dominion_rl.env.observations import feature_names

# The single definition of all 41 divisors, in FEATURE_NAMES order.
# Six-card blocks are Copper, Silver, Gold, Estate, Duchy, Province.
FIXED_DIVISORS: tuple[float, ...] = (
    10, 1,                          # Coins, Buys.
    5, 5, 5, 5, 5, 5,              # Hand counts (five-card V0 hands).
    5, 5, 5, 5, 5, 5,              # In-play counts.
    60, 40, 30, 14, 8, 8,          # Agent ownership / total physical V0 cards.
    60, 40, 30, 14, 8, 8,          # Opponent ownership / same totals.
    46, 40, 30, 8, 8, 8,           # Supply / initial Supply after starting decks.
    160, 160,                      # Deck/discard sizes / all physical V0 cards.
    48, 48,                        # Agent/opponent VP (values may exceed one).
    200, 100, 100,                 # Game turn, completed agent/opponent turns.
    8, 6,                          # Provinces remaining, empty Supply piles.
)


def preprocessing_config(mode: str, game_version: str = "V0") -> dict[str, Any]:
    """Serializable, versioned policy-input contract. No clipping or statistics."""
    if mode not in ("raw", "fixed"):
        raise ValueError("observation_scaling must be 'raw' or 'fixed'")
    names = feature_names(game_version)
    if game_version in ("V1", "V2", "V3") and mode != "raw":
        raise ValueError(f"{game_version} supports raw observations only")
    return {
        "version": 1, "mode": mode, "features": list(names),
        "divisors": list(FIXED_DIVISORS) if mode == "fixed" else [1] * len(names),
    }


class PolicyObservation(gym.ObservationWrapper):
    """Scale only observations returned to SB3; preserve base rewards and info."""

    def __init__(self, env: gym.Env, mode: str = "raw") -> None:
        super().__init__(env)
        self.preprocessing = preprocessing_config(mode, env.unwrapped.game_config.version)
        self.divisors = np.asarray(self.preprocessing["divisors"], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=env.observation_space.low / self.divisors,
            high=env.observation_space.high / self.divisors,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation / self.divisors
