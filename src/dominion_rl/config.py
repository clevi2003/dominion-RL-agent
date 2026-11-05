"""Small typed configuration records loaded with PyYAML's safe loader."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import yaml

from dominion_rl.game.cards import CARD_REGISTRY, CardId


@dataclass(frozen=True)
class GameConfig:
    """V0 experiment settings; fixed card properties and rules stay in Python.

    max_turns counts total individual player turns, not rounds or purchases.
    A seed of None requests fresh randomness; an integer requests repeatable setup.
    The engine owns and uses a local RNG rather than global RNG state.
    """

    version: Literal["V0"] = "V0"
    cards: tuple[CardId, ...] = tuple(CARD_REGISTRY)
    num_players: int = 2
    seed: int | None = 0
    max_turns: int = 200

    def __post_init__(self) -> None:
        if self.version != "V0" or self.num_players != 2:
            raise ValueError("Only two-player V0 configuration is supported")
        if len(self.cards) != 6 or set(self.cards) != set(CARD_REGISTRY):
            raise ValueError("V0 requires each of the six basic card IDs exactly once")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer or null")
        if type(self.max_turns) is not int or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")


@dataclass(frozen=True)
class RewardConfig:
    """Settings for terminal-only and potential-based reward calculations.

    shaping_strength applies only in potential_based mode. Shaping uses
    beta * (gamma * Phi(next) - Phi(current)), with Phi = public VP difference
    / vp_normalization. gamma must match the learner's per-purchase discount.
    """

    mode: Literal["terminal_only", "potential_based"] = "terminal_only"
    gamma: float = 0.99
    shaping_strength: float = 1.0
    vp_normalization: float = 48.0
    win_reward: float = 1.0
    tie_reward: float = 0.0
    loss_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.mode not in ("terminal_only", "potential_based"):
            raise ValueError("Unknown reward mode")
        numeric = (
            self.gamma, self.shaping_strength, self.vp_normalization,
            self.win_reward, self.tie_reward, self.loss_reward,
        )
        if any(type(value) not in (int, float) or not isfinite(value) for value in numeric):
            raise ValueError("Reward parameters must be finite numbers")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.shaping_strength < 0 or self.vp_normalization <= 0:
            raise ValueError("shaping_strength must be >= 0 and vp_normalization > 0")


def _load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValueError("Configuration must be a YAML mapping with string keys")
    return data


def load_game_config(path: str | Path) -> GameConfig:
    """Read a game YAML file, rejecting invalid cards and unsupported settings."""
    data = _load_mapping(path)
    if "cards" in data:
        if not isinstance(data["cards"], list) or any(
            not isinstance(card_id, str) for card_id in data["cards"]
        ):
            raise ValueError("cards must be a YAML list of card ID strings")
        data["cards"] = tuple(data["cards"])
    return GameConfig(**data)


def load_reward_config(path: str | Path) -> RewardConfig:
    """Read settings for the terminal-only/shaped reward comparison."""
    return RewardConfig(**_load_mapping(path))
