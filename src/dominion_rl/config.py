"""Small typed configuration records loaded with PyYAML's safe loader."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import yaml

from dominion_rl.game.cards import CardId, card_ids_for_version


@dataclass(frozen=True)
class GameConfig:
    """Versioned experiment settings; fixed card properties and rules stay in Python.

    max_turns counts total individual player turns, not rounds or purchases.
    A seed of None requests fresh randomness; an integer requests repeatable setup.
    The engine owns and uses a local RNG rather than global RNG state.
    """

    version: Literal["V0", "V1", "V2", "V3"] = "V0"
    cards: tuple[CardId, ...] | None = None
    num_players: int = 2
    seed: int | None = 0
    max_turns: int = 200

    def __post_init__(self) -> None:
        if self.num_players != 2:
            raise ValueError("Only two-player configuration is supported")
        expected = card_ids_for_version(self.version)
        if self.cards is None:
            object.__setattr__(self, "cards", expected)
        if len(self.cards) != len(expected) or set(self.cards) != set(expected):
            raise ValueError(f"{self.version} requires each configured card ID exactly once: {expected}")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer or null")
        if type(self.max_turns) is not int or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")


@dataclass(frozen=True)
class RewardConfig:
    """Settings for terminal-only and potential-based reward calculations.

    shaping_strength applies only in potential_based mode. Shaping uses
    beta * (gamma * Phi(next) - Phi(current)), with Phi = public VP difference
    / vp_normalization, optionally plus a relative hypothetical-hand P8 term.
    gamma must match the learner's per-purchase discount.
    """

    mode: Literal["terminal_only", "potential_based"] = "terminal_only"
    gamma: float = 0.99
    shaping_strength: float = 1.0
    vp_normalization: float = 48.0
    win_reward: float = 1.0
    tie_reward: float = 0.0
    loss_reward: float = -1.0
    # Defaults preserve legacy VP shaping when these fields are absent in YAML.
    potential: Literal["vp", "vp_economy"] = "vp"
    economy_coefficient: float = 0.25
    # Separate from reward mode/potential. Missing on old YAML/archives means win_loss.
    terminal_objective: Literal["win_loss", "vp_margin"] = "win_loss"

    def __post_init__(self) -> None:
        if self.mode not in ("terminal_only", "potential_based"):
            raise ValueError("Unknown reward mode")
        if self.terminal_objective not in ("win_loss", "vp_margin"):
            raise ValueError("terminal_objective must be win_loss or vp_margin")
        if self.potential not in ("vp", "vp_economy"):
            raise ValueError("potential must be vp or vp_economy")
        numeric = (
            self.gamma, self.shaping_strength, self.vp_normalization,
            self.win_reward, self.tie_reward, self.loss_reward, self.economy_coefficient,
        )
        if any(type(value) not in (int, float) or not isfinite(value) for value in numeric):
            raise ValueError("Reward parameters must be finite numbers")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.economy_coefficient < 0:
            raise ValueError("economy_coefficient must be nonnegative")
        if self.shaping_strength < 0 or self.vp_normalization <= 0:
            raise ValueError("shaping_strength must be >= 0 and vp_normalization > 0")


def _load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValueError("Configuration must be a YAML mapping with string keys")
    return data


def load_game_config(path: str | Path) -> GameConfig:
    """Read a game YAML file, rejecting invalid version/card combinations."""
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
