"""Gymnasium interface: one step buys one card or explicitly ends buying."""

from __future__ import annotations

from dataclasses import replace
from collections import Counter
from random import Random
from math import isfinite
from typing import Any, Literal
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.random_agent import random_purchase
from dominion_rl.agents.engine_money import engine_money_purchase
from dominion_rl.agents.festival_money import festival_money_purchase
from dominion_rl.agents.militia_money import militia_money_purchase
from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env.observations import CARD_IDS, FEATURE_INDEX, feature_names, encode_observation
from dominion_rl.game.cards import card_ids_for_version
from dominion_rl.env.rewards import calculate_reward
from dominion_rl.game import DominionGame

# V0 action IDs: Copper, Silver, Gold, Estate, Duchy, Province, End buying.
ACTION_NAMES: tuple[str, ...] = (*CARD_IDS, "end_buying")
END_BUYING: int = len(CARD_IDS)


DEFAULT_CURRICULUM = (
    {"start_fraction": 0.0, "random_probability": 1.0},
    {"start_fraction": 0.2, "random_probability": 0.5},
    {"start_fraction": 0.5, "random_probability": 0.2},
)


def validate_curriculum(stages):
    """Copy a fractional schedule; each stage lasts until the next start."""
    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError("curriculum must be a nonempty sequence of stages")
    copied = []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"start_fraction", "random_probability"}:
            raise ValueError("Each curriculum stage requires start_fraction and random_probability")
        start, probability = stage["start_fraction"], stage["random_probability"]
        if any(type(v) not in (float, int) or not isfinite(v) for v in (start, probability)):
            raise ValueError("Curriculum values must be finite numbers")
        if not 0 <= start < 1 or not 0 <= probability <= 1:
            raise ValueError("Curriculum fractions/probabilities are out of range")
        if (not copied and start != 0) or (copied and start <= copied[-1]["start_fraction"]):
            raise ValueError("Curriculum must start at zero with strictly increasing starts")
        copied.append(dict(start_fraction=float(start), random_probability=float(probability)))
    return copied


def curriculum_stage(steps, total_timesteps, stages):
    """Stage at an episode's first decision; the last stage extends past budget."""
    return max(i for i, stage in enumerate(stages)
               if steps >= stage["start_fraction"] * total_timesteps)


class DominionEnv(gym.Env[np.ndarray, int]):
    """The selected player purchases; a fixed opponent runs outside the engine.

    The configured seed initializes Gymnasium's RNG on the first unseeded reset.
    Later reset() calls advance its stream; reset(seed=N) starts it over. Engine
    and policy seeds are drawn independently per episode and never exposed in info.
    Observations/scores are agent-relative; winner/current_player and turns_taken
    retain physical player indices/order. Player 1 reset runs the opening opponent.
    episode_opponent and curriculum stage are logging metadata only.
    Curriculum progress counts lifetime agent decisions, including truncations;
    reset(seed=N) reseeds games but does not rewind that training clock.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        game_config: GameConfig | None = None,
        reward_config: RewardConfig | None = None,
        opponent: Literal["big-money", "random", "mixed", "curriculum", "engine-money", "festival-money", "militia-money", "historical", "frozen"] = "big-money",
        render_mode: Literal["ansi"] | None = None,
        agent_player: int = 0,
        curriculum_total_timesteps: int | None = None,
        curriculum: list[dict[str, float]] | None = None,
        opponent_pool: str | Path | None = None,
        frozen_opponent_id: str | None = None,
    ) -> None:
        super().__init__()
        if opponent not in ("big-money", "random", "mixed", "curriculum", "engine-money", "festival-money", "militia-money", "historical", "frozen"):
            raise ValueError("Unknown opponent mode")
        if opponent == "curriculum":
            if type(curriculum_total_timesteps) is not int or curriculum_total_timesteps <= 0:
                raise ValueError("curriculum requires a positive curriculum_total_timesteps")
        elif curriculum is not None or curriculum_total_timesteps is not None:
            raise ValueError("Curriculum settings require opponent='curriculum'")
        self.curriculum = validate_curriculum(DEFAULT_CURRICULUM if curriculum is None else curriculum) if opponent == "curriculum" else None
        self.curriculum_total_timesteps = curriculum_total_timesteps
        self.curriculum_stage_index = None
        self.episode_start_step = 0
        self.curriculum_step_counts = [{"random": 0, "big-money": 0} for _ in self.curriculum or ()]
        if render_mode not in (None, "ansi"):
            raise ValueError("Only render_mode='ansi' is supported")
        self.game_config = game_config if game_config is not None else GameConfig()
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        if opponent == "engine-money" and self.game_config.version not in ("V1", "V2", "V3"):
            raise ValueError("engine-money requires a V1, V2 or V3 game")
        if opponent == "festival-money" and self.game_config.version not in ("V2", "V3"):
            raise ValueError("festival-money requires a V2 or V3 game")
        if opponent == "militia-money" and self.game_config.version != "V3":
            raise ValueError("militia-money requires a V3 game")
        self.card_ids = card_ids_for_version(self.game_config.version)
        self.feature_names = feature_names(self.game_config.version)
        self.action_names = (*self.card_ids, "end_buying")
        self.end_buying = len(self.card_ids)
        self.game_contract = dict(version=self.game_config.version,
                                  features=list(self.feature_names), actions=list(self.action_names))
        if type(agent_player) is not int or agent_player not in (0, 1):
            raise ValueError("agent_player must be 0 or 1")
        if agent_player == 1 and self.game_config.max_turns < 2:
            raise ValueError("player 1 requires max_turns >= 2 to reach its first decision")
        self.frozen_policies = {}
        self.frozen_pool_metadata = None
        self.historical_probabilities = None
        if opponent in ("historical", "frozen"):
            if self.game_config.version != "V1" or opponent_pool is None:
                raise ValueError("Frozen/historical opponents require V1 and an explicit opponent pool")
            if (opponent == "frozen") != (frozen_opponent_id is not None):
                raise ValueError("Only frozen mode requires a frozen_opponent_id")
            from dominion_rl.agents.frozen_policy import load_frozen_policies, pool_provenance
            pool,self.frozen_policies = load_frozen_policies(opponent_pool,
                identifiers=[frozen_opponent_id] if opponent == "frozen" else None,training=opponent == "historical")
            self.frozen_pool_metadata = pool_provenance(opponent_pool,pool,list(self.frozen_policies))
            self.historical_probabilities = ({r['opponent']:r['probability'] for r in pool['training_mixture']}
                if opponent == "historical" else {frozen_opponent_id:1.})
        elif opponent_pool is not None or frozen_opponent_id is not None:
            raise ValueError("opponent_pool/frozen_opponent_id require historical or frozen mode")
        self.agent_player = agent_player
        self.opponent = opponent
        self.episode_opponent: str | None = None
        # Lifetime agent decision counts, including an unfinished final episode.
        self.opponent_step_counts = {"big-money": 0, "random": 0}
        if opponent in ("engine-money", "festival-money", "militia-money"):
            self.opponent_step_counts[opponent] = 0
        if self.historical_probabilities is not None:
            self.opponent_step_counts = {name:0 for name in self.historical_probabilities}
        self.opponent_reset_counts = {name:0 for name in self.opponent_step_counts}
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.feature_names),), dtype=np.float32
        )
        self.game: DominionGame | None = None
        self._opponent_rng = Random(0)  # Replaced by an episode seed at reset.

    def _require_game(self) -> DominionGame:
        if self.game is None:
            raise gym.error.ResetNeeded("Call reset() before using the environment")
        return self.game

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None and self._np_random is None:
            seed = self.game_config.seed
        super().reset(seed=seed)
        engine_seed, opponent_seed = (
            int(value) for value in self.np_random.integers(0, 2**63 - 1, size=2)
        )
        if self.game is None:
            self.game = DominionGame(replace(self.game_config, seed=engine_seed))
        else:
            self.game.reset(seed=engine_seed)
        self._opponent_rng = Random(opponent_seed)
        # Fixed opponents consume exactly the same RNG draws as before. Mixed
        # selection is independent per reset and never changes within an episode.
        if self.opponent in ("historical", "frozen"):
            names = list(self.historical_probabilities)
            self.episode_opponent = (str(self.np_random.choice(names,p=list(self.historical_probabilities.values())))
                if self.opponent == "historical" else names[0])
        elif self.opponent == "curriculum":
            # Lifetime decisions survive reset(seed=...), just like exposure counts.
            # SB3 auto-reset therefore sees the terminating decision immediately.
            self.episode_start_step = sum(self.opponent_step_counts.values())
            self.curriculum_stage_index = curriculum_stage(
                self.episode_start_step, self.curriculum_total_timesteps, self.curriculum)
            probability = self.curriculum[self.curriculum_stage_index]["random_probability"]
            # A certain choice needs no RNG draw. The Random-only warm-up then
            # shares the exact game stream of the paired Random control.
            if probability == 1:
                self.episode_opponent = "random"
            elif probability == 0:
                self.episode_opponent = "big-money"
            else:
                self.episode_opponent = "random" if self.np_random.random() < probability else "big-money"
        else:
            self.episode_opponent = (
                "random" if self.np_random.random() < 0.5 else "big-money"
            ) if self.opponent == "mixed" else self.opponent
        self.opponent_reset_counts[self.episode_opponent] += 1
        self._advance_opponent()
        observation = encode_observation(self.game.state, self.agent_player)
        return observation, self._info(observation)

    def _advance_opponent(self) -> None:
        game = self._require_game()
        state = game.state
        while not (state.terminated or state.truncated) and state.current_player != self.agent_player:
            legal = game.legal_purchases()
            if self.episode_opponent in self.frozen_policies:
                from dominion_rl.agents.frozen_policy import frozen_action
                action = frozen_action(self.frozen_policies[self.episode_opponent],game)
                if action == self.end_buying:game.end_buying()
                else:game.buy_card(self.card_ids[action])
                continue
            if self.episode_opponent in ("engine-money", "festival-money", "militia-money"):
                player = state.players[state.current_player]
                owned = Counter(player.deck + player.hand + player.discard + player.in_play)
                purchase = {"engine-money": engine_money_purchase, "festival-money": festival_money_purchase,
                            "militia-money": militia_money_purchase}[self.episode_opponent]
                decision = purchase(legal, state.supply["province"], owned)
            else:
                decision = (
                    random_purchase(legal, self._opponent_rng) if self.episode_opponent == "random"
                    else big_money_purchase(legal, state.supply["province"])
                )
            if decision is None:
                game.end_buying()
            else:
                game.buy_card(decision)

    def action_masks(self) -> np.ndarray:
        """Fresh bool mask; End buying is legal only during an active episode."""
        game = self._require_game()
        mask = np.zeros(len(self.action_names), dtype=np.bool_)
        if not (game.state.terminated or game.state.truncated):
            legal = game.legal_purchases()
            mask[:-1] = [card_id in legal for card_id in self.card_ids]
            mask[self.end_buying] = True
        return mask

    def _info(self, observation: np.ndarray) -> dict[str, Any]:
        state = self._require_game().state
        # Public game summaries and logging metadata; no private zones or RNGs.
        # The purchasing policy receives encode_observation(), not this info dict.
        curriculum_info = {} if self.curriculum is None else {
            "curriculum_stage": self.curriculum_stage_index,
            "episode_start_step": self.episode_start_step,
            "curriculum_random_probability": self.curriculum[self.curriculum_stage_index]["random_probability"],
        }
        return {
            **curriculum_info,
            "action_mask": self.action_masks(),
            "turn_number": state.turn_number,
            "current_player": state.current_player,
            "agent_player": self.agent_player,
            # Logging metadata only; never included in encode_observation().
            "episode_opponent": self.episode_opponent,
            "scores": tuple(int(observation[FEATURE_INDEX[name]])
                            for name in ("agent_vp", "opponent_vp")),
            "turns_taken": tuple(player.turns_taken for player in state.players),
            "winner": state.winner,
            "end_reason": state.end_reason,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        game = self._require_game()
        state = game.state
        if state.terminated or state.truncated:
            raise RuntimeError("Episode has ended; call reset() before stepping")
        if state.current_player != self.agent_player:
            raise RuntimeError("The environment expects the controlled player to purchase")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not self.action_space.contains(action)
        ):
            raise ValueError(f"Action must be an integer in 0..{self.end_buying}")
        if not self.action_masks()[action]:
            raise ValueError(f"Illegal action: {self.action_names[action]}")
        previous = encode_observation(state, self.agent_player)
        if action != self.end_buying:
            game.buy_card(self.card_ids[action])
        else:
            game.end_buying()
        self._advance_opponent()
        observation = encode_observation(state, self.agent_player)
        reward = calculate_reward(
            previous, observation, self.reward_config,
            terminated=state.terminated, truncated=state.truncated,
            # Reward helpers use 0 for the agent; engine/info winners remain absolute.
            winner=None if state.winner is None else int(state.winner != self.agent_player),
        )
        self.opponent_step_counts[self.episode_opponent] += 1
        if self.curriculum is not None:
            self.curriculum_step_counts[self.curriculum_stage_index][self.episode_opponent] += 1
        return observation, reward, state.terminated, state.truncated, self._info(observation)

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        game = self._require_game()
        state = game.state
        agent = state.players[self.agent_player]
        observation = encode_observation(state, self.agent_player)
        if state.truncated:
            result = f"truncated ({state.end_reason}); no winner"
        elif state.terminated:
            result = "shared tie" if state.winner is None else f"player {state.winner} wins"
            result += f" ({state.end_reason})"
        else:
            result = "ongoing"
        return "\n".join([
            f"Turn {state.turn_number}; current player {state.current_player}",
            f"Agent hand: {agent.hand}; played: {agent.in_play}",
            f"Coins: {agent.coins}; Buys: {agent.buys}",
            f"Supply: {state.supply}",
            f"Available purchases: {[name for name, legal in zip(self.card_ids, self.action_masks()) if legal]}",
            f"Inferable VP: agent={int(observation[FEATURE_INDEX['agent_vp']])}, "
            f"opponent={int(observation[FEATURE_INDEX['opponent_vp']])}; "
            f"completed turns: {[player.turns_taken for player in state.players]}",
            f"Result: {result}",
        ])
