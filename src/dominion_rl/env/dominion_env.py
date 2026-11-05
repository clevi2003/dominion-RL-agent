"""Gymnasium interface: one step buys one card or explicitly ends buying."""

from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.random_agent import random_purchase
from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env.observations import CARD_IDS, FEATURE_INDEX, FEATURE_NAMES, encode_observation
from dominion_rl.env.rewards import calculate_reward
from dominion_rl.game import DominionGame

# V0 action IDs: Copper, Silver, Gold, Estate, Duchy, Province, End buying.
ACTION_NAMES: tuple[str, ...] = (*CARD_IDS, "end_buying")
END_BUYING: int = len(CARD_IDS)


class DominionEnv(gym.Env[np.ndarray, int]):
    """Player 0 purchases; a fixed policy handles player 1 outside the engine.

    The configured seed initializes Gymnasium's RNG on the first unseeded reset.
    Later reset() calls advance its stream; reset(seed=N) starts it over. Engine
    and policy seeds are drawn independently per episode and never exposed in info.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        game_config: GameConfig | None = None,
        reward_config: RewardConfig | None = None,
        opponent: Literal["big-money", "random"] = "big-money",
        render_mode: Literal["ansi"] | None = None,
    ) -> None:
        super().__init__()
        if opponent not in ("big-money", "random"):
            raise ValueError("opponent must be 'big-money' or 'random'")
        if render_mode not in (None, "ansi"):
            raise ValueError("Only render_mode='ansi' is supported")
        self.game_config = game_config if game_config is not None else GameConfig()
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.opponent = opponent
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(FEATURE_NAMES),), dtype=np.float32
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
        observation = encode_observation(self.game.state)
        return observation, self._info(observation)

    def action_masks(self) -> np.ndarray:
        """Fresh bool mask; End buying is legal only during an active episode."""
        game = self._require_game()
        mask = np.zeros(len(ACTION_NAMES), dtype=np.bool_)
        if not (game.state.terminated or game.state.truncated):
            legal = game.legal_purchases()
            mask[:-1] = [card_id in legal for card_id in CARD_IDS]
            mask[END_BUYING] = True
        return mask

    def _info(self, observation: np.ndarray) -> dict[str, Any]:
        state = self._require_game().state
        # Public/inferable summaries only: no backend state, RNGs, or private zones.
        return {
            "action_mask": self.action_masks(),
            "turn_number": state.turn_number,
            "current_player": state.current_player,
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
        if state.current_player != 0:
            raise RuntimeError("The environment expects a player 0 purchasing decision")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not self.action_space.contains(action)
        ):
            raise ValueError(f"Action must be an integer in 0..{END_BUYING}")
        if not self.action_masks()[action]:
            raise ValueError(f"Illegal action: {ACTION_NAMES[action]}")
        previous = encode_observation(state)
        if action != END_BUYING:
            game.buy_card(CARD_IDS[action])
        else:
            game.end_buying()
            while not (state.terminated or state.truncated) and state.current_player == 1:
                legal = game.legal_purchases()
                decision = (
                    random_purchase(legal, self._opponent_rng) if self.opponent == "random"
                    else big_money_purchase(legal, state.supply["province"])
                )
                if decision is None:
                    game.end_buying()
                else:
                    game.buy_card(decision)
        observation = encode_observation(state)
        reward = calculate_reward(
            previous, observation, self.reward_config,
            terminated=state.terminated, truncated=state.truncated, winner=state.winner,
        )
        return observation, reward, state.terminated, state.truncated, self._info(observation)

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        game = self._require_game()
        state = game.state
        agent = state.players[0]
        observation = encode_observation(state)
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
            f"Available purchases: {[name for name, legal in zip(CARD_IDS, self.action_masks()) if legal]}",
            f"Inferable VP: agent={int(observation[FEATURE_INDEX['agent_vp']])}, "
            f"opponent={int(observation[FEATURE_INDEX['opponent_vp']])}; "
            f"completed turns: {[player.turns_taken for player in state.players]}",
            f"Result: {result}",
        ])
