"""V3 tactical Action decisions with fixed purchasing and deterministic opponents."""

from collections import Counter
from dataclasses import replace

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dominion_rl.agents.action_rich import action_rich_purchase, buying_config
from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.engine_money import engine_money_purchase
from dominion_rl.agents.festival_money import festival_money_purchase
from dominion_rl.agents.militia_money import militia_money_purchase
from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env.observations import V3_FEATURE_NAMES, encode_observation
from dominion_rl.env.rewards import calculate_reward
from dominion_rl.game import DominionGame

ACTION_CARDS = ("village", "council_room", "festival", "militia", "moat")
ACTION_NAMES = (*ACTION_CARDS, "end_action")
ACTION_FEATURES = (*V3_FEATURE_NAMES, "actions_remaining")
END_ACTION = 5
EXPOSURE_FIELDS = ("action_decisions", "forced_end_decisions", "single_type_decisions",
                   "multiple_type_decisions", "voluntary_end_with_playable")
OPPONENTS = ("big-money", "engine-money", "festival-money", "militia-money")
METRICS = (
    "action_decisions", "forced_end_decisions", "single_type_decisions", "multiple_type_decisions",
    "more_than_one_legal_choice", "voluntary_end_with_playable", "unused_actions",
    "agent_turns", "multiple_action_turns", "draw_before_attack_pairs", "attack_before_draw_pairs",
    "militia_moat_choices", "buy_phase_coins", "province_opportunity_turns", "extra_buys_generated",
    "extra_buys_used", "purchases", "no_purchase_turns", "multiple_purchase_turns",
    *(f"plays_{card}" for card in ACTION_CARDS),
    *(f"purchases_{card}" for card in GameConfig(version="V3").cards),
)


def action_contract():
    return dict(version="V3", features=list(ACTION_FEATURES), actions=list(ACTION_NAMES))


def action_preprocessing():
    return dict(version=1, mode="raw", features=list(ACTION_FEATURES), divisors=[1.0] * len(ACTION_FEATURES))


def action_reward():
    return RewardConfig(mode="terminal_only", terminal_objective="vp_margin", gamma=1.0,
                        shaping_strength=0.0)


class DominionActionEnv(gym.Env):
    """One step is one Action choice, including explicitly logged forced Ends.

    Own purchasing is frozen action-rich-v1. All other decisions reuse engine
    rules, printed cards and fixed heuristics. Info/logs contain only own/public
    quantities. Opponent ownership is never accessed except by that opponent's
    private controller, and no ordered private zone is logged.
    """

    metadata = {}

    def __init__(self, game_config=None, reward_config=None, opponent="militia-money",
                 agent_player=0, *, record_decisions=False):
        super().__init__()
        self.game_config = game_config or GameConfig(version="V3")
        self.reward_config = reward_config or action_reward()
        if self.game_config.version != "V3":
            raise ValueError("Action policies require V3")
        if self.reward_config != action_reward():
            raise ValueError("Action study requires terminal-only VP margin, gamma=1, no shaping")
        if opponent not in OPPONENTS:
            raise ValueError("Action study requires a fixed deterministic opponent")
        if type(agent_player) is not int or agent_player not in (0, 1):
            raise ValueError("agent_player must be 0 or 1")
        if agent_player == 1 and self.game_config.max_turns < 2:
            raise ValueError("player 1 requires max_turns >=2")
        self.agent_player, self.opponent = agent_player, opponent
        self.episode_opponent = opponent
        self.record_decisions = record_decisions
        self.fixed_buying_controller = buying_config()
        self.feature_names, self.action_names = ACTION_FEATURES, ACTION_NAMES
        self.game_contract = action_contract()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, np.inf, (67,), np.float32)
        self.opponent_step_counts = {opponent: 0}
        self.opponent_reset_counts = {opponent: 0}
        self.curriculum = None  # Common training bookkeeping, not a curriculum.
        self.game = None

    def _require_game(self):
        if self.game is None:
            raise gym.error.ResetNeeded("Call reset() first")
        return self.game

    def _observation(self):
        game = self._require_game()
        return np.append(encode_observation(game.state, self.agent_player),
                         game.state.players[self.agent_player].actions).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is None and self._np_random is None:
            seed = self.game_config.seed
        super().reset(seed=seed)
        # Match purchasing env episode seeding; fixed controllers consume no RNG.
        engine_seed, _ = (int(v) for v in self.np_random.integers(0, 2**63-1, size=2))
        if self.game is None:
            self.game = DominionGame(replace(self.game_config, seed=engine_seed), automatic_actions=False)
        else:
            self.game.reset(seed=engine_seed)
        self.metrics = dict.fromkeys(METRICS, 0)
        self.decisions = []
        self.turn_actions = []
        self.opponent_reset_counts[self.opponent] += 1
        self._advance_opponent()
        return self._observation(), self._info()

    def action_masks(self):
        game = self._require_game()
        mask = np.zeros(6, dtype=np.bool_)
        state = game.state
        if not (state.terminated or state.truncated) and state.current_player == self.agent_player and game.phase == "action":
            mask[:5] = [card in game.legal_actions() for card in ACTION_CARDS]
            mask[END_ACTION] = True
        return mask

    def _buy_turn(self, own):
        game = self._require_game()
        state = game.state
        player = state.players[state.current_player]
        if own:
            self.metrics["agent_turns"] += 1
            self.metrics["unused_actions"] += player.actions
            self.metrics["multiple_action_turns"] += int(len(self.turn_actions) > 1)
            self.metrics["buy_phase_coins"] += player.coins
            self.metrics["extra_buys_generated"] += player.buys - 1
            self.metrics["province_opportunity_turns"] += int("province" in game.legal_purchases())
        purchases = 0
        while True:
            legal = game.legal_purchases()
            owned = Counter(player.deck + player.hand + player.discard + player.in_play)
            if own:
                choice = action_rich_purchase(legal, state.supply["province"], owned)
            elif self.opponent == "big-money":
                choice = big_money_purchase(legal, state.supply["province"])
            else:
                choose = {"engine-money": engine_money_purchase, "festival-money": festival_money_purchase,
                          "militia-money": militia_money_purchase}[self.opponent]
                choice = choose(legal, state.supply["province"], owned)
            if choice is None:
                break
            game.buy_card(choice)
            purchases += 1
            if own:
                self.metrics["purchases_" + choice] += 1
        if own:
            self.metrics["purchases"] += purchases
            self.metrics["extra_buys_used"] += max(purchases - 1, 0)
            self.metrics["no_purchase_turns"] += int(purchases == 0)
            self.metrics["multiple_purchase_turns"] += int(purchases > 1)
            self.turn_actions = []
        game.end_buying()

    def _advance_opponent(self):
        game = self._require_game()
        while not (game.state.terminated or game.state.truncated) and game.state.current_player != self.agent_player:
            while (card := game.automatic_action_choice()) is not None:
                game.play_action(card)
            game.end_action_phase()
            self._buy_turn(False)

    def _info(self):
        state = self._require_game().state
        return dict(action_mask=self.action_masks(), turn_number=state.turn_number,
                    current_player=state.current_player, agent_player=self.agent_player,
                    episode_opponent=self.opponent,
                    scores=tuple(self.game.score(i) for i in (self.agent_player, 1-self.agent_player)),
                    turns_taken=tuple(p.turns_taken for p in state.players), winner=state.winner,
                    end_reason=state.end_reason, action_metrics=dict(self.metrics),
                    **{key:self.metrics[key] for key in EXPOSURE_FIELDS})

    def step(self, action):
        game = self._require_game()
        if game.state.terminated or game.state.truncated:
            raise RuntimeError("Episode ended; reset before stepping")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not self.action_space.contains(action):
            raise ValueError("Action must be integer 0..5")
        mask = self.action_masks()
        if not mask[action]:
            raise ValueError(f"Illegal Action choice: {ACTION_NAMES[action]}")
        previous = self._observation()
        player = game.state.players[self.agent_player]
        types = int(mask[:5].sum())
        self.metrics["action_decisions"] += 1
        self.metrics["forced_end_decisions"] += int(types == 0)
        self.metrics["single_type_decisions"] += int(types == 1)
        self.metrics["multiple_type_decisions"] += int(types >= 2)
        self.metrics["more_than_one_legal_choice"] += int(types > 0)
        self.metrics["militia_moat_choices"] += int(mask[3] and mask[4])
        self.metrics["voluntary_end_with_playable"] += int(action == END_ACTION and types > 0)
        if self.record_decisions:
            # Similar-context analysis: hand counts are own; no opponent zones.
            self.decisions.append(dict(turn=player.turns_taken+1, actions=player.actions,
                coins=player.coins, buys=player.buys, playable_types=types,
                legal_types="|".join(c for c, ok in zip(ACTION_CARDS, mask[:5]) if ok),
                hand_counts="|".join(f"{c}:{player.hand.count(c)}" for c in self.game_config.cards),
                choice=ACTION_NAMES[action]))
        if action == END_ACTION:
            game.end_action_phase()
        else:
            card = ACTION_CARDS[action]
            draw_cards = ("village", "council_room", "moat")
            if card == "militia":
                self.metrics["draw_before_attack_pairs"] += sum(c in draw_cards for c in self.turn_actions)
            elif card in draw_cards:
                self.metrics["attack_before_draw_pairs"] += self.turn_actions.count("militia")
            game.play_action(card)
            self.turn_actions.append(card)
            self.metrics["plays_" + card] += 1
            if not game.legal_actions():
                game.end_action_phase()
        if game.phase == "buy":
            self._buy_turn(True)
            self._advance_opponent()
        observation = self._observation()
        state = game.state
        reward = calculate_reward(previous[:66], observation[:66], self.reward_config,
            terminated=state.terminated, truncated=state.truncated,
            winner=None if state.winner is None else int(state.winner != self.agent_player))
        self.opponent_step_counts[self.opponent] += 1
        return observation, reward, state.terminated, state.truncated, self._info()
