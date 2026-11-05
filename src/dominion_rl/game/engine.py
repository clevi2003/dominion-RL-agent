"""Two-player V0 rules, advancing one purchasing decision or turn at a time."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from dominion_rl.game.cards import CARD_REGISTRY, CardId, TreasureCard, VictoryCard
from dominion_rl.game.state import GameState, PlayerState

if TYPE_CHECKING:
    from dominion_rl.config import GameConfig

# Counts after dealing starting decks. Starting Estates are separate from Supply.
INITIAL_SUPPLY: dict[CardId, int] = {
    "copper": 46, "silver": 40, "gold": 30,
    "estate": 8, "duchy": 8, "province": 8,
}
STARTING_DECK: tuple[CardId, ...] = ("copper",) * 7 + ("estate",) * 3


class DominionGame:
    """Own mutable state and a local shuffle RNG; never run a policy internally.

    Construction and reset prepare player 0's first purchasing decision, including
    automatic Treasure play. reset() reuses the configured seed; reset(seed=N)
    overrides it for that reset only. A configured seed of None uses fresh entropy.
    """

    def __init__(self, config: GameConfig | None = None) -> None:
        if config is None:
            from dominion_rl.config import GameConfig

            config = GameConfig()
        self.config = config
        self.reset()

    def reset(self, seed: int | None = None) -> GameState:
        """Replace the entire episode state and reset the local shuffle stream."""
        chosen_seed = self.config.seed if seed is None else seed
        if chosen_seed is not None and (type(chosen_seed) is not int or chosen_seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        self._rng = Random(chosen_seed)
        self.state = GameState(
            supply={card_id: INITIAL_SUPPLY[card_id] for card_id in self.config.cards}
        )
        for player in self.state.players:
            player.deck = list(STARTING_DECK)
            self._rng.shuffle(player.deck)
            self.draw(player, 5)
        self._start_turn()
        return self.state

    def _require_active(self) -> None:
        if self.state.terminated or self.state.truncated:
            raise RuntimeError("Game has ended; reset before making further transitions")

    def draw(self, player: PlayerState, count: int) -> list[CardId]:
        """Move up to count cards into hand, reshuffling discard only as needed."""
        self._require_active()
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        drawn: list[CardId] = []
        for _ in range(count):
            if not player.deck:
                if not player.discard:
                    break
                player.deck.extend(player.discard)
                player.discard.clear()
                self._rng.shuffle(player.deck)
            card_id = player.deck.pop()
            player.hand.append(card_id)
            drawn.append(card_id)
        return drawn

    def _start_turn(self) -> None:
        player = self.state.players[self.state.current_player]
        player.actions, player.buys, player.coins = 1, 1, 0
        remaining_hand: list[CardId] = []
        for card_id in player.hand:
            card = CARD_REGISTRY[card_id]
            if isinstance(card, TreasureCard):
                player.in_play.append(card_id)
                card.play(self, player)
            else:
                remaining_hand.append(card_id)
        player.hand = remaining_hand

    def legal_purchases(self) -> list[CardId]:
        """Return affordable nonempty configured piles, in configuration order."""
        if self.state.terminated or self.state.truncated:
            return []
        player = self.state.players[self.state.current_player]
        if player.buys < 1:
            return []
        return [
            card_id for card_id in self.config.cards
            if self.state.supply.get(card_id, 0) > 0
            and CARD_REGISTRY[card_id].cost <= player.coins
        ]

    def buy_card(self, card_id: CardId) -> None:
        """Apply one legal purchase without cleanup, drawing, or auto-ending."""
        self._require_active()
        if card_id not in self.legal_purchases():
            raise ValueError(f"Illegal purchase: {card_id!r} (check Supply, coins, and Buys)")
        player = self.state.players[self.state.current_player]
        player.coins -= CARD_REGISTRY[card_id].cost
        player.buys -= 1
        self.state.supply[card_id] -= 1
        player.discard.append(card_id)

    def end_buying(self) -> None:
        """Clean up, draw five, complete this turn, then end or switch players.

        Normal end conditions take precedence over the total completed-turn cap.
        No next player's Treasures are played after termination or truncation.
        """
        self._require_active()
        player = self.state.players[self.state.current_player]
        player.discard.extend(player.hand)
        player.hand.clear()
        player.discard.extend(player.in_play)
        player.in_play.clear()
        player.actions, player.buys, player.coins = 0, 0, 0
        self.draw(player, 5)
        player.turns_taken += 1

        if self.state.supply["province"] == 0:
            self.state.end_reason = "province_empty"
        elif sum(count == 0 for count in self.state.supply.values()) >= 3:
            self.state.end_reason = "three_piles_empty"

        if self.state.end_reason is not None:
            self.state.terminated = True
            # Larger score wins; at equal VP fewer completed turns wins.
            results = [
                (self.score(index), -state.turns_taken)
                for index, state in enumerate(self.state.players)
            ]
            if results[0] != results[1]:
                self.state.winner = 0 if results[0] > results[1] else 1
            return

        if sum(state.turns_taken for state in self.state.players) >= self.config.max_turns:
            self.state.truncated = True
            self.state.end_reason = "turn_limit"
            return

        self.state.current_player = 1 - self.state.current_player
        self.state.turn_number += 1
        self._start_turn()

    def score(self, player_index: int) -> int:
        """Score every owned card zone, excluding Supply and trash."""
        if player_index not in (0, 1):
            raise ValueError("player_index must be 0 or 1")
        player = self.state.players[player_index]
        return sum(
            card.victory_points
            for zone in (player.deck, player.hand, player.discard, player.in_play)
            for card_id in zone
            if isinstance(card := CARD_REGISTRY[card_id], VictoryCard)
        )
