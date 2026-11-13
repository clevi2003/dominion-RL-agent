"""Two-player V0–V3 rules with default automatic or optional manual Action play."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from dominion_rl.game.cards import ActionCard, CARD_REGISTRY, CardId, TreasureCard, VictoryCard
from dominion_rl.game.state import GameState, PlayerState
from dominion_rl.agents.discard import militia_discards

if TYPE_CHECKING:
    from dominion_rl.config import GameConfig

# Counts after dealing starting decks. Starting Estates are separate from Supply.
INITIAL_SUPPLY: dict[CardId, int] = {
    "copper": 46, "silver": 40, "gold": 30,
    "estate": 8, "duchy": 8, "province": 8,
}
V1_INITIAL_SUPPLY = {**INITIAL_SUPPLY, "village": 10, "council_room": 10}
V2_INITIAL_SUPPLY = {**V1_INITIAL_SUPPLY, "festival": 10}
V3_INITIAL_SUPPLY = {**V2_INITIAL_SUPPLY, "militia": 10, "moat": 10}
STARTING_DECK: tuple[CardId, ...] = ("copper",) * 7 + ("estate",) * 3


class DominionGame:
    """Own state and shuffle RNG; purchasing policies stay outside the engine.

    Construction and reset prepare player 0's first purchasing decision, including
    automatic Actions (V1/V2/V3) then Treasure play. reset() reuses the configured seed; reset(seed=N)
    overrides it for that reset only. A configured seed of None uses fresh entropy.
    automatic_actions=False pauses before Actions; end_action_phase() plays all
    Treasures and enters buying. Phase lives outside the historical GameState
    schema, keeping archived full-state traces and purchasing observations intact.
    """

    def __init__(self, config: GameConfig | None = None, *, automatic_actions: bool = True) -> None:
        if config is None:
            from dominion_rl.config import GameConfig

            config = GameConfig()
        if type(automatic_actions) is not bool:
            raise ValueError("automatic_actions must be bool")
        self.automatic_actions = automatic_actions
        self.config = config
        self.reset()

    def reset(self, seed: int | None = None) -> GameState:
        """Replace the entire episode state and reset the local shuffle stream."""
        chosen_seed = self.config.seed if seed is None else seed
        if chosen_seed is not None and (type(chosen_seed) is not int or chosen_seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        self._rng = Random(chosen_seed)
        initial = {"V0": INITIAL_SUPPLY, "V1": V1_INITIAL_SUPPLY, "V2": V2_INITIAL_SUPPLY, "V3": V3_INITIAL_SUPPLY}[self.config.version]
        self.state = GameState(
            supply={card_id: initial[card_id] for card_id in self.config.cards},
            version=self.config.version,
        )
        for player in self.state.players:
            player.deck = list(STARTING_DECK)
            self._rng.shuffle(player.deck)
            self.draw(player, 5)
        # Public Militia summaries only: no discarded identities or private draws.
        self.militia_resolutions: list[dict[str, int | bool]] = []
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
        self.phase = "action"
        if not self.automatic_actions:
            return
        while (card_id := self.automatic_action_choice()) is not None:
            self.play_action(card_id)
        self.end_action_phase()

    def legal_actions(self) -> list[CardId]:
        """Distinct playable types; duplicate copies remain separate future plays."""
        if self.state.terminated or self.state.truncated or self.phase != "action":
            return []
        player = self.state.players[self.state.current_player]
        if player.actions < 1:
            return []
        return [card for card in self.config.cards
                if card in player.hand and isinstance(CARD_REGISTRY[card], ActionCard)]

    def automatic_action_choice(self) -> CardId | None:
        """The historical priority, also used by fixed manual-mode controllers."""
        priority = {"V0": (), "V1": ("village", "council_room"),
                    "V2": ("village", "festival", "council_room"),
                    "V3": ("village", "festival", "council_room", "moat", "militia")}
        legal = self.legal_actions()
        return next((card for card in priority[self.config.version] if card in legal), None)

    def end_action_phase(self) -> None:
        """Voluntarily stop Actions, play Treasures in hand order, enter Buy."""
        self._require_active()
        if self.phase != "action":
            raise ValueError("End Action requires the Action phase")
        player = self.state.players[self.state.current_player]
        remaining_hand: list[CardId] = []
        for card_id in player.hand:
            card = CARD_REGISTRY[card_id]
            if isinstance(card, TreasureCard):
                player.in_play.append(card_id)
                card.play(self, player)
            else:
                remaining_hand.append(card_id)
        player.hand = remaining_hand
        self.phase = "buy"

    def play_action(self, card_id: CardId) -> None:
        """Spend one Action and move hand -> in_play before printed effects.

        Automatic turns and manual controllers share these exact card effects;
        purchased cards in discard are never eligible for immediate play.
        """
        self._require_active()
        player = self.state.players[self.state.current_player]
        if (self.phase != "action" or self.config.version not in ("V1", "V2", "V3") or card_id not in self.config.cards
                or player.actions < 1 or card_id not in player.hand
                or not isinstance(CARD_REGISTRY.get(card_id), ActionCard)):
            raise ValueError(f"Illegal Action play: {card_id!r}")
        player.actions -= 1
        player.hand.remove(card_id)
        player.in_play.append(card_id)
        CARD_REGISTRY[card_id].play(self, player)

    def resolve_militia(self, attacker: PlayerState) -> None:
        """React with a held Moat, otherwise discard down to three; never advance.

        Only the active player can attack, and every defender is a different
        player. Selection reads that defender's hand only. Discards preserve
        ownership. Public summaries support demonstrations without exposing cards.
        """
        self._require_active()
        if self.config.version != "V3" or attacker is not self.state.players[self.state.current_player]:
            raise ValueError("Militia must be resolved by the active V3 player")
        for index, defender in enumerate(self.state.players):
            if defender is attacker:
                continue
            protected = CARD_REGISTRY["moat"].reveal_against_militia(defender)
            discarded = [] if protected else militia_discards(defender.hand)
            for card in discarded:
                defender.hand.remove(card)
                defender.discard.append(card)
            self.militia_resolutions.append(dict(attacker=self.state.current_player,
                defender=index, moat_revealed=protected, discard_count=len(discarded)))

    def legal_purchases(self) -> list[CardId]:
        """Return affordable nonempty configured piles, in configuration order."""
        if self.state.terminated or self.state.truncated or self.phase != "buy":
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
        No next player's Action/Treasure phase runs after termination or truncation.
        """
        self._require_active()
        if self.phase != "buy":
            raise ValueError("End buying requires the Buy phase")
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
