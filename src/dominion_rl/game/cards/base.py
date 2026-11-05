"""Immutable card definitions; physical copies are represented by string IDs."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeAlias

if TYPE_CHECKING:
    from dominion_rl.game.engine import DominionGame
    from dominion_rl.game.state import PlayerState

CardId: TypeAlias = str


@dataclass(frozen=True, slots=True)
class Card:
    """Shared printed properties of a card type."""

    name: str
    cost: int
    card_types: ClassVar[frozenset[str]] = frozenset()


@dataclass(frozen=True, slots=True)
class TreasureCard(Card):
    """A Treasure adds its printed coin value when played."""

    coin_value: int
    card_types: ClassVar[frozenset[str]] = frozenset({"treasure"})

    def play(self, game: "DominionGame", player: "PlayerState") -> None:
        """Apply the coin effect; the engine handles moving the card's ID."""
        player.coins += self.coin_value


@dataclass(frozen=True, slots=True)
class VictoryCard(Card):
    """A Victory card has a fixed printed VP value in V0."""

    victory_points: int
    card_types: ClassVar[frozenset[str]] = frozenset({"victory"})


@dataclass(frozen=True, slots=True)
class ActionCard(Card):
    """Base for future concrete Actions with their own play(game, player) effects.

    Introduce that method with the engine and first Action implementations.
    A future Moat can add a reaction method without another inheritance layer.
    """

    card_types: ClassVar[frozenset[str]] = frozenset({"action"})
