"""First-edition V1/V2/V3 Actions; movement/action spending belong to the engine."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from dominion_rl.game.cards.base import ActionCard

if TYPE_CHECKING:
    from dominion_rl.game.engine import DominionGame
    from dominion_rl.game.state import PlayerState


@dataclass(frozen=True, slots=True)
class Village(ActionCard):
    name: str = field(default='Village', init=False)
    cost: int = field(default=3, init=False)

    def play(self, game: 'DominionGame', player: 'PlayerState') -> None:
        game.draw(player, 1)
        player.actions += 2


@dataclass(frozen=True, slots=True)
class CouncilRoom(ActionCard):
    name: str = field(default='Council Room', init=False)
    cost: int = field(default=5, init=False)

    def play(self, game: 'DominionGame', player: 'PlayerState') -> None:
        game.draw(player, 4)
        player.buys += 1
        for other in game.state.players:
            if other is not player:
                game.draw(other, 1)  # Immediate private draw, retained until their turn.


@dataclass(frozen=True, slots=True)
class Festival(ActionCard):
    name: str = field(default='Festival', init=False)
    cost: int = field(default=5, init=False)

    def play(self, game: 'DominionGame', player: 'PlayerState') -> None:
        player.actions += 2
        player.buys += 1
        player.coins += 2


@dataclass(frozen=True, slots=True)
class Militia(ActionCard):
    name: str = field(default='Militia', init=False)
    cost: int = field(default=4, init=False)
    card_types: ClassVar[frozenset[str]] = frozenset({'action', 'attack'})

    def play(self, game: 'DominionGame', player: 'PlayerState') -> None:
        player.coins += 2
        game.resolve_militia(player)


@dataclass(frozen=True, slots=True)
class Moat(ActionCard):
    name: str = field(default='Moat', init=False)
    cost: int = field(default=2, init=False)
    card_types: ClassVar[frozenset[str]] = frozenset({'action', 'reaction'})

    def play(self, game: 'DominionGame', player: 'PlayerState') -> None:
        game.draw(player, 2)

    def reveal_against_militia(self, player: 'PlayerState') -> bool:
        """Always reveal from hand; revealing changes no zones or resources."""
        return 'moat' in player.hand
