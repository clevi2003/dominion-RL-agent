"""Mutable full-information state; constructors do not set up or advance a game."""

from dataclasses import dataclass, field
from typing import Literal

from dominion_rl.game.cards.base import CardId

EndReason = Literal["province_empty", "three_piles_empty", "turn_limit"]


@dataclass
class PlayerState:
    """Card zones contain IDs, never per-copy card definition objects.

    The draw deck's last element is its top (drawing uses pop()).
    turns_taken counts completed individual turns for the official VP tiebreak.
    Empty zones are intentional: dealing, shuffling and drawing belong to setup.
    """

    deck: list[CardId] = field(default_factory=list)
    hand: list[CardId] = field(default_factory=list)
    discard: list[CardId] = field(default_factory=list)
    in_play: list[CardId] = field(default_factory=list)
    actions: int = 1
    buys: int = 1
    coins: int = 0
    turns_taken: int = 0


@dataclass
class GameState:
    """Two-player simulator state, including hidden information.

    current_player is a zero-based index. turn_number starts at one and counts
    individual player turns, not rounds. During play it equals total completed
    turns plus one; after an ending it stays at the just-completed turn's number.
    DominionGame populates Supply and starting decks; bare state defaults remain
    empty. A turn cap is a truncation with no declared winner.
    winner is a player index for a decided normal ending, otherwise None (a tie,
    unfinished game, or truncation). The engine maintains these outcome fields.
    This backend state must not be exposed directly as an RL observation.
    """

    players: tuple[PlayerState, PlayerState] = field(
        default_factory=lambda: (PlayerState(), PlayerState())
    )
    supply: dict[CardId, int] = field(default_factory=dict)
    trash: list[CardId] = field(default_factory=list)
    current_player: int = 0
    turn_number: int = 1
    terminated: bool = False
    truncated: bool = False
    end_reason: EndReason | None = None
    winner: int | None = None

    def __post_init__(self) -> None:
        if len(self.players) != 2:
            raise ValueError("GameState requires exactly two players")
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
