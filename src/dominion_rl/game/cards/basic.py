"""The six first-edition Treasure and Victory cards used in V0."""

from dataclasses import dataclass, field

from dominion_rl.game.cards.base import TreasureCard, VictoryCard


@dataclass(frozen=True, slots=True)
class Copper(TreasureCard):
    name: str = field(default="Copper", init=False)
    cost: int = field(default=0, init=False)
    coin_value: int = field(default=1, init=False)


@dataclass(frozen=True, slots=True)
class Silver(TreasureCard):
    name: str = field(default="Silver", init=False)
    cost: int = field(default=3, init=False)
    coin_value: int = field(default=2, init=False)


@dataclass(frozen=True, slots=True)
class Gold(TreasureCard):
    name: str = field(default="Gold", init=False)
    cost: int = field(default=6, init=False)
    coin_value: int = field(default=3, init=False)


@dataclass(frozen=True, slots=True)
class Estate(VictoryCard):
    name: str = field(default="Estate", init=False)
    cost: int = field(default=2, init=False)
    victory_points: int = field(default=1, init=False)


@dataclass(frozen=True, slots=True)
class Duchy(VictoryCard):
    name: str = field(default="Duchy", init=False)
    cost: int = field(default=5, init=False)
    victory_points: int = field(default=3, init=False)


@dataclass(frozen=True, slots=True)
class Province(VictoryCard):
    name: str = field(default="Province", init=False)
    cost: int = field(default=8, init=False)
    victory_points: int = field(default=6, init=False)
