"""Uniform random purchase-or-end policy, independent of the game shuffle RNG."""

from random import Random
from typing import Sequence

from dominion_rl.game.cards import CardId


def random_purchase(legal_purchases: Sequence[CardId], rng: Random) -> CardId | None:
    """Choose uniformly among legal purchases and None (end buying)."""
    return rng.choice([*legal_purchases, None])
