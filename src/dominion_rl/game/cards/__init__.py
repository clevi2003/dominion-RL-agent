"""Shared card definitions indexed by stable, lowercase card IDs."""

from dominion_rl.game.cards.base import (
    ActionCard,
    Card,
    CardId,
    TreasureCard,
    VictoryCard,
)
from dominion_rl.game.cards.basic import Copper, Duchy, Estate, Gold, Province, Silver

# One definition per card type, reused by every physical copy in state zones.
# Keep registry order stable: future observation/action encoders need fixed IDs.
CARD_REGISTRY: dict[CardId, Card] = {
    "copper": Copper(),
    "silver": Silver(),
    "gold": Gold(),
    "estate": Estate(),
    "duchy": Duchy(),
    "province": Province(),
}

__all__ = [
    "ActionCard", "CARD_REGISTRY", "Card", "CardId", "Copper", "Duchy",
    "Estate", "Gold", "Province", "Silver", "TreasureCard", "VictoryCard",
]
