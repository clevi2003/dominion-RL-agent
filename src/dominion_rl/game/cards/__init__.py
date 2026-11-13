"""Shared card definitions indexed by stable, lowercase card IDs."""

from dominion_rl.game.cards.base import (
    ActionCard,
    Card,
    CardId,
    TreasureCard,
    VictoryCard,
)
from dominion_rl.game.cards.actions import CouncilRoom, Festival, Militia, Moat, Village
from dominion_rl.game.cards.basic import Copper, Duchy, Estate, Gold, Province, Silver

V0_CARD_IDS: tuple[CardId, ...] = ("copper", "silver", "gold", "estate", "duchy", "province")
V1_CARD_IDS: tuple[CardId, ...] = (*V0_CARD_IDS, "village", "council_room")

V2_CARD_IDS: tuple[CardId, ...] = (*V1_CARD_IDS, "festival")

V3_CARD_IDS: tuple[CardId, ...] = (*V2_CARD_IDS, "militia", "moat")


def card_ids_for_version(version: str = "V0") -> tuple[CardId, ...]:
    if version == "V0":
        return V0_CARD_IDS
    if version == "V1":
        return V1_CARD_IDS
    if version == "V2":
        return V2_CARD_IDS
    if version == "V3":
        return V3_CARD_IDS
    raise ValueError("Game version must be V0, V1, V2 or V3")


# One definition per card type, reused by every physical copy in state zones.
# Keep registry order stable: future observation/action encoders need fixed IDs.
CARD_REGISTRY: dict[CardId, Card] = {
    "copper": Copper(),
    "silver": Silver(),
    "gold": Gold(),
    "estate": Estate(),
    "duchy": Duchy(),
    "province": Province(),
    "village": Village(),
    "council_room": CouncilRoom(),
    "festival": Festival(),
    "militia": Militia(),
    "moat": Moat(),
}

__all__ = [
    "CouncilRoom", "Festival", "Militia", "Moat", "Village", "V0_CARD_IDS", "V1_CARD_IDS", "V2_CARD_IDS", "V3_CARD_IDS", "card_ids_for_version",
    "ActionCard", "CARD_REGISTRY", "Card", "CardId", "Copper", "Duchy",
    "Estate", "Gold", "Province", "Silver", "TreasureCard", "VictoryCard",
]
