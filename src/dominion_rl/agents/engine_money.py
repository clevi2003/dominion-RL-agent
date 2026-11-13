"""Untuned V1 reference: a small Council Room engine followed by Big Money."""
from typing import Collection, Mapping

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.game.cards import CardId


def engine_money_purchase(
    legal_purchases: Collection[CardId], provinces_remaining: int, owned: Mapping[CardId, int],
) -> CardId | None:
    """Province first. Above 4 Provinces, acquire <=2 Rooms then <=1 Village.

    Village needs an owned Council Room. Otherwise use unchanged Big Money,
    including its late Duchy/Estate thresholds. Caps count all owned zones.
    """
    if "province" in legal_purchases:
        return "province"
    if provinces_remaining > 4:
        if owned.get("council_room", 0) < 2 and "council_room" in legal_purchases:
            return "council_room"
        if owned.get("council_room", 0) > 0 and owned.get("village", 0) < 1 and "village" in legal_purchases:
            return "village"
    return big_money_purchase(legal_purchases, provinces_remaining)
