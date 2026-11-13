"""Untuned V2 comparison: at most two Festivals and two Council Rooms."""
from typing import Collection, Mapping

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.game.cards import CardId


def festival_money_purchase(
    legal_purchases: Collection[CardId], provinces_remaining: int, owned: Mapping[CardId, int],
) -> CardId | None:
    """Province first; above four Provinces, Festival then Room, each capped at two.

    Otherwise follow unchanged Big Money, including its late victory thresholds.
    Ownership includes every zone. This fixed rule has not been performance tuned.
    """
    if "province" in legal_purchases:
        return "province"
    if provinces_remaining > 4:
        for card in ("festival", "council_room"):
            if owned.get(card, 0) < 2 and card in legal_purchases:
                return card
    return big_money_purchase(legal_purchases, provinces_remaining)
