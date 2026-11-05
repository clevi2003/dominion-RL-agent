"""Fixed V0 economic baseline using only legal purchases and public Supply."""

from typing import Collection

from dominion_rl.game.cards import CardId


def big_money_purchase(
    legal_purchases: Collection[CardId], provinces_remaining: int
) -> CardId | None:
    """Province first; Duchy at <=4 Provinces, Estate at <=2; then Gold/Silver.

    End buying if none of these choices is legal; never buy Copper. Late Victory
    purchases take precedence over economic investment. Thresholds are fixed,
    transparent heuristics, not tuned or learned experimental results.
    """
    if "province" in legal_purchases:
        return "province"
    if provinces_remaining <= 4 and "duchy" in legal_purchases:
        return "duchy"
    if provinces_remaining <= 2 and "estate" in legal_purchases:
        return "estate"
    for card_id in ("gold", "silver"):
        if card_id in legal_purchases:
            return card_id
    return None
