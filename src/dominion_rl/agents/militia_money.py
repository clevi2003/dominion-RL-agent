"""Untuned V3 fixed opponent: at most two early Militias, then Big Money."""
from typing import Collection, Mapping

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.game.cards import CardId


def militia_money_purchase(
    legal_purchases: Collection[CardId], provinces_remaining: int, owned: Mapping[CardId, int],
) -> CardId | None:
    """Province first; above four Provinces buy up to two Militias; otherwise BM."""
    if 'province' in legal_purchases:
        return 'province'
    if provinces_remaining > 4 and owned.get('militia', 0) < 2 and 'militia' in legal_purchases:
        return 'militia'
    return big_money_purchase(legal_purchases, provinces_remaining)
