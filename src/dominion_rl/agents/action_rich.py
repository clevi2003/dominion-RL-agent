"""Fixed, bounded deck construction to expose tactical Action decisions."""

from copy import deepcopy
from dominion_rl.agents.big_money import big_money_purchase

# Versioned rules are saved inside every Action checkpoint. No experiment tuning.
BUYING_CONFIG = {
    "name": "action-rich-v1",
    "province_first": True,
    "stop_actions_at_provinces": 4,
    "economy_floor": {"silver": 3, "gold": 2},
    "limits": {"village": 2, "council_room": 2, "festival": 1, "militia": 1, "moat": 1},
    "action_priority": ["village", "council_room", "festival", "militia", "moat"],
    "village_requires_other_action": True,
    "fallback": "big-money",
}


def buying_config():
    return deepcopy(BUYING_CONFIG)


def action_rich_purchase(legal, provinces, owned):
    """Only legal choices, public Supply and unordered own ownership are read.

    Province first; with <=4 Provinces use Big Money's endgame. Otherwise buy
    the first two Gold when affordable, first three Silver before adding Actions,
    then bounded Actions in saved priority. Village requires another Action.
    Economic fallback continues buying Gold/Silver; no Copper or unlimited Actions.
    """
    if "province" in legal:
        return "province"
    if provinces <= BUYING_CONFIG["stop_actions_at_provinces"]:
        return big_money_purchase(legal, provinces)
    for card in ("gold", "silver"):
        if card in legal and owned.get(card, 0) < BUYING_CONFIG["economy_floor"][card]:
            return card
    for card in BUYING_CONFIG["action_priority"]:
        if card == "village" and not any(owned.get(c, 0) for c in ("council_room", "festival", "militia", "moat")):
            continue
        if card in legal and owned.get(card, 0) < BUYING_CONFIG["limits"][card]:
            return card
    return big_money_purchase(legal, provinces)
