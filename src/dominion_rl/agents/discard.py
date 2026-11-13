"""Untuned Militia discards: only the defending player's actual hand is used."""
from collections.abc import Sequence

# Discard first -> keep longest. Victory cards have no purchasing value; Copper
# is weaker than Silver/Gold. Actions use this fixed, context-free order rather
# than guessing future draws. Stable sorting breaks equal-card ties left to right.
MILITIA_DISCARD_PRIORITY = (
    'estate', 'duchy', 'province', 'copper', 'moat', 'militia', 'council_room',
    'village', 'festival', 'silver', 'gold',
)


def militia_discards(hand: Sequence[str]) -> list[str]:
    """Choose copies to discard down to three, without modifying the hand.

    Kept separate from the attack rule so a later choice policy can replace this
    function. No opponent state, deck order or future draws are inputs.
    """
    return sorted(hand, key=MILITIA_DISCARD_PRIORITY.index)[:max(0, len(hand) - 3)]
