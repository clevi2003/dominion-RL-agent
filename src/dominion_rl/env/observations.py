"""Versioned features containing only the controlled player's legally available information.

Opponent totals remain publicly inferable through V3: everyone starts with
known cards, purchases are public, and drawing/discarding never changes ownership.
Militia selects from the defender's private hand internally; that selection and
the opponent's hand are not encoded. A Moat reveal adds no observation feature.
Revisit this shortcut before introducing hidden gains or trashing.
Neither hidden ordering nor the opponent's distribution across zones is encoded.
"""

from collections import Counter

import numpy as np

from dominion_rl.game.cards import CARD_REGISTRY, V0_CARD_IDS, V1_CARD_IDS, V2_CARD_IDS, V3_CARD_IDS, VictoryCard
from dominion_rl.game.state import GameState, PlayerState

# Canonical registry order, independent of configuration ordering.
CARD_IDS: tuple[str, ...] = V0_CARD_IDS  # Historical public constant stays six cards.
FEATURE_NAMES: tuple[str, ...] = (
    "coins", "buys",
    *(f"hand_{card_id}" for card_id in CARD_IDS),
    *(f"in_play_{card_id}" for card_id in CARD_IDS),
    *(f"owned_{card_id}" for card_id in CARD_IDS),
    *(f"opponent_owned_{card_id}" for card_id in CARD_IDS),
    *(f"supply_{card_id}" for card_id in CARD_IDS),
    "deck_size", "discard_size", "agent_vp", "opponent_vp", "turn_number",
    "agent_turns_taken", "opponent_turns_taken", "provinces_remaining",
    "empty_supply_piles",
)
FEATURE_INDEX: dict[str, int] = {name: index for index, name in enumerate(FEATURE_NAMES)}
# Append all V1 extras: indices 0..40, including VP/progress, remain unchanged.
V1_FEATURE_NAMES: tuple[str, ...] = (*FEATURE_NAMES, *(
    f"{prefix}_{card}" for prefix in ("hand", "in_play", "owned", "opponent_owned", "supply")
    for card in V1_CARD_IDS[len(CARD_IDS):]))
V1_FEATURE_INDEX = {name: index for index, name in enumerate(V1_FEATURE_NAMES)}

V2_FEATURE_NAMES: tuple[str, ...] = (*V1_FEATURE_NAMES, *(
    f"{prefix}_{card}" for prefix in ("hand", "in_play", "owned", "opponent_owned", "supply")
    for card in V2_CARD_IDS[len(V1_CARD_IDS):]))
V2_FEATURE_INDEX = {name: index for index, name in enumerate(V2_FEATURE_NAMES)}

V3_FEATURE_NAMES: tuple[str, ...] = (*V2_FEATURE_NAMES, *(
    f"{prefix}_{card}" for prefix in ("hand", "in_play", "owned", "opponent_owned", "supply")
    for card in V3_CARD_IDS[len(V2_CARD_IDS):]))
V3_FEATURE_INDEX = {name: index for index, name in enumerate(V3_FEATURE_NAMES)}


def feature_names(version: str = "V0") -> tuple[str, ...]:
    if version == "V0":
        return FEATURE_NAMES
    if version == "V1":
        return V1_FEATURE_NAMES
    if version == "V2":
        return V2_FEATURE_NAMES
    if version == "V3":
        return V3_FEATURE_NAMES
    raise ValueError("Game version must be V0, V1, V2 or V3")


def _ownership(player: PlayerState) -> Counter[str]:
    return Counter(player.deck + player.hand + player.discard + player.in_play)


def _inferable_vp(ownership: Counter[str]) -> int:
    return sum(
        count * card.victory_points
        for card_id, count in ownership.items()
        if isinstance(card := CARD_REGISTRY[card_id], VictoryCard)
    )


def encode_observation(state: GameState, agent_player: int = 0) -> np.ndarray:
    """Raw float32; each version appends features without moving older indices."""
    if type(agent_player) is not int or agent_player not in (0, 1):
        raise ValueError("agent_player must be 0 or 1")
    agent, opponent = state.players[agent_player], state.players[1 - agent_player]
    hand, in_play = Counter(agent.hand), Counter(agent.in_play)
    owned, opponent_owned = _ownership(agent), _ownership(opponent)
    values = [
        agent.coins, agent.buys,
        *(hand[card_id] for card_id in CARD_IDS),
        *(in_play[card_id] for card_id in CARD_IDS),
        *(owned[card_id] for card_id in CARD_IDS),
        *(opponent_owned[card_id] for card_id in CARD_IDS),
        *(state.supply[card_id] for card_id in CARD_IDS),
        len(agent.deck), len(agent.discard), _inferable_vp(owned),
        _inferable_vp(opponent_owned), state.turn_number,
        agent.turns_taken, opponent.turns_taken, state.supply["province"],
        sum(count == 0 for count in state.supply.values()),
    ]
    feature_names(state.version)  # Reject unsupported versions instead of guessing.
    if state.version in ("V1", "V2", "V3"):
        for counts in (hand, in_play, owned, opponent_owned, state.supply):
            values.extend(counts[card] for card in V1_CARD_IDS[len(CARD_IDS):])
    if state.version in ("V2", "V3"):
        for counts in (hand, in_play, owned, opponent_owned, state.supply):
            values.extend(counts[card] for card in V2_CARD_IDS[len(V1_CARD_IDS):])
    if state.version == "V3":
        for counts in (hand, in_play, owned, opponent_owned, state.supply):
            values.extend(counts[card] for card in V3_CARD_IDS[len(V2_CARD_IDS):])
    return np.array(values, dtype=np.float32)
