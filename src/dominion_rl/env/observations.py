"""Fixed V0 features containing only player 0's legally available information.

Opponent totals are computed across zones but are publicly inferable in V0:
everyone starts with known cards and all purchases are public. No hidden gains or
trashing exist. This shortcut MUST be revisited before supporting Kingdom cards.
Neither hidden ordering nor the opponent's distribution across zones is encoded.
"""

from collections import Counter

import numpy as np

from dominion_rl.game.cards import CARD_REGISTRY, VictoryCard
from dominion_rl.game.state import GameState, PlayerState

# Canonical registry order, independent of configuration ordering.
CARD_IDS: tuple[str, ...] = tuple(CARD_REGISTRY)
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


def _ownership(player: PlayerState) -> Counter[str]:
    return Counter(player.deck + player.hand + player.discard + player.in_play)


def _inferable_vp(ownership: Counter[str]) -> int:
    return sum(
        count * card.victory_points
        for card_id, count in ownership.items()
        if isinstance(card := CARD_REGISTRY[card_id], VictoryCard)
    )


def encode_observation(state: GameState) -> np.ndarray:
    """Return a fresh float32 vector in FEATURE_NAMES order; raw counts, no scaling."""
    agent, opponent = state.players
    hand, in_play = Counter(agent.hand), Counter(agent.in_play)
    owned, opponent_owned = _ownership(agent), _ownership(opponent)
    return np.array(
        [
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
        ],
        dtype=np.float32,
    )
