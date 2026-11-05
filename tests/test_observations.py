from copy import deepcopy

import numpy as np

from dominion_rl.env.observations import CARD_IDS, FEATURE_INDEX, FEATURE_NAMES, encode_observation
from dominion_rl.game import DominionGame, GameState, PlayerState


def test_documented_feature_order_and_exact_constructed_values():
    assert CARD_IDS == ("copper", "silver", "gold", "estate", "duchy", "province")
    assert FEATURE_NAMES[:2] == ("coins", "buys")
    for offset, prefix in ((2, "hand"), (8, "in_play"), (14, "owned"),
                           (20, "opponent_owned"), (26, "supply")):
        assert FEATURE_NAMES[offset:offset + 6] == tuple(f"{prefix}_{card}" for card in CARD_IDS)
    assert FEATURE_NAMES[32:] == (
        "deck_size", "discard_size", "agent_vp", "opponent_vp", "turn_number",
        "agent_turns_taken", "opponent_turns_taken", "provinces_remaining", "empty_supply_piles",
    )
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 41
    state = GameState(
        players=(
            PlayerState(deck=["gold", "estate"], hand=["duchy", "estate"],
                        in_play=["copper", "silver"], discard=["province"],
                        coins=3, buys=2, turns_taken=3),
            PlayerState(deck=["province", "copper"], hand=["estate"],
                        discard=["silver", "duchy"], in_play=["gold"], turns_taken=3),
        ),
        supply=dict(zip(CARD_IDS, (40, 38, 27, 0, 0, 6))), turn_number=7,
    )
    expected = [
        3, 2,
        0, 0, 0, 1, 1, 0,  # Hand.
        1, 1, 0, 0, 0, 0,  # In play.
        1, 1, 1, 2, 1, 1,  # Owned.
        1, 1, 1, 1, 1, 1,  # Opponent owned.
        40, 38, 27, 0, 0, 6,
        2, 1, 11, 10, 7, 3, 3, 6, 2,
    ]
    observation = encode_observation(state)
    np.testing.assert_array_equal(observation, np.array(expected, dtype=np.float32))
    assert observation.dtype == np.float32
    assert FEATURE_INDEX["agent_vp"] == 34


def test_starting_totals_supply_and_vp_are_correct():
    game = DominionGame()
    obs = dict(zip(FEATURE_NAMES, encode_observation(game.state)))
    for prefix in ("owned", "opponent_owned"):
        assert [obs[f"{prefix}_{card}"] for card in CARD_IDS] == [7, 0, 0, 3, 0, 0]
    assert [obs[f"supply_{card}"] for card in CARD_IDS] == [46, 40, 30, 8, 8, 8]
    assert (obs["agent_vp"], obs["opponent_vp"], obs["turn_number"]) == (3, 3, 1)
    assert obs["deck_size"] == 5 and obs["discard_size"] == 0


def test_hidden_opponent_zone_distribution_and_all_deck_orders_are_invisible():
    state = DominionGame().state
    state.players[0].discard = ["silver", "estate"]
    original = encode_observation(state)
    changed = deepcopy(state)
    changed.players[0].deck.reverse()
    changed.players[0].discard.reverse()
    opponent = changed.players[1]
    all_owned = opponent.deck + opponent.hand + opponent.discard + opponent.in_play
    # Even radically different hidden zone sizes/compositions must remain invisible.
    opponent.deck = []
    opponent.hand = all_owned[:2]
    opponent.discard = all_owned[2:]
    opponent.in_play = []
    opponent.coins, opponent.buys = 99, 7
    np.testing.assert_array_equal(original, encode_observation(changed))


def test_observations_are_independent_values_and_do_not_mutate_state():
    state = DominionGame().state
    before = deepcopy(state)
    first, second = encode_observation(state), encode_observation(state)
    assert not np.shares_memory(first, second)
    first[:] = -999
    assert state == before
    np.testing.assert_array_equal(second, encode_observation(state))
