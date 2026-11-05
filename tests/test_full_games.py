from collections import Counter
from copy import deepcopy
from random import Random
import subprocess
import sys
from pathlib import Path

import pytest

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.random_agent import random_purchase
from dominion_rl.config import GameConfig
from dominion_rl.game import DominionGame


def assert_game_invariants(game):
    state = game.state
    cards = Counter(state.supply)
    for index, player in enumerate(state.players):
        owned = player.deck + player.hand + player.discard + player.in_play
        cards.update(owned)
        # Independent printed VP values, not registry values or engine helpers.
        assert game.score(index) == sum(
            {"estate": 1, "duchy": 3, "province": 6}.get(card_id, 0)
            for card_id in owned
        )
        assert player.buys >= 0 and player.coins >= 0
    cards.update(state.trash)
    assert cards == {"copper": 60, "silver": 40, "gold": 30,
                     "estate": 14, "duchy": 8, "province": 8}
    assert all(count >= 0 for count in state.supply.values())
    completed = sum(p.turns_taken for p in state.players)
    if state.terminated or state.truncated:
        assert state.turn_number == completed
    else:
        assert state.turn_number == completed + 1
        active = state.players[state.current_player]
        assert set(active.hand) <= {"estate", "duchy", "province"}
        assert set(active.in_play) <= {"copper", "silver", "gold"}


def play_seeded_game(seed, policies, max_turns=200, trace=False):
    game = DominionGame(GameConfig(seed=seed, max_turns=max_turns))
    rngs = (Random(seed * 2), Random(seed * 2 + 1))
    history = [deepcopy(game.state)] if trace else []
    assert_game_invariants(game)
    decisions = 0
    while not (game.state.terminated or game.state.truncated):
        index = game.state.current_player
        legal = game.legal_purchases()
        decision = (
            random_purchase(legal, rngs[index]) if policies[index] == "random"
            else big_money_purchase(legal, game.state.supply["province"])
        )
        assert decision is None or decision in legal
        if decision is None:
            game.end_buying()
        else:
            game.buy_card(decision)
        assert_game_invariants(game)
        if trace:
            history.append((decision, deepcopy(game.state)))
        decisions += 1
        assert decisions <= 2 * max_turns  # V0 has at most one Buy per turn.
    assert game.state.terminated != game.state.truncated
    if game.state.terminated:
        assert game.state.end_reason in ("province_empty", "three_piles_empty")
        rankings = [(game.score(i), -p.turns_taken) for i, p in enumerate(game.state.players)]
        expected = None if rankings[0] == rankings[1] else int(rankings[1] > rankings[0])
        assert game.state.winner == expected
    else:
        assert game.state.winner is None
        assert sum(p.turns_taken for p in game.state.players) == max_turns
    return game, history


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("policies", [
    ("big-money", "big-money"), ("big-money", "random"),
    ("random", "big-money"), ("random", "random"),
])
def test_complete_seeded_games_preserve_cards_and_use_only_legal_actions(seed, policies):
    play_seeded_game(seed, policies)


def test_complete_trajectory_is_reproducible():
    first, first_trace = play_seeded_game(42, ("random", "big-money"), trace=True)
    second, second_trace = play_seeded_game(42, ("random", "big-money"), trace=True)
    assert first_trace == second_trace
    assert first.state == second.state


def test_demonstration_runs_with_human_readable_trace():
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_engine.py"
    result = subprocess.run(
        [sys.executable, str(script), "--seed", "42", "--trace"],
        capture_output=True, text=True, check=True,
    )
    for label in ("Turn 1, P0", "played=", "legal=", "Supply", "End buying", "Final VP:"):
        assert label in result.stdout
