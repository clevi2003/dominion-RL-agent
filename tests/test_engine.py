from collections import Counter
from copy import deepcopy
from dataclasses import replace
import random

import pytest

from dominion_rl.config import GameConfig
from dominion_rl.game import DominionGame, GameState, PlayerState


def owned_cards(player):
    return player.deck + player.hand + player.discard + player.in_play


def all_cards(game):
    cards = Counter(game.state.supply)
    for player in game.state.players:
        cards.update(owned_cards(player))
    cards.update(game.state.trash)
    return cards


def small_game(player=None, opponent=None, max_turns=200):
    game = DominionGame(GameConfig(max_turns=max_turns))
    game.state = GameState(
        players=(player or PlayerState(), opponent or PlayerState()),
        supply={"copper": 46, "silver": 40, "gold": 30,
                "estate": 8, "duchy": 8, "province": 8},
    )
    return game


def test_starting_decks_supply_and_first_purchasing_boundary():
    game = DominionGame(GameConfig(seed=42))
    assert game.state.supply == {
        "copper": 46, "silver": 40, "gold": 30,
        "estate": 8, "duchy": 8, "province": 8,
    }
    for player in game.state.players:
        assert Counter(owned_cards(player)) == {"copper": 7, "estate": 3}
        assert len(player.deck) == 5
        assert len(player.hand) + len(player.in_play) == 5
        assert player.discard == [] and player.turns_taken == 0
    first, second = game.state.players
    assert set(first.hand) <= {"estate"}
    assert set(first.in_play) == {"copper"}
    assert first.coins == len(first.in_play)
    assert (first.actions, first.buys) == (1, 1)
    assert len(second.hand) == 5 and second.in_play == [] and second.coins == 0
    assert (game.score(0), game.score(1)) == (3, 3)
    assert all_cards(game) == {
        "copper": 60, "silver": 40, "gold": 30,
        "estate": 14, "duchy": 8, "province": 8,
    }


def test_seeded_initialization_and_global_random_independence():
    before = random.getstate()
    first = DominionGame(GameConfig(seed=42))
    second = DominionGame(GameConfig(seed=42))
    assert first.state == second.state
    assert first.state != DominionGame(GameConfig(seed=43)).state
    for _ in range(8):
        first.end_buying()
    assert random.getstate() == before


def test_reset_replaces_episode_state_and_reseeds():
    game = DominionGame(GameConfig(seed=17, max_turns=1))
    initial = deepcopy(game.state)
    old_state = game.state
    old_player = old_state.players[0]
    game.buy_card("copper")
    game.end_buying()
    assert old_state.truncated
    result = game.reset()
    assert result is game.state and result is not old_state
    assert game.state == initial
    assert game.state.players[0] is not old_player
    old_player.hand.append("province")
    old_state.supply["province"] = 0
    assert game.state == initial
    game.reset(seed=42)
    assert game.state == DominionGame(GameConfig(seed=42)).state
    game.reset()
    assert game.state == initial  # Per-reset override does not rewrite config.


@pytest.mark.parametrize("seed", [-1, True, "42"])
def test_invalid_reset_seed_does_not_change_state(seed):
    game = DominionGame()
    before, rng = deepcopy(game.state), game._rng.getstate()
    with pytest.raises(ValueError, match="seed"):
        game.reset(seed=seed)
    assert game.state == before and game._rng.getstate() == rng


def test_draw_uses_top_and_keeps_discard_until_deck_exhausted():
    player = PlayerState(deck=["estate", "copper"], discard=["gold", "silver"])
    game = small_game(player)
    before = all_cards(game)
    assert game.draw(player, 1) == ["copper"]
    assert player.deck == ["estate"]
    assert player.discard == ["gold", "silver"]
    assert game.draw(player, 4)[0] == "estate"
    assert Counter(player.hand) == {"copper": 1, "estate": 1, "gold": 1, "silver": 1}
    assert player.deck == player.discard == []
    assert all_cards(game) == before


def test_reshuffle_is_seeded_and_draw_zero_does_not_consume_randomness():
    games = [small_game(PlayerState(discard=["copper", "silver", "gold", "estate"]))
             for _ in range(2)]
    rng = games[0]._rng.getstate()
    assert games[0].draw(games[0].state.players[0], 0) == []
    assert games[0]._rng.getstate() == rng
    draws = [game.draw(game.state.players[0], 3) for game in games]
    assert draws[0] == draws[1]
    for game in games:
        player = game.state.players[0]
        assert player.discard == [] and len(player.deck) == 1 and len(player.hand) == 3


def test_empty_player_draws_nothing():
    game = small_game()
    assert game.draw(game.state.players[0], 5) == []


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_draw_count_is_atomic(count):
    game = DominionGame()
    before = deepcopy(game.state)
    with pytest.raises(ValueError, match="count"):
        game.draw(game.state.players[0], count)
    assert game.state == before


def test_buy_spends_resources_and_only_moves_supply_to_discard():
    player = PlayerState(deck=["estate"], hand=["estate"], coins=5, buys=2)
    game = small_game(player)
    before = all_cards(game)
    assert game.legal_purchases() == ["copper", "silver", "estate", "duchy"]
    game.buy_card("silver")
    assert (player.coins, player.buys, game.state.supply["silver"]) == (2, 1, 39)
    assert player.discard == ["silver"]
    assert player.deck == player.hand == ["estate"]
    assert (game.state.current_player, game.state.turn_number, player.turns_taken) == (0, 1, 0)
    assert all_cards(game) == before


def test_free_copper_and_multiple_buys():
    player = PlayerState(coins=8, buys=3)
    game = small_game(player)
    game.buy_card("duchy")
    game.buy_card("silver")
    assert (player.coins, player.buys) == (0, 1)
    assert game.legal_purchases() == ["copper"]
    game.buy_card("copper")
    assert player.discard == ["duchy", "silver", "copper"]
    assert (player.coins, player.buys) == (0, 0)
    assert game.legal_purchases() == []


@pytest.mark.parametrize("case", ["unknown", "unaffordable", "empty", "absent", "no_buys"])
def test_illegal_purchase_is_atomic(case):
    game = small_game(PlayerState(coins=3))
    target = "silver"
    if case == "unknown":
        target = "village"
    elif case == "unaffordable":
        target = "gold"
    elif case == "empty":
        game.state.supply["silver"] = 0
    elif case == "absent":
        del game.state.supply["silver"]
    else:
        game.state.players[0].buys = 0
    before, rng = deepcopy(game.state), game._rng.getstate()
    with pytest.raises(ValueError, match="Illegal purchase"):
        game.buy_card(target)
    assert game.state == before and game._rng.getstate() == rng


def test_legal_purchases_follow_configured_order():
    config = GameConfig()
    game = DominionGame(replace(config, cards=tuple(reversed(config.cards))))
    game.state.players[0].coins = 8
    assert game.legal_purchases() == list(reversed(config.cards))


def test_cleanup_draw_and_alternating_turn_treasure_effects():
    player = PlayerState(
        deck=["copper", "silver", "gold", "estate", "duchy"],
        hand=["estate"], in_play=["gold"], discard=["silver"], coins=3, buys=0,
    )
    opponent = PlayerState(hand=["copper", "estate", "gold"], coins=99, buys=9, actions=9)
    game = small_game(player, opponent)
    before = all_cards(game)
    game.end_buying()
    assert player.hand == ["duchy", "estate", "gold", "silver", "copper"]
    assert player.discard == ["silver", "estate", "gold"]
    assert player.deck == player.in_play == []
    assert (player.actions, player.buys, player.coins, player.turns_taken) == (0, 0, 0, 1)
    assert (game.state.current_player, game.state.turn_number) == (1, 2)
    assert opponent.hand == ["estate"] and opponent.in_play == ["copper", "gold"]
    assert (opponent.actions, opponent.buys, opponent.coins) == (1, 1, 4)
    assert all_cards(game) == before
    game.end_buying()  # Ending without buying is always allowed.
    assert (game.state.current_player, game.state.turn_number) == (0, 3)
    assert (player.coins, player.buys, player.actions) == (6, 1, 1)
    assert player.hand == ["duchy", "estate"]
    assert [p.turns_taken for p in game.state.players] == [1, 1]
    assert all_cards(game) == before


def test_cleanup_can_reshuffle_a_new_purchase_into_next_hand():
    game = small_game(PlayerState(hand=["estate"], in_play=["copper"], coins=3))
    before = all_cards(game)
    game.buy_card("silver")
    game.end_buying()
    player = game.state.players[0]
    assert Counter(player.hand) == {"estate": 1, "copper": 1, "silver": 1}
    assert player.deck == player.discard == player.in_play == []
    assert all_cards(game) == before


def test_score_includes_all_owned_zones_and_excludes_trash_and_supply():
    game = small_game(PlayerState(
        deck=["estate", "copper"], hand=["duchy"],
        discard=["province", "gold"], in_play=["estate"],
    ))
    game.state.trash = ["province"]
    assert game.score(0) == 11
    assert game.score(1) == 0
    with pytest.raises(ValueError, match="player_index"):
        game.score(-1)


def test_last_province_ends_at_turn_boundary_not_purchase():
    game = small_game(PlayerState(coins=8))
    game.state.supply["province"] = 1
    before = all_cards(game)
    game.buy_card("province")
    assert game.state.supply["province"] == 0
    assert not game.state.terminated and game.state.end_reason is None
    assert game.state.current_player == 0
    game.state.players[0].buys = 1
    game.buy_card("copper")  # Additional purchases still legal on the ending turn.
    game.end_buying()
    assert game.state.terminated and not game.state.truncated
    assert game.state.end_reason == "province_empty" and game.state.winner == 0
    assert (game.state.turn_number, game.state.current_player) == (1, 0)
    assert game.state.players[0].turns_taken == 1
    assert all_cards(game) == before


def test_third_empty_pile_is_checked_after_buying():
    game = small_game(PlayerState(coins=5))
    game.state.supply.update(copper=0, estate=0, duchy=1)
    game.buy_card("duchy")
    assert not game.state.terminated
    game.end_buying()
    assert game.state.end_reason == "three_piles_empty"
    assert game.state.terminated and game.state.winner == 0


def test_two_empty_piles_do_not_end_game():
    game = small_game()
    game.state.supply.update(copper=0, estate=0)
    game.end_buying()
    assert not game.state.terminated and game.state.end_reason is None
    assert game.state.current_player == 1


@pytest.mark.parametrize(
    "scores,previous_turns,winner",
    [
        ((2, 1), (1, 0), 0),  # VP takes precedence over more turns.
        ((1, 2), (0, 2), 1),
        ((1, 1), (0, 0), 1),  # Ending player completes one extra turn.
        ((1, 1), (0, 2), 0),
        ((1, 1), (0, 1), None),
    ],
)
def test_final_vp_and_completed_turn_tiebreak(scores, previous_turns, winner):
    game = small_game(*[
        PlayerState(hand=["estate"] * score, turns_taken=turns)
        for score, turns in zip(scores, previous_turns)
    ])
    game.state.turn_number = sum(previous_turns) + 1
    game.state.supply["province"] = 0
    game.end_buying()
    assert game.state.terminated and game.state.winner == winner
    assert game.state.players[0].turns_taken == previous_turns[0] + 1


def test_opponent_turn_can_end_with_equal_turns_and_shared_tie():
    game = small_game()
    game.end_buying()
    game.state.supply["province"] = 0
    game.end_buying()
    assert game.state.terminated and game.state.winner is None
    assert game.state.current_player == 1 and game.state.turn_number == 2
    assert [p.turns_taken for p in game.state.players] == [1, 1]


def test_turn_cap_counts_completed_individual_turns_and_never_declares_winner():
    game = small_game(PlayerState(hand=["province"]), max_turns=2)
    game.end_buying()
    assert not game.state.truncated
    game.end_buying()
    assert game.state.truncated and not game.state.terminated
    assert game.state.end_reason == "turn_limit" and game.state.winner is None
    assert game.state.turn_number == 2 and game.state.current_player == 1
    assert [p.turns_taken for p in game.state.players] == [1, 1]
    assert game.score(0) > game.score(1)


@pytest.mark.parametrize("reason", ["province_empty", "three_piles_empty"])
def test_normal_ending_takes_precedence_over_cap(reason):
    game = small_game(max_turns=1)
    if reason == "province_empty":
        game.state.supply["province"] = 0
    else:
        game.state.supply.update(copper=0, estate=0, duchy=0)
    game.end_buying()
    assert game.state.terminated and not game.state.truncated
    assert game.state.end_reason == reason


@pytest.mark.parametrize("normal_ending", [True, False])
def test_no_transitions_or_next_player_preparation_after_ending(normal_ending):
    opponent = PlayerState(hand=["gold"])
    game = small_game(opponent=opponent, max_turns=1)
    if normal_ending:
        game.state.supply["province"] = 0
    game.end_buying()
    assert opponent.hand == ["gold"] and opponent.in_play == []
    before, rng = deepcopy(game.state), game._rng.getstate()
    assert game.legal_purchases() == []
    for transition in (
        lambda: game.buy_card("copper"), game.end_buying,
        lambda: game.draw(game.state.players[0], 1),
    ):
        with pytest.raises(RuntimeError, match="ended"):
            transition()
        assert game.state == before and game._rng.getstate() == rng
    game.reset()
    assert not game.state.terminated and not game.state.truncated
