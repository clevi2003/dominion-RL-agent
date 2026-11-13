"""Festival printed effects, deterministic chains and conserved physical cards."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.engine_money import engine_money_purchase
from dominion_rl.agents.festival_money import festival_money_purchase
from dominion_rl.config import GameConfig, load_game_config
from dominion_rl.game import DominionGame
from dominion_rl.game.cards import CARD_REGISTRY, Festival, V1_CARD_IDS, V2_CARD_IDS
from dominion_rl.game.engine import V1_INITIAL_SUPPLY, V2_INITIAL_SUPPLY
from test_v1_engine import arrange, total_cards


def test_festival_definition_supply_and_config():
    assert type(CARD_REGISTRY['festival']) is Festival
    assert (Festival().name, Festival().cost) == ('Festival', 5)
    with pytest.raises(FrozenInstanceError):
        CARD_REGISTRY['festival'].cost = 6
    config = load_game_config(Path(__file__).resolve().parents[1] / 'configs/game_v2.yaml')
    game = DominionGame(config)
    assert game.state.supply == V2_INITIAL_SUPPLY == {**V1_INITIAL_SUPPLY, 'festival': 10}
    assert V2_CARD_IDS[:-1] == V1_CARD_IDS
    with pytest.raises(ValueError):
        GameConfig(version='V1', cards=V2_CARD_IDS)
    with pytest.raises(ValueError):
        GameConfig(version='V2', cards=V1_CARD_IDS)


def test_festival_spends_moves_grants_resources_without_drawing_and_cleans_up():
    game = DominionGame(GameConfig(version='V2'))
    p = arrange(game, ['festival', 'estate'], ['gold', 'silver'])
    before, deck = total_cards(game), list(p.deck)
    p.actions, p.buys, p.coins = 1, 2, 3
    game.play_action('festival')
    assert (p.actions, p.buys, p.coins) == (2, 3, 5)
    assert p.hand == ['estate'] and p.in_play == ['festival'] and p.deck == deck
    assert total_cards(game) == before
    game.end_action_phase()
    game.buy_card('festival')
    assert p.discard[-1] == 'festival' and p.in_play == ['festival']
    assert (p.coins, p.buys, game.state.supply['festival']) == (0, 2, 8)
    game.end_buying()
    assert p.in_play == [] and len(p.hand) == 5
    assert (p.actions, p.buys, p.coins) == (0, 0, 0)
    assert total_cards(game) == before


def test_festival_cannot_play_without_action_or_outside_version_or_from_discard():
    game = DominionGame(GameConfig(version='V2'))
    p = arrange(game, ['festival'], [])
    p.actions = 0
    before = deepcopy(game.state)
    with pytest.raises(ValueError):
        game.play_action('festival')
    assert game.state == before
    p.actions = 1
    p.hand.remove('festival'); p.discard.append('festival')
    with pytest.raises(ValueError):
        game.play_action('festival')
    old = DominionGame(GameConfig(version='V1'))
    old.state.players[0].hand.append('festival')  # Malformed state must not enable a V2 Action.
    with pytest.raises(ValueError):
        old.play_action('festival')


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('hand,draws,played,actions,buys,coins', [
    (['council_room', 'festival'], ['gold', 'silver', 'copper', 'estate'],
     ['festival', 'council_room'], 1, 3, 8),
    (['festival', 'village'], ['gold'], ['village', 'festival'], 3, 2, 5),
    (['festival', 'festival'], [], ['festival', 'festival'], 3, 3, 4),
    (['council_room', 'festival', 'council_room'], ['estate'] * 8,
     ['festival', 'council_room', 'council_room'], 0, 4, 2),
    # Room draws a Festival, which in turn permits the newly drawn Room.
    (['festival', 'council_room'], ['festival', 'council_room', 'estate', 'estate'] + ['estate'] * 4,
     ['festival', 'council_room', 'festival', 'council_room'], 1, 5, 4),
    (['village'], ['festival'], ['village', 'festival'], 3, 2, 2),
])
def test_v2_automatic_priority_newly_drawn_actions_and_finite_chains(seat, hand, draws, played, actions, buys, coins):
    game = DominionGame(GameConfig(version='V2'))
    game.state.current_player = seat
    p = arrange(game, hand, draws)
    before = total_cards(game)
    game._start_turn()
    assert [c for c in p.in_play if c in ('village', 'festival', 'council_room')] == played
    assert (p.actions, p.buys, p.coins) == (actions, buys, coins)
    assert total_cards(game) == before
    # Every Action played is physically in_play, never recycled by these effects.
    assert len(played) <= sum(before[c] for c in ('village', 'festival', 'council_room'))


def test_multiple_buys_use_combined_festival_and_room_resources():
    game = DominionGame(GameConfig(version='V2'))
    p = arrange(game, ['festival', 'council_room'], ['gold', 'silver', 'copper', 'estate'])
    game._start_turn()
    before = total_cards(game)
    for card in ('festival', 'silver', 'copper'):
        game.buy_card(card)
    assert p.buys == p.coins == 0
    assert p.discard[-3:] == ['festival', 'silver', 'copper']
    assert game.legal_purchases() == [] and total_cards(game) == before


@pytest.mark.parametrize('reason', ['province_empty', 'three_piles_empty', 'turn_limit'])
def test_v2_end_conditions_and_card_conservation(reason):
    game = DominionGame(GameConfig(version='V2', max_turns=1))
    p = arrange(game, ['festival'], [])
    game._start_turn()
    # Move depleted piles to public owned cards; do not destroy fixture copies.
    depleted = ['province'] if reason == 'province_empty' else (
        ['festival', 'village', 'council_room'] if reason == 'three_piles_empty' else [])
    for card in depleted:
        p.discard.extend([card] * game.state.supply[card]); game.state.supply[card] = 0
    before = total_cards(game)
    game.end_buying()
    assert game.state.end_reason == reason
    assert game.state.truncated == (reason == 'turn_limit')
    assert game.state.terminated == (reason != 'turn_limit')
    if game.state.truncated:
        assert game.state.winner is None
    assert total_cards(game) == before


def test_fixed_festival_heuristic_caps_and_old_heuristics():
    legal = list(V2_CARD_IDS)
    assert festival_money_purchase(legal, 8, {}) == 'province'
    legal.remove('province')
    assert festival_money_purchase(legal, 8, {}) == 'festival'
    assert festival_money_purchase(legal, 8, {'festival': 2}) == 'council_room'
    capped = {'festival': 2, 'council_room': 2}
    assert festival_money_purchase(legal, 8, capped) == 'gold'
    assert festival_money_purchase(legal, 4, {}) == 'duchy'
    assert festival_money_purchase(['festival', 'copper'], 8, capped) is None
    assert big_money_purchase(['village', 'festival', 'council_room'], 8) is None
    assert engine_money_purchase(legal, 8, {}) == 'council_room'




def test_all_thirty_actions_chain_without_recycling_in_play():
    game = DominionGame(GameConfig(version='V2'))
    p = arrange(game, ['council_room'] * 10 + ['festival'] * 10 + ['village'] * 10, [])
    before = total_cards(game)
    game._start_turn()
    assert [c for c in p.in_play if c in ('village', 'festival', 'council_room')] == (
        ['village'] * 10 + ['festival'] * 10 + ['council_room'] * 10)
    assert (p.actions, p.buys, p.coins) == (11, 21, 27)
    assert total_cards(game) == before
