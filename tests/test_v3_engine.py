"""V3 printed effects, private defensive choices, Action chains and conservation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from dominion_rl.agents.discard import MILITIA_DISCARD_PRIORITY, militia_discards
from dominion_rl.agents.militia_money import militia_money_purchase
from dominion_rl.config import GameConfig, load_game_config
from dominion_rl.game import DominionGame
from dominion_rl.game.cards import CARD_REGISTRY, Militia, Moat, V2_CARD_IDS, V3_CARD_IDS
from dominion_rl.game.engine import V2_INITIAL_SUPPLY, V3_INITIAL_SUPPLY
from test_v1_engine import arrange, owned, total_cards


def scene(hand, other_hand, draws=(), other_draws=(), seat=0, max_turns=200):
    game = DominionGame(GameConfig(version='V3', seed=71, max_turns=max_turns))
    game.state.current_player = seat
    p = arrange(game, list(hand), list(draws))
    game.state.current_player = 1-seat
    other = arrange(game, list(other_hand), list(other_draws))
    game.state.current_player = seat
    return game, p, other


@pytest.mark.parametrize('card,cls,cost,types', [
    ('militia', Militia, 4, {'action', 'attack'}), ('moat', Moat, 2, {'action', 'reaction'})])
def test_printed_definitions_config_and_supply(card, cls, cost, types):
    assert type(CARD_REGISTRY[card]) is cls and CARD_REGISTRY[card].cost == cost
    assert CARD_REGISTRY[card].card_types == frozenset(types)
    with pytest.raises(FrozenInstanceError):
        CARD_REGISTRY[card].cost = 99
    config = load_game_config(Path(__file__).resolve().parents[1]/'configs/game_v3.yaml')
    assert config.cards == V3_CARD_IDS and V3_CARD_IDS[:9] == V2_CARD_IDS
    assert DominionGame(config).state.supply == V3_INITIAL_SUPPLY == {**V2_INITIAL_SUPPLY, 'militia': 10, 'moat': 10}


@pytest.mark.parametrize('seat', [0, 1])
def test_militia_spends_moves_adds_coins_discards_only_other_and_preserves_ownership(seat):
    game, p, other = scene(['militia'], ['estate', 'duchy', 'gold', 'silver', 'copper'], seat=seat)
    other.actions, other.buys, other.coins = 7, 3, 9
    total, before_owned, own_deck = total_cards(game), owned(other), list(p.deck)
    trash = list(game.state.trash)
    game.play_action('militia')
    assert (p.actions, p.buys, p.coins) == (0, 1, 2) and p.in_play == ['militia']
    assert p.hand == [] and p.deck == own_deck
    assert other.hand == ['gold', 'silver', 'copper'] and other.discard[-2:] == ['estate', 'duchy']
    assert (other.actions, other.buys, other.coins) == (7, 3, 9)
    assert owned(other) == before_owned and total_cards(game) == total and game.state.trash == trash
    assert game.state.current_player == seat
    assert game.militia_resolutions == [dict(attacker=seat, defender=1-seat, moat_revealed=False, discard_count=2)]
    before = deepcopy(game.state)
    with pytest.raises(ValueError, match='active V3'):
        game.resolve_militia(other)
    assert game.state == before


@pytest.mark.parametrize('hand', [[], ['gold'], ['gold', 'silver', 'copper']])
def test_small_defender_hands_are_unchanged(hand):
    game, p, other = scene(['militia'], hand)
    before = deepcopy(other)
    game.play_action('militia')
    assert other == before and p.coins == 2 and game.militia_resolutions[0]['discard_count'] == 0


def test_discard_priority_is_fixed_stable_and_reads_only_the_supplied_hand():
    hand = ['gold', 'estate', 'militia', 'copper', 'silver', 'copper', 'village']
    before = list(hand)
    assert militia_discards(hand) == ['estate', 'copper', 'copper', 'militia']
    assert hand == before
    assert MILITIA_DISCARD_PRIORITY.index('copper') < MILITIA_DISCARD_PRIORITY.index('silver') < MILITIA_DISCARD_PRIORITY.index('gold')
    assert all(MILITIA_DISCARD_PRIORITY.index(c) < MILITIA_DISCARD_PRIORITY.index('copper') for c in ('estate', 'duchy', 'province'))


@pytest.mark.parametrize('card', ['militia', 'moat'])
def test_actions_require_a_hand_copy_and_available_action(card):
    game, p, _ = scene([card], ['gold'])
    p.actions = 0; before = deepcopy(game.state)
    with pytest.raises(ValueError, match='Illegal Action'):
        game.play_action(card)
    assert game.state == before and game.militia_resolutions == []
    p.actions = 1; p.hand.remove(card); p.discard.append(card)
    with pytest.raises(ValueError):
        game.play_action(card)


@pytest.mark.parametrize('seat', [0, 1])
def test_revealing_moat_is_pure_and_it_can_later_be_played_for_two_cards(seat):
    game, p, other = scene(['militia'], ['moat', 'estate', 'gold', 'silver', 'copper'],
                           other_draws=['silver', 'gold'], seat=seat)
    before, total = deepcopy(other), total_cards(game)
    game.play_action('militia')
    assert other == before and other.in_play == []
    assert game.militia_resolutions[-1]['moat_revealed'] and game.militia_resolutions[-1]['discard_count'] == 0
    game.state.current_player = 1-seat
    game.play_action('moat')
    assert other.actions == 0 and other.in_play == ['moat'] and len(other.hand) == 6
    assert other.hand[-2:] == ['silver', 'gold']
    assert total_cards(game) == total


@pytest.mark.parametrize('zone', ['deck', 'discard', 'in_play'])
def test_moat_outside_hand_does_not_protect(zone):
    game, _, other = scene(['militia'], ['estate', 'estate', 'copper', 'silver', 'gold'])
    game.state.supply['moat'] -= 1; getattr(other, zone).append('moat')
    game.play_action('militia')
    assert len(other.hand) == 3 and not game.militia_resolutions[-1]['moat_revealed']
    assert 'moat' in getattr(other, zone)


@pytest.mark.parametrize('protected', [False, True])
def test_multiple_militias_resolve_separately_and_each_generates_two_coins(protected):
    hand = ['moat' if protected else 'estate', 'estate', 'copper', 'silver', 'gold']
    game, p, other = scene(['festival', 'militia', 'militia'], hand)
    before, owned_before = total_cards(game), owned(other)
    game._start_turn()
    assert p.in_play == ['festival', 'militia', 'militia']
    assert (p.actions, p.buys, p.coins) == (0, 2, 6)
    assert [e['discard_count'] for e in game.militia_resolutions] == ([0, 0] if protected else [2, 0])
    assert [e['moat_revealed'] for e in game.militia_resolutions] == [protected, protected]
    assert len(other.hand) == (5 if protected else 3)
    assert total_cards(game) == before and owned(other) == owned_before


def test_council_room_can_draw_defenders_moat_before_automatic_militia():
    game, p, other = scene(['festival', 'council_room', 'militia'],
        ['estate', 'estate', 'copper', 'silver', 'gold'],
        draws=['gold', 'silver', 'copper', 'estate'], other_draws=['moat'])
    total = total_cards(game)
    game._start_turn()
    assert p.in_play == ['festival', 'council_room', 'militia', 'gold', 'silver', 'copper']
    assert (p.actions, p.buys, p.coins) == (0, 3, 10)
    assert len(other.hand) == 6 and other.hand[-1] == 'moat'
    assert game.militia_resolutions[-1]['moat_revealed'] and total_cards(game) == total


def test_moat_drawn_after_an_earlier_attack_does_not_retroactively_restore_discards():
    game, p, other = scene(['militia', 'council_room', 'militia'],
        ['estate', 'estate', 'copper', 'silver', 'gold'], other_draws=['moat'])
    p.actions = 3
    game.play_action('militia')
    discarded = list(other.discard)
    assert len(other.hand) == 3
    game.play_action('council_room')
    assert len(other.hand) == 4 and other.hand[-1] == 'moat'
    game.play_action('militia')
    assert other.discard == discarded and len(other.hand) == 4
    assert [e['moat_revealed'] for e in game.militia_resolutions] == [False, True]
    assert [e['discard_count'] for e in game.militia_resolutions] == [2, 0]
    assert p.coins == 4


@pytest.mark.parametrize('seat', [0, 1])
def test_priority_drawn_actions_treasures_attacks_and_multiple_buys(seat):
    game, p, other = scene(['council_room', 'festival', 'village'],
        ['estate', 'estate', 'copper', 'silver', 'gold'],
        draws=['militia', 'moat', 'gold', 'silver', 'copper', 'estate', 'estate'],
        other_draws=['estate'], seat=seat)
    total = total_cards(game)
    game._start_turn()
    assert p.in_play == ['village', 'festival', 'council_room', 'moat', 'militia', 'gold', 'silver', 'copper']
    assert (p.actions, p.buys, p.coins) == (0, 3, 10)
    assert len(other.hand) == 3 and game.militia_resolutions[-1]['discard_count'] == 3
    for card in ('militia', 'silver', 'moat'):
        game.buy_card(card)
    assert (p.coins, p.buys) == (1, 0) and p.discard[-3:] == ['militia', 'silver', 'moat']
    assert total_cards(game) == total


def test_cleanup_and_future_opponent_turn_use_the_reduced_hand():
    game, p, other = scene(['militia'], ['estate', 'duchy', 'gold', 'silver', 'copper'])
    before = total_cards(game)
    game.play_action('militia'); game.end_action_phase(); game.end_buying()
    assert game.state.current_player == 1 and p.in_play == [] and len(p.hand) == 5
    assert other.in_play == ['gold', 'silver', 'copper'] and other.coins == 6 and other.hand == []
    assert total_cards(game) == before


def test_fifty_actions_terminate_without_recycling_played_actions():
    hand = ['militia']*10 + ['moat']*10 + ['council_room']*10 + ['festival']*10 + ['village']*10
    game, p, _ = scene(hand, ['estate', 'estate', 'copper', 'silver', 'gold'])
    before = total_cards(game)
    game._start_turn()
    # 20 branching Actions give 21 available plays for 30 terminal Actions:
    # 10 Rooms then 10 Moats then one Militia under the documented priority.
    assert p.in_play[:30] == ['village']*10 + ['festival']*10 + ['council_room']*10
    assert p.in_play[30:41] == ['moat']*10 + ['militia']
    assert p.hand.count('militia') == 9 and p.actions == 0
    assert len(game.militia_resolutions) == 1 and total_cards(game) == before


@pytest.mark.parametrize('equal_turns', [False, True])
def test_normal_three_pile_ending_tiebreak_precedes_truncation(equal_turns):
    game, p, _ = scene(['militia'], ['gold'], max_turns=1)
    for card in ('militia', 'moat', 'village'):
        p.discard.extend([card]*game.state.supply[card]); game.state.supply[card] = 0
    game.state.players[0].turns_taken = 0
    game.state.players[1].turns_taken = 1 if equal_turns else 0
    before = total_cards(game)
    game.end_action_phase()
    game.end_buying()
    assert game.state.terminated and not game.state.truncated and game.state.end_reason == 'three_piles_empty'
    assert game.state.winner == (None if equal_turns else 1)
    assert total_cards(game) == before


def test_militia_money_is_fixed_capped_and_preserves_big_money_late_rules():
    legal = list(V3_CARD_IDS)
    assert militia_money_purchase(legal, 8, {}) == 'province'
    legal.remove('province')
    assert militia_money_purchase(legal, 8, {}) == 'militia'
    assert militia_money_purchase(legal, 8, {'militia': 2}) == 'gold'
    assert militia_money_purchase(legal, 4, {}) == 'duchy'
    assert militia_money_purchase(['militia', 'moat', 'copper'], 8, {'militia': 2}) is None




def test_discard_selector_receives_only_defenders_hand_and_reset_clears_public_summaries(monkeypatch):
    import dominion_rl.game.engine as engine
    game, _, other = scene(['militia'], ['estate', 'copper', 'copper', 'silver', 'gold'])
    before = tuple(other.hand); calls = []
    def choose(hand):
        calls.append(tuple(hand))
        return militia_discards(hand)
    monkeypatch.setattr(engine, 'militia_discards', choose)
    game.play_action('militia')
    assert calls == [before] and game.militia_resolutions[0]['discard_count'] == 2
    game.reset(seed=71)
    assert game.militia_resolutions == []
