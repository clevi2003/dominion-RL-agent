"""Focused V1 printed effects, automatic chains, purchases and conservation."""
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError
from random import Random

import pytest

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.engine_money import engine_money_purchase
from dominion_rl.agents.random_agent import random_purchase
from dominion_rl.config import GameConfig
from dominion_rl.game import DominionGame
from dominion_rl.game.cards import CARD_REGISTRY, CouncilRoom, Village
from dominion_rl.game.engine import INITIAL_SUPPLY, V1_INITIAL_SUPPLY


def owned(player):
    return Counter(player.deck+player.hand+player.discard+player.in_play)


def total_cards(game):
    cards = Counter(game.state.supply)
    for player in game.state.players: cards.update(owned(player))
    cards.update(game.state.trash)
    return cards


def arrange(game, hand, draws):
    """Arrange owned cards; acquire missing fixture cards from Supply, conserving all copies."""
    player = game.state.players[game.state.current_player]
    stock = owned(player)
    for card in hand+draws:
        if stock[card]: stock[card] -= 1
        else: game.state.supply[card] -= 1
    player.hand,player.deck = list(hand),list(reversed(draws))
    player.discard,player.in_play = list(stock.elements()),[]
    player.actions,player.buys,player.coins = 1,1,0
    game.phase = "action"  # Card-effect fixture is explicitly before Treasures.
    return player


@pytest.mark.parametrize('card,cost,cls',[('village',3,Village),('council_room',5,CouncilRoom)])
def test_v1_printed_definitions_are_immutable(card,cost,cls):
    assert type(CARD_REGISTRY[card]) is cls and CARD_REGISTRY[card].cost==cost
    with pytest.raises(FrozenInstanceError): CARD_REGISTRY[card].cost=99


def test_council_room_draws_four_grants_buy_and_immediately_draws_for_other_player():
    game = DominionGame(GameConfig(version='V1'))
    player = arrange(game,['council_room'],['gold','silver','copper','estate'])
    opponent = game.state.players[1]
    before,hand_size,other_size = total_cards(game),len(player.hand),len(opponent.hand)
    expected_other_card = opponent.deck[-1]
    game.play_action('council_room')
    assert len(player.hand)==hand_size-1+4 and player.actions==0 and player.buys==2
    assert player.in_play==['council_room']
    assert len(opponent.hand)==other_size+1 and opponent.hand[-1]==expected_other_card
    assert total_cards(game)==before


def test_village_spends_one_action_then_draws_one_and_grants_two():
    game = DominionGame(GameConfig(version='V1'))
    player = arrange(game,['village'],['gold'])
    game.play_action('village')
    assert player.hand==['gold'] and player.in_play==['village'] and player.actions==2
    player.actions=0
    player.hand.append('village')  # Deliberately invalid attempted play, no transition.
    before = deepcopy(game.state)
    with pytest.raises(ValueError,match='Illegal Action'): game.play_action('village')
    assert game.state==before


def test_automatic_village_priority_drawn_council_chaining_and_same_turn_treasures():
    game = DominionGame(GameConfig(version='V1'))
    player = arrange(game,['council_room','village'],
                     ['council_room','gold','silver','copper','estate','estate','estate','estate','estate'])
    before,other_hand = total_cards(game),len(game.state.players[1].hand)
    game._start_turn()
    assert player.in_play==['village','council_room','council_room','gold','silver','copper']
    assert player.actions==0 and player.buys==3 and player.coins==6
    assert player.hand==['estate']*5
    assert len(game.state.players[1].hand)==other_hand+2
    assert total_cards(game)==before


def test_newly_drawn_village_is_played_and_second_terminal_waits_without_action():
    game = DominionGame(GameConfig(version='V1'))
    player = arrange(game,['village'],['village','gold'])
    game._start_turn()
    assert player.in_play==['village','village','gold'] and player.coins==3 and player.actions==3
    player = arrange(game,['council_room','council_room'],['estate']*4)
    game._start_turn()
    assert player.in_play==['council_room'] and 'council_room' in player.hand and player.actions==0
    with pytest.raises(ValueError): game.play_action('council_room')


def test_all_twenty_actions_chain_finitely_without_recycling_in_play():
    game = DominionGame(GameConfig(version='V1'))
    player = arrange(game,['village']*10+['council_room']*10,[])
    before = total_cards(game)
    game._start_turn()
    assert player.in_play.count('village')==player.in_play.count('council_room')==10
    assert player.buys==11 and player.coins==7
    assert not set(player.hand)&{'village','council_room'}
    assert total_cards(game)==before


def test_multiple_buys_purchase_actions_to_discard_cleanup_and_unused_buy_end():
    game = DominionGame(GameConfig(version='V1'))
    player = arrange(game,['council_room'],['gold','silver','copper','estate'])
    before = total_cards(game)
    game._start_turn()
    assert player.buys==2 and player.coins==6
    game.buy_card('silver')
    assert player.buys==1 and player.coins==3
    game.buy_card('village')
    assert player.buys==0 and player.coins==0 and player.discard[-2:]==['silver','village']
    assert 'village' not in player.in_play
    player.actions=1  # Even an unused Action cannot play a bought card from discard.
    with pytest.raises(ValueError): game.play_action('village')
    with pytest.raises(ValueError): game.buy_card('copper')
    game.end_buying()
    assert player.turns_taken==1 and len(player.hand)==5 and not player.in_play
    assert player.actions==player.buys==player.coins==0
    assert total_cards(game)==before
    # A separate turn can end immediately, leaving all Council Room Buys unused.
    game = DominionGame(GameConfig(version='V1'))
    player = arrange(game,['council_room'],['estate']*4)
    game._start_turn()
    assert player.buys==2
    game.end_buying()
    assert player.buys==0 and player.turns_taken==1


def test_v1_supply_and_normal_end_precedes_truncation_with_turn_tiebreak():
    game = DominionGame(GameConfig(version='V1',max_turns=1))
    assert game.state.supply==V1_INITIAL_SUPPLY
    assert {k:game.state.supply[k] for k in INITIAL_SUPPLY}==INITIAL_SUPPLY
    game.state.supply['village']=game.state.supply['council_room']=game.state.supply['copper']=0
    game.end_buying()
    assert game.state.terminated and not game.state.truncated and game.state.end_reason=='three_piles_empty'
    assert game.state.winner==1  # Equal VP, P1 has fewer completed turns.
    game = DominionGame(GameConfig(version='V1'))
    game.state.players[1].turns_taken=1
    game.state.turn_number=2
    game.state.supply['province']=0
    game.end_buying()
    assert game.state.terminated and game.state.winner is None


@pytest.mark.parametrize('seed',[42,43,44])
def test_full_v1_games_are_seeded_conservative_and_use_shared_action_logic(seed):
    traces=[]
    for _ in range(2):
        game = DominionGame(GameConfig(version='V1',seed=seed))
        baseline=total_cards(game)
        rng=Random(seed)
        history=[]
        while not (game.state.terminated or game.state.truncated):
            p=game.state.players[game.state.current_player]
            legal=game.legal_purchases()
            decision=(engine_money_purchase(legal,game.state.supply['province'],owned(p)) if game.state.current_player==0
                      else random_purchase(legal,rng))
            if decision is None: game.end_buying()
            else: game.buy_card(decision)
            assert total_cards(game)==baseline
            assert all(n>=0 for n in game.state.supply.values())
            history.append((decision,deepcopy(game.state)))
            assert len(history)<=12*game.config.max_turns
        assert owned(game.state.players[0])['council_room']<=2
        assert owned(game.state.players[0])['village']<=1
        traces.append(history)
    assert traces[0]==traces[1]


def test_engine_money_caps_early_actions_and_preserves_big_money_reference():
    legal=list(CARD_REGISTRY)
    assert engine_money_purchase(legal,8,{})=='province'
    legal.remove('province')
    assert engine_money_purchase(legal,8,{})=='council_room'
    assert engine_money_purchase(legal,8,{'council_room':2})=='village'
    assert engine_money_purchase(legal,8,{'council_room':2,'village':1})=='gold'
    assert engine_money_purchase(legal,4,{})=='duchy'
    assert engine_money_purchase(['village','silver'],8,{})=='silver'
    assert big_money_purchase(['village','council_room'],8) is None
