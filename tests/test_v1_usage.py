"""Evaluation-only V1 Action-use metrics with controlled short game fixtures."""
from collections import Counter
import csv
import io
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('sb3_contrib')
from dominion_rl.config import GameConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.observations import encode_observation
from dominion_rl.ppo import V1_USAGE_FIELDS, play_episode


def actor(env, purchases):
    choices=iter(purchases)
    def predict(obs, deterministic=True, action_masks=None):
        card=next(choices,None)
        action=env.end_buying if card is None else env.card_ids.index(card)
        assert action_masks[action]
        return np.asarray(action),None
    return SimpleNamespace(gamma=.99,game_contract=env.game_contract,
        observation_space=env.observation_space,action_space=env.action_space,
        set_random_seed=lambda seed:None,predict=predict)


def short_game(monkeypatch, seat, hand, draws, version="V1"):
    env=DominionEnv(GameConfig(version=version,max_turns=2),opponent='big-money',agent_player=seat)
    reset=env.reset
    def arranged_reset(**kwargs):
        reset(**kwargs)
        game=env.game;player=game.state.players[seat]
        stock=Counter(player.deck+player.hand+player.discard+player.in_play)
        for card in hand+draws:
            if stock[card]:stock[card]-=1
            else:game.state.supply[card]-=1
        player.hand,player.deck=list(hand),list(reversed(draws))
        player.discard,player.in_play=list(stock.elements()),[]
        player.actions,player.buys,player.coins=1,1,0
        game._start_turn()
        obs=encode_observation(game.state,seat)
        return obs,env._info(obs)
    monkeypatch.setattr(env,'reset',arranged_reset)
    return env


def serialized(row):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(row))
    writer.writeheader();writer.writerow(row);stream.seek(0)
    return next(csv.DictReader(stream))


@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('purchases,non_copper,multiple',[
    (['silver','silver'],1,1),(['silver'],1,0),(['gold'],0,0)])
def test_extra_buy_grant_opportunity_and_actual_use_are_separate(monkeypatch,seat,purchases,non_copper,multiple):
    env=short_game(monkeypatch,seat,['council_room'],['gold','silver','copper','estate'])
    try:
        row=play_episode(actor(env,purchases),env,31000)
        assert row['observed_agent_turns']==row['agent_turns']==1
        assert row['council_room_plays']==row['turns_with_council_room_plays']==1
        assert row['council_room_extra_buys_granted']==1
        assert row['turns_with_legal_extra_purchase_opportunity']==1
        assert row['turns_with_non_copper_extra_purchase_opportunity']==non_copper
        assert row['turns_with_multiple_purchases']==multiple
        assert row['purchases_council_room']==0 and row['first_council_room_turn'] is None
        assert row['outcome']=='truncated' and row['tie_adjusted_score'] is None
    finally:env.close()


def test_action_purchase_timing_does_not_count_unplayed_purchase(monkeypatch):
    env=short_game(monkeypatch,0,['village'],['gold'])
    try:
        row=play_episode(actor(env,['village']),env,31001)
        assert row['village_plays']==1 and row['purchases_village']==1
        assert row['first_village_turn']==1 and row['first_council_room_turn'] is None
        assert row['turns_with_multiple_purchases']==row['council_room_extra_buys_granted']==0
    finally:env.close()




def test_v0_evaluation_has_no_v1_usage_or_action_timing_fields():
    env=DominionEnv(GameConfig(max_turns=2))
    try:
        row=play_episode(actor(env,[]),env,31003)
        assert not set(V1_USAGE_FIELDS).intersection(row)
        assert 'first_village_turn' not in row and 'first_council_room_turn' not in row
        assert row['game_version']=='V0' and row['outcome']=='truncated'
    finally:env.close()
