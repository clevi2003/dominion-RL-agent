"""Versioned purchasing spaces, information boundaries and unchanged PBRS."""
from copy import deepcopy

import numpy as np
import pytest

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import ACTION_NAMES, END_BUYING
from dominion_rl.env.observations import (CARD_IDS, FEATURE_INDEX, FEATURE_NAMES, V1_FEATURE_INDEX,
                                         V1_FEATURE_NAMES, encode_observation)
from dominion_rl.env.preprocessing import PolicyObservation, preprocessing_config
from dominion_rl.env.rewards import ownership_p8, potential, terminal_reward
from dominion_rl.game.cards import V0_CARD_IDS, V1_CARD_IDS


def grant(game, player_index, card):
    game.state.supply[card]-=1
    game.state.players[player_index].discard.append(card)


def test_v0_constants_and_version_config_are_explicit():
    assert GameConfig().version=='V0' and GameConfig().cards==V0_CARD_IDS==CARD_IDS
    assert len(FEATURE_NAMES)==41 and len(ACTION_NAMES)==7 and END_BUYING==6
    assert V1_FEATURE_NAMES[:41]==FEATURE_NAMES and len(V1_FEATURE_NAMES)==51
    assert GameConfig(version='V1').cards==V1_CARD_IDS
    for version,cards in (('V0',V1_CARD_IDS),('V1',V0_CARD_IDS)):
        with pytest.raises(ValueError): GameConfig(version=version,cards=cards)
    with pytest.raises(ValueError): DominionEnv(opponent='engine-money')
    with pytest.raises(ValueError,match='raw'): preprocessing_config('fixed','V1')
    with pytest.raises(ValueError,match='raw'): PolicyObservation(DominionEnv(GameConfig(version='V1')),'fixed')


@pytest.mark.parametrize('seat',[0,1])
def test_v1_schema_masks_two_buys_and_end_buying_for_either_seat(seat):
    env=DominionEnv(GameConfig(version='V1',max_turns=8),opponent='big-money',agent_player=seat)
    try:
        obs,_=env.reset(seed=42)
        assert obs.shape==(51,) and obs.dtype==np.float32 and env.observation_space.contains(obs)
        assert env.action_space.n==9 and env.end_buying==8
        assert env.card_ids==V1_CARD_IDS and env.action_names[:6]==ACTION_NAMES[:6]
        assert env.action_names[6:]==('village','council_room','end_buying')
        p=env.game.state.players[seat]
        p.coins,p.buys=8,2
        obs,reward,terminated,truncated,_=env.step(6)
        assert obs[V1_FEATURE_INDEX['owned_village']]==1 and obs[V1_FEATURE_INDEX['supply_village']]==9
        assert p.discard[-1]=='village' and 'village' not in p.in_play
        assert p.coins==5 and p.buys==1 and env.action_masks()[7] and not env.action_masks()[5]
        env.step(7)
        np.testing.assert_array_equal(env.action_masks(),[False]*8+[True])
        assert p.discard[-1]=='council_room' and p.buys==0
        _,_,terminated,truncated,_=env.step(8)
        assert not terminated and not truncated and env.game.state.current_player==seat
        # An empty pile and insufficient coins independently affect purchase masks.
        env.game.state.supply['village']=0
        p.coins,p.buys=4,1
        assert not env.action_masks()[6] and not env.action_masks()[7] and env.action_masks()[8]
        with pytest.raises(ValueError): env.step(6)
    finally:
        env.close()


def test_opponent_private_gifts_zone_distribution_and_order_are_not_observed():
    env=DominionEnv(GameConfig(version='V1'))
    try:
        env.reset(seed=42)
        grant(env.game,1,'council_room')
        before=encode_observation(env.game.state)
        # This is Council Room's immediate opponent-draw operation.
        env.game.draw(env.game.state.players[1],1)
        np.testing.assert_array_equal(before,encode_observation(env.game.state))
        changed=deepcopy(env.game.state)
        other=changed.players[1]
        all_cards=other.deck+other.hand+other.discard+other.in_play
        other.deck,other.hand,other.discard,other.in_play=all_cards[:1],all_cards[1:3],all_cards[3:],[]
        other.coins,other.buys=999,99
        changed.players[0].deck.reverse()
        np.testing.assert_array_equal(before,encode_observation(changed))
        assert set(env._info(before))=={'action_mask','turn_number','current_player','agent_player','episode_opponent',
                                         'scores','turns_taken','winner','end_reason'}
    finally:
        env.close()


@pytest.mark.parametrize('seat',[0,1])
def test_p8_actions_dilute_both_players_and_preserve_vp_relative_symmetry(seat):
    reward=RewardConfig(mode='potential_based',potential='vp_economy')
    env=DominionEnv(GameConfig(version='V1'),reward,agent_player=seat)
    try:
        env.reset(seed=42)
        for player in (0,1):
            grant(env.game,player,'gold')
            grant(env.game,player,'silver')
        before=encode_observation(env.game.state,seat)
        original=ownership_p8(before)
        assert original>0 and original==ownership_p8(before[:41])
        p=env.game.state.players[seat]
        p.coins,p.buys=3,1
        after,actual,_,_,_=env.step(6)
        assert ownership_p8(after)<original and ownership_p8(after[:41])==original
        assert actual==pytest.approx(.99*potential(after,reward)-potential(before,reward))
        other=encode_observation(env.game.state,1-seat)
        assert ownership_p8(after)==ownership_p8(other,opponent=True)
        assert potential(after,reward)==pytest.approx(-potential(other,reward))
        # Council Room also counts as zero Treasure in the hypothetical sample.
        previous=ownership_p8(after)
        p.coins,p.buys=5,1
        final,*_=env.step(7)
        assert ownership_p8(final)<previous
    finally:
        env.close()


@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('true_terminal',[False,True])
def test_v1_terminal_and_truncated_rewards_have_correct_endpoints(seat,true_terminal):
    config=RewardConfig(mode='potential_based',potential='vp_economy')
    env=DominionEnv(GameConfig(version='V1',max_turns=2),config,opponent='big-money',agent_player=seat)
    try:
        env.reset(seed=42)
        grant(env.game,seat,'gold');grant(env.game,seat,'silver')
        before=encode_observation(env.game.state,seat)
        if true_terminal: env.game.state.supply['province']=0
        after,reward,terminated,truncated,info=env.step(env.end_buying)
        assert terminated==true_terminal and truncated!=true_terminal
        winner=None if info['winner'] is None else int(info['winner']!=seat)
        base=terminal_reward(winner,config,terminated=terminated,truncated=truncated)
        assert reward==pytest.approx(base+.99*potential(after,config,terminated=terminated)-potential(before,config))
        if true_terminal: assert potential(after,config,terminated=True)==0
        else: assert potential(after,config)!=0 and info['winner'] is None
    finally:
        env.close()


def test_v1_seed_reproducibility_including_engine_money_and_action_purchases():
    traces=[]
    for _ in range(2):
        config=RewardConfig(mode='potential_based',potential='vp_economy')
        env=DominionEnv(GameConfig(version='V1',max_turns=20),config,opponent='engine-money')
        try:
            obs,_=env.reset(seed=43)
            trace=[obs.copy()]
            initial=potential(obs,config)
            shaping_sum=0.
            decisions=0
            while True:
                mask=env.action_masks()
                action=next((i for i in (5,7,6,2,1) if mask[i]),env.end_buying)
                obs,reward,terminated,truncated,info=env.step(action)
                base=terminal_reward(info['winner'],config,terminated=terminated,truncated=truncated)
                shaping_sum+=config.gamma**decisions*(reward-base)
                decisions+=1
                trace.append((obs.copy(),reward,terminated,truncated,info['scores']))
                if terminated or truncated: break
            assert shaping_sum==pytest.approx(-initial+config.gamma**decisions*potential(obs,config,terminated=terminated),abs=1e-10)
            traces.append(trace)
        finally: env.close()
    assert len(traces[0])==len(traces[1])
    np.testing.assert_array_equal(traces[0][0],traces[1][0])
    for a,b in zip(traces[0][1:],traces[1][1:]):
        np.testing.assert_array_equal(a[0],b[0])
        assert a[1:]==b[1:]


def test_v1_all_five_appended_card_groups_have_exact_meanings():
    env=DominionEnv(GameConfig(version='V1'))
    try:
        env.reset(seed=42)
        state=env.game.state
        state.players[0].hand.append('village');state.supply['village']-=1
        state.players[0].in_play.append('council_room');state.supply['council_room']-=1
        grant(env.game,1,'village')
        obs=encode_observation(state)
        expected=dict(hand_village=1,hand_council_room=0,in_play_village=0,in_play_council_room=1,
            owned_village=1,owned_council_room=1,opponent_owned_village=1,opponent_owned_council_room=0,
            supply_village=8,supply_council_room=9)
        assert {name:int(obs[V1_FEATURE_INDEX[name]]) for name in expected}==expected
        assert obs[FEATURE_INDEX['agent_vp']]==3 and obs[FEATURE_INDEX['opponent_vp']]==3
        copied=deepcopy(state);copied.version='V0'
        np.testing.assert_array_equal(obs[:41],encode_observation(copied))
    finally: env.close()
