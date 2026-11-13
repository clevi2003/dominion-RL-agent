"""Exact public-composition P8 and potential-based reward invariants."""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest
import yaml

from dominion_rl.config import RewardConfig, load_reward_config
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import END_BUYING
from dominion_rl.env.observations import CARD_IDS, FEATURE_INDEX, FEATURE_NAMES, encode_observation
from dominion_rl.env.rewards import calculate_reward, five_card_p8, ownership_p8, potential
from dominion_rl.game import DominionGame
from dominion_rl.agents.big_money import big_money_purchase


ECONOMY = RewardConfig(mode='potential_based', potential='vp_economy')


def observation(agent=(7,1,1,3,0,0), opponent=(7,0,0,3,0,0)):
    obs = np.zeros(len(FEATURE_NAMES),dtype=np.float32)
    for prefix,counts in (('owned_',agent), ('opponent_owned_',opponent)):
        for card,n in zip(CARD_IDS,counts): obs[FEATURE_INDEX[prefix+card]] = n
    for name,counts in (('agent_vp',agent),('opponent_vp',opponent)):
        obs[FEATURE_INDEX[name]] = counts[3]+3*counts[4]+6*counts[5]
    return obs


@pytest.mark.parametrize('counts,expected', [((7,0,0,3),0), ((7,1,0,3),0), ((7,0,1,3),0),
    ((5,0,0,0),0), ((4,0,1,0),0), ((0,0,0,10),0), ((0,5,0,0),1), ((0,0,5,0),1),
    ((3,1,1,0),1), ((4,1,1,0),Fraction(2,3)), ((4,2,1,3),Fraction(13,126))])
def test_exact_simple_p8_values(counts, expected):
    assert five_card_p8(*counts) == pytest.approx(float(expected))


@pytest.mark.parametrize('counts', [(4,2,1,3),(2,2,2,2),(0,3,1,3)])
def test_mixed_compositions_against_independent_exhaustive_physical_hands(counts):
    # Enumerate individual card positions, independently of count-tuple weights.
    deck = [coins for coins,n in zip((1,2,3,0),counts) for _ in range(n)]
    hands = list(combinations(range(len(deck)),5))
    exact = Fraction(sum(sum(deck[i] for i in hand)>=8 for hand in hands),len(hands))
    assert five_card_p8(*counts) == pytest.approx(float(exact))


@pytest.mark.parametrize('counts', [(-1,3,3,3), (True,3,3,3), (1.,3,3,3), (1,0,0,3)])
def test_invalid_or_undersized_compositions_are_rejected(counts):
    with pytest.raises(ValueError): five_card_p8(*counts)


def test_victory_dilution_and_probability_bounds():
    probabilities = [five_card_p8(7,1,1,v) for v in range(3,9)]
    assert all(0 <= p <= 1 for p in probabilities)
    assert all(a>b for a,b in zip(probabilities,probabilities[1:]))
    assert five_card_p8(7,1,1,3)>five_card_p8(7,0,1,3)
    assert five_card_p8(7,0,2,3)>five_card_p8(7,0,1,3)


def test_agent_opponent_ownership_and_combined_potential():
    obs = observation(agent=(7,1,1,2,1,1), opponent=(7,0,2,1,0,2))
    before = obs.copy()
    a,b = five_card_p8(7,1,1,4),five_card_p8(7,0,2,3)
    assert ownership_p8(obs)==a and ownership_p8(obs,opponent=True)==b
    assert potential(obs,ECONOMY)==pytest.approx((11-13)/48+.25*(a-b))
    np.testing.assert_array_equal(obs,before)
    assert abs(potential(obs,ECONOMY)-potential(obs,replace(ECONOMY,potential='vp'))) <= .25


def test_player_position_symmetry_and_no_hidden_zone_or_order_leakage():
    game = DominionGame()
    state = game.state
    for player,cards in ((0,['gold','silver']), (1,['gold','gold','estate'])):
        for card in cards:
            state.players[player].discard.append(card)
            state.supply[card] -= 1
    a,b = encode_observation(state,0),encode_observation(state,1)
    assert a.shape==b.shape==(41,)
    assert ownership_p8(a)==ownership_p8(b,opponent=True)
    assert potential(a,ECONOMY)==pytest.approx(-potential(b,ECONOMY))
    changed = deepcopy(state)
    changed.players[0].deck.reverse()
    opponent = changed.players[1]
    cards = opponent.deck+opponent.hand+opponent.discard+opponent.in_play
    opponent.deck,opponent.hand,opponent.discard,opponent.in_play = cards[:1],cards[1:4],cards[4:],[]
    opponent.coins,opponent.buys = 999,99
    hidden_changed = encode_observation(changed,0)
    np.testing.assert_array_equal(a,hidden_changed)
    assert potential(a,ECONOMY)==potential(hidden_changed,ECONOMY)


def test_only_vp_and_total_ownership_fields_affect_potential():
    obs = observation()
    modified = np.full(41,999,dtype=np.float32)
    for name,index in FEATURE_INDEX.items():
        if name.startswith(('owned_','opponent_owned_')) or name in ('agent_vp','opponent_vp'):
            modified[index] = obs[index]
    assert potential(obs,ECONOMY)==potential(modified,ECONOMY)
    modified[FEATURE_INDEX['owned_silver']] = .5
    with pytest.raises(ValueError,match='raw'): potential(modified,ECONOMY)


def test_terminal_zero_and_truncation_retains_economy_without_outcome_reward():
    before,after = observation(),observation(agent=(7,1,2,3,0,0))
    assert potential(after,ECONOMY,terminated=True)==0
    terminal = calculate_reward(before,after,ECONOMY,terminated=True,truncated=False,winner=0)
    assert terminal==pytest.approx(1-potential(before,ECONOMY))
    truncation = calculate_reward(before,after,ECONOMY,terminated=False,truncated=True,winner=None)
    assert truncation==pytest.approx(.99*potential(after,ECONOMY)-potential(before,ECONOMY))
    # The terminal branch must not inspect invalid/empty ownership counts.
    assert potential(np.full(41,np.nan),ECONOMY,terminated=True)==0


def test_complete_legal_trajectory_discounted_shaping_telescopes_including_nonzero_start_suffix():
    env = DominionEnv(reward_config=ECONOMY,opponent='random')
    observations,rewards,base = [],[],[]
    try:
        obs,_ = env.reset(seed=47)
        while True:
            observations.append(obs.copy())
            legal = [card for card,allowed in zip(CARD_IDS,env.action_masks()) if allowed]
            card = big_money_purchase(legal,int(obs[FEATURE_INDEX['provinces_remaining']]))
            action = END_BUYING if card is None else CARD_IDS.index(card)
            obs,reward,terminated,truncated,info = env.step(action)
            rewards.append(reward)
            base.append(calculate_reward(observations[-1],obs,replace(ECONOMY,mode='terminal_only'),
                terminated=terminated,truncated=truncated,winner=info['winner']))
            if terminated or truncated: break
        assert terminated and not truncated
        nonzero = next(i for i,o in enumerate(observations) if abs(potential(o,ECONOMY))>1e-12)
        for start in (0,nonzero):
            delta = sum(.99**i*(r-b) for i,(r,b) in enumerate(zip(rewards[start:],base[start:])))
            assert delta==pytest.approx(-potential(observations[start],ECONOMY),abs=1e-10)
    finally:
        env.close()


def test_legacy_vp_behavior_and_reward_yaml_defaults_roundtrip(tmp_path):
    legacy = dict(mode='potential_based',gamma=.99,shaping_strength=1.,vp_normalization=48.,
                  win_reward=1.,tie_reward=0.,loss_reward=-1.)
    path = tmp_path/'legacy.yaml'
    path.write_text(yaml.safe_dump(legacy))
    config = load_reward_config(path)
    assert config.potential=='vp' and config.economy_coefficient==.25
    before,after = observation(),observation(agent=(7,1,2,3,1,0))
    vp = lambda obs:float(obs[FEATURE_INDEX['agent_vp']]-obs[FEATURE_INDEX['opponent_vp']])/48
    assert potential(before,config)==vp(before)
    assert calculate_reward(before,after,config,terminated=False,truncated=False,winner=None)==.99*vp(after)-vp(before)
    assert calculate_reward(before,after,replace(ECONOMY,mode='terminal_only'),terminated=False,truncated=False,winner=None)==0
    path.write_text(yaml.safe_dump(asdict(ECONOMY)))
    assert load_reward_config(path)==ECONOMY


@pytest.mark.parametrize('kwargs', [dict(potential='other'),dict(economy_coefficient=-.1),dict(economy_coefficient=float('nan'))])
def test_reward_config_validation(kwargs):
    with pytest.raises(ValueError): RewardConfig(**kwargs)
