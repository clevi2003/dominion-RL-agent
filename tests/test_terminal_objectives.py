"""Terminal targets, legacy arithmetic and reward-only interface invariants."""
from dataclasses import replace
import struct

import numpy as np
import pytest

from dominion_rl.config import GameConfig, RewardConfig, load_reward_config
from dominion_rl.env import DominionEnv
from dominion_rl.env.observations import FEATURE_INDEX
from dominion_rl.env.rewards import calculate_reward, terminal_reward
from dominion_rl.agents.big_money import big_money_purchase


def obs(agent,opponent,version='V1'):
    values = np.zeros(51 if version == 'V1' else 41,dtype=np.float32)
    values[FEATURE_INDEX['agent_vp']],values[FEATURE_INDEX['opponent_vp']] = agent,opponent
    return values


def legacy_reward(previous,current,c,terminated,truncated,winner):
    """Frozen pre-objective arithmetic, including the original operation order."""
    reward = 0.0
    if terminated and not truncated:
        reward = float(c.tie_reward if winner is None else c.win_reward if winner == 0 else c.loss_reward)
    if c.mode == 'potential_based':
        before = (float(previous[34])-float(previous[35]))/c.vp_normalization
        after = 0.0 if terminated else (float(current[34])-float(current[35]))/c.vp_normalization
        reward += c.shaping_strength*(c.gamma*after-before)
    return float(reward)


@pytest.mark.parametrize('version',['V0','V1'])
@pytest.mark.parametrize('gamma',[.99,1.])
@pytest.mark.parametrize('mode',['terminal_only','potential_based'])
def test_win_loss_preserves_legacy_reward_bits(version,gamma,mode):
    default = RewardConfig(mode=mode,gamma=gamma,win_reward=2.125,tie_reward=.375,loss_reward=-3.25)
    for ended,truncated in ((False,False),(False,True),(True,False),(True,True)):
        for winner in (0,1,None):
            previous,current = obs(9,22,version),obs(15,27,version)
            expected = legacy_reward(previous,current,default,ended,truncated,winner)
            for config in (default,replace(default,terminal_objective='win_loss')):
                actual = calculate_reward(previous,current,config,terminated=ended,truncated=truncated,winner=winner)
                assert struct.pack('!d',actual) == struct.pack('!d',expected)


@pytest.mark.parametrize('version',['V0','V1'])
@pytest.mark.parametrize('agent,opponent,winner,expected',[
    (30,24,0,6/48),(24,24,None,0.),(24,24,0,0.),(23,24,1,-1/48),
    (3,51,1,-1.),(0,120,1,-2.5),(120,0,0,2.5)])
def test_margin_uses_final_vp_not_official_outcome_or_potential_scale(version,agent,opponent,winner,expected):
    config = RewardConfig(gamma=1.,terminal_objective='vp_margin',vp_normalization=12.,
                          win_reward=100,tie_reward=5,loss_reward=-100)
    assert calculate_reward(obs(3,3,version),obs(agent,opponent,version),config,
        terminated=True,truncated=False,winner=winner) == expected
    assert terminal_reward(None,config,terminated=True,truncated=False,observation=obs(opponent,agent,version)) == -expected


@pytest.mark.parametrize('ended,truncated',[(False,False),(False,True),(True,True)])
def test_margin_not_paid_before_true_completion(ended,truncated):
    config = RewardConfig(gamma=1.,terminal_objective='vp_margin')
    assert calculate_reward(obs(3,3),obs(120,0),config,terminated=ended,truncated=truncated,winner=0) == 0.
    assert terminal_reward(0,config,terminated=ended,truncated=truncated) == 0.


def test_margin_requires_actual_final_scores_and_valid_objective():
    with pytest.raises(ValueError,match='final observation'):
        terminal_reward(0,RewardConfig(terminal_objective='vp_margin'),terminated=True,truncated=False)
    with pytest.raises(ValueError,match='terminal_objective'):RewardConfig(terminal_objective='unknown')


def test_terminal_objective_and_shaping_are_independent():
    config = RewardConfig(mode='potential_based',gamma=1.,terminal_objective='vp_margin')
    actual = calculate_reward(obs(9,3),obs(15,3),config,terminated=True,truncated=False,winner=0)
    assert actual == (15-3)/48-(9-3)/48
    # Truncation has no margin payout, but retains the nonterminal potential.
    assert calculate_reward(obs(9,3),obs(15,3),config,terminated=False,truncated=True,winner=None) == 6/48


@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('version',['V0','V1'])
def test_env_pays_margin_after_cleanup_using_actual_final_state(seat,version):
    env = DominionEnv(GameConfig(version=version),RewardConfig(gamma=1.,terminal_objective='vp_margin'),agent_player=seat)
    try:
        env.reset(seed=32000)
        # Leave the last Province for this purchase; put the others into real ownership.
        game = env.game;state = game.state
        state.players[1-seat].discard += ['province']*(state.supply['province']-1)
        state.supply['province'] = 1
        state.players[seat].coins,state.players[seat].buys = 8,1
        observation,reward,terminated,truncated,_ = env.step(5)
        assert not terminated and not truncated and reward == 0.
        assert observation[FEATURE_INDEX['agent_vp']] == game.score(seat)
        observation,reward,terminated,truncated,info = env.step(env.end_buying)
        assert terminated and not truncated
        assert reward == (game.score(seat)-game.score(1-seat))/48
        assert info['scores'] == (game.score(seat),game.score(1-seat))
    finally:env.close()


@pytest.mark.parametrize('objective',['win_loss','vp_margin'])
def test_gamma_one_removes_zero_reward_loss_delay_advantage(objective):
    terminal = calculate_reward(obs(3,3),obs(3,27),RewardConfig(terminal_objective=objective),
        terminated=True,truncated=False,winner=1)
    def discounted(gamma,extra):
        return sum(gamma**i*r for i,r in enumerate([0.]*(2+extra)+[terminal]))
    assert discounted(.99,3) > discounted(.99,0)
    assert discounted(1.,3) == discounted(1.,0) == terminal


@pytest.mark.parametrize('version,opponent',[('V0','random'),('V0','big-money'),('V1','random'),('V1','big-money'),('V1','engine-money')])
@pytest.mark.parametrize('seat',[0,1])
def test_objective_changes_no_games_observations_masks_or_heuristic_decisions(version,opponent,seat):
    base = DominionEnv(GameConfig(version=version,max_turns=8),RewardConfig(gamma=1.),opponent=opponent,agent_player=seat)
    margin = DominionEnv(GameConfig(version=version,max_turns=8),RewardConfig(gamma=1.,terminal_objective='vp_margin'),opponent=opponent,agent_player=seat)
    try:
        a,_ = base.reset(seed=32001);b,_ = margin.reset(seed=32001)
        while True:
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(base.action_masks(),margin.action_masks())
            assert base.game.state == margin.game.state
            legal = [card for card,allowed in zip(base.card_ids,base.action_masks()) if allowed]
            card = big_money_purchase(legal,int(a[FEATURE_INDEX['provinces_remaining']]))
            action = base.end_buying if card is None else base.card_ids.index(card)
            a,_,ended,truncated,info = base.step(action)
            b,_,other_end,other_truncated,other_info = margin.step(action)
            assert (ended,truncated) == (other_end,other_truncated)
            assert base.game.state == margin.game.state
            if ended or truncated:
                np.testing.assert_array_equal(a,b)
                assert info['scores'] == other_info['scores'] and info['winner'] == other_info['winner']
                break
    finally:base.close();margin.close()


def test_legacy_yaml_missing_objective_defaults_to_win_loss(tmp_path):
    path = tmp_path/'old_rewards.yaml'
    path.write_text('mode: terminal_only\ngamma: 1.0\n')
    assert load_reward_config(path) == RewardConfig(gamma=1.)
