"""V3 append-only schemas, both seats, private defense and unchanged rewards."""
from copy import deepcopy

import numpy as np
import pytest

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.observations import (FEATURE_NAMES, V1_FEATURE_NAMES, V2_FEATURE_NAMES,
    V2_FEATURE_INDEX, V3_FEATURE_NAMES, V3_FEATURE_INDEX, encode_observation)
from dominion_rl.env.preprocessing import preprocessing_config
from dominion_rl.env.rewards import ownership_p8, potential, terminal_reward
from dominion_rl.game.cards import V2_CARD_IDS, V3_CARD_IDS
from test_v1_env import grant


def test_old_indices_spaces_and_v3_card_mappings():
    assert V3_FEATURE_NAMES[:56] == V2_FEATURE_NAMES
    assert V3_FEATURE_NAMES[:51] == V1_FEATURE_NAMES and V3_FEATURE_NAMES[:41] == FEATURE_NAMES
    assert all(V3_FEATURE_INDEX[name] == index for name, index in V2_FEATURE_INDEX.items())
    assert V3_CARD_IDS[:9] == V2_CARD_IDS and V3_CARD_IDS[9:] == ('militia', 'moat')
    assert V3_FEATURE_NAMES[56:] == tuple(f'{prefix}_{card}' for prefix in
        ('hand', 'in_play', 'owned', 'opponent_owned', 'supply') for card in ('militia', 'moat'))
    for version, observations, actions in (('V0', 41, 7), ('V1', 51, 9), ('V2', 56, 10), ('V3', 66, 12)):
        env = DominionEnv(GameConfig(version=version))
        try:
            obs, _ = env.reset(seed=42)
            assert obs.shape == (observations,) and env.action_space.n == actions
            assert env.observation_space.contains(obs)
        finally:
            env.close()
    with pytest.raises(ValueError):
        GameConfig(version='V2', cards=V3_CARD_IDS)
    with pytest.raises(ValueError, match='raw'):
        preprocessing_config('fixed', 'V3')
    assert preprocessing_config('raw', 'V3')['divisors'] == [1]*66


@pytest.mark.parametrize('seat', [0, 1])
def test_militia_moat_affordability_masks_discard_purchases_and_no_mid_episode_selection(seat):
    env = DominionEnv(GameConfig(version='V3', max_turns=8), opponent='militia-money', agent_player=seat)
    try:
        env.reset(seed=71); counts = dict(env.opponent_reset_counts)
        initial_militia = env.game.state.supply['militia']
        p = env.game.state.players[seat]
        assert env.end_buying == 11
        p.coins, p.buys = 3, 2
        assert env.action_masks()[10] and not env.action_masks()[9]
        p.coins = 6
        obs, _, _, _, _ = env.step(9)
        assert p.discard[-1] == 'militia' and p.coins == 2 and p.buys == 1 and p.in_play.count('militia') == 0
        assert obs[V3_FEATURE_INDEX['owned_militia']] == 1 and obs[V3_FEATURE_INDEX['supply_militia']] == initial_militia-1
        assert env.action_masks()[10] and not env.action_masks()[9]
        obs, _, _, _, _ = env.step(10)
        assert p.discard[-1] == 'moat' and obs[V3_FEATURE_INDEX['owned_moat']] == 1
        np.testing.assert_array_equal(env.action_masks(), [False]*11+[True])
        env.step(11)
        assert env.opponent_reset_counts == counts and env.episode_opponent == 'militia-money'
        env.reset(); assert sum(env.opponent_reset_counts.values()) == 2
        env.game.state.supply['militia'] = 0; p = env.game.state.players[seat]; p.coins, p.buys = 8, 1
        assert not env.action_masks()[9] and env.action_masks()[10]
    finally:
        env.close()


@pytest.mark.parametrize('seat', [0, 1])
def test_five_new_feature_groups_have_exact_meanings(seat):
    env = DominionEnv(GameConfig(version='V3'), agent_player=seat)
    try:
        env.reset(seed=71); state = env.game.state; p = state.players[seat]
        p.hand.append('militia'); p.in_play.append('moat')
        state.supply['militia'] -= 1; state.supply['moat'] -= 1
        grant(env.game, seat, 'moat'); grant(env.game, 1-seat, 'militia')
        obs = encode_observation(state, seat)
        values = [obs[V3_FEATURE_INDEX[f'{prefix}_{card}']] for prefix in
            ('hand', 'in_play', 'owned', 'opponent_owned', 'supply') for card in ('militia', 'moat')]
        assert values == [1, 0, 0, 1, 1, 2, 1, 0, 8, 8]
        old = deepcopy(state); old.version = 'V2'
        np.testing.assert_array_equal(obs[:56], encode_observation(old, seat))
    finally:
        env.close()


@pytest.mark.parametrize('seat', [0, 1])
def test_attack_and_reaction_use_private_hand_without_exposing_it_in_observations_or_info(seat):
    env = DominionEnv(GameConfig(version='V3'), agent_player=seat)
    try:
        env.reset(seed=71); game = env.game; p = game.state.players[seat]; other = game.state.players[1-seat]
        game.state.supply['militia'] -= 1; p.hand.append('militia'); p.actions = 1
        game.state.supply['moat'] -= 1; other.hand.append('moat')
        game.phase = 'action'  # Isolated private attack/reaction fixture.
        different = deepcopy(game)
        hidden = different.state.players[1-seat]
        hidden.hand.remove('moat'); hidden.deck.append('moat')
        hidden.deck.reverse(); different.state.players[seat].deck.reverse()
        np.testing.assert_array_equal(encode_observation(game.state, seat), encode_observation(different.state, seat))
        before_ownership = [encode_observation(game.state, seat)[V3_FEATURE_INDEX[f'opponent_owned_{c}']] for c in V3_CARD_IDS]
        game.play_action('militia'); different.play_action('militia')
        assert game.militia_resolutions[-1]['moat_revealed'] and not different.militia_resolutions[-1]['moat_revealed']
        assert len(different.state.players[1-seat].hand) == 3
        obs = encode_observation(game.state, seat)
        np.testing.assert_array_equal(obs, encode_observation(different.state, seat))
        assert [obs[V3_FEATURE_INDEX[f'opponent_owned_{c}']] for c in V3_CARD_IDS] == before_ownership
        assert set(env._info(obs)) == {'action_mask', 'turn_number', 'current_player', 'agent_player',
            'episode_opponent', 'scores', 'turns_taken', 'winner', 'end_reason'}
        assert set(game.militia_resolutions[-1]) == {'attacker', 'defender', 'moat_revealed', 'discard_count'}
    finally:
        env.close()


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('card', ['militia', 'moat'])
def test_new_actions_dilute_p8_without_modeling_attack_or_drawing_power(seat, card):
    env = DominionEnv(GameConfig(version='V3'), agent_player=seat)
    try:
        env.reset(seed=71)
        for who in (0, 1):
            grant(env.game, who, 'gold'); grant(env.game, who, 'silver')
        before = encode_observation(env.game.state, seat)
        assert ownership_p8(before) == ownership_p8(before[:56])
        grant(env.game, seat, card)
        after = encode_observation(env.game.state, seat)
        assert ownership_p8(after) < ownership_p8(before)
        assert ownership_p8(after[:56]) == ownership_p8(before)
        assert ownership_p8(after) == ownership_p8(encode_observation(env.game.state, 1-seat), opponent=True)
        changed = after.copy(); changed[0] += 2
        assert ownership_p8(changed) == ownership_p8(after)
    finally:
        env.close()


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('objective', ['win_loss', 'vp_margin'])
@pytest.mark.parametrize('potential_name', ['vp', 'vp_economy'])
@pytest.mark.parametrize('true_terminal', [False, True])
def test_v3_terminal_pbrs_endpoints_and_retained_truncation_potential(seat, objective, potential_name, true_terminal):
    config = RewardConfig(mode='potential_based', potential=potential_name, terminal_objective=objective, gamma=1.)
    env = DominionEnv(GameConfig(version='V3', max_turns=2), config, agent_player=seat, opponent='militia-money')
    try:
        env.reset(seed=71)
        for card in ('province', 'militia', 'moat', 'gold', 'silver'):
            grant(env.game, seat, card)
        previous = encode_observation(env.game.state, seat)
        if true_terminal:
            env.game.state.supply['province'] = 0
        obs, reward, terminated, truncated, info = env.step(env.end_buying)
        assert terminated == true_terminal and truncated != true_terminal
        winner = None if info['winner'] is None else int(info['winner'] != seat)
        base = terminal_reward(winner, config, terminated=terminated, truncated=truncated, observation=obs)
        assert base == (0 if truncated else 6/48 if objective == 'vp_margin' else 1.)
        assert reward == pytest.approx(base + potential(obs, config, terminated=terminated) - potential(previous, config))
        if truncated:
            assert info['winner'] is None and potential(obs, config) > 0
        else:
            assert info['winner'] == seat and potential(obs, config, terminated=True) == 0
        assert not env.action_masks().any()
    finally:
        env.close()


def test_v3_opponents_reject_incompatible_versions_and_frozen_pools():
    for version in ('V0', 'V1', 'V2'):
        with pytest.raises(ValueError, match='V3'):
            DominionEnv(GameConfig(version=version), opponent='militia-money')
    for mode in ('historical', 'frozen'):
        with pytest.raises(ValueError, match='V1'):
            DominionEnv(GameConfig(version='V3'), opponent=mode, opponent_pool='missing.yaml',
                frozen_opponent_id='ppo_seed54' if mode == 'frozen' else None)
    for opponent in ('big-money', 'random', 'engine-money', 'festival-money', 'militia-money'):
        env = DominionEnv(GameConfig(version='V3'), opponent=opponent)
        try:
            a, _ = env.reset(seed=71); b, _ = env.reset(seed=71)
            np.testing.assert_array_equal(a, b)
        finally:
            env.close()
