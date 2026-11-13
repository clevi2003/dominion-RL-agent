"""Append-only V2 schemas, information boundaries, masks and reward endpoints."""
from copy import deepcopy

import numpy as np
import pytest

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.observations import (
    FEATURE_NAMES, V1_FEATURE_NAMES, V1_FEATURE_INDEX, V2_FEATURE_NAMES,
    V2_FEATURE_INDEX, encode_observation,
)
from dominion_rl.env.preprocessing import preprocessing_config
from dominion_rl.env.rewards import calculate_reward, five_card_p8, ownership_p8, potential, terminal_reward
from dominion_rl.game.cards import V1_CARD_IDS, V2_CARD_IDS
from test_v1_env import grant


def test_all_version_dimensions_and_preserved_indices():
    for version, n_obs, n_actions in (('V0', 41, 7), ('V1', 51, 9), ('V2', 56, 10)):
        env = DominionEnv(GameConfig(version=version))
        try:
            obs, _ = env.reset(seed=42)
            assert obs.shape == (n_obs,) and env.action_space.n == n_actions
            assert env.observation_space.contains(obs)
        finally:
            env.close()
    assert V2_FEATURE_NAMES[:51] == V1_FEATURE_NAMES
    assert V2_FEATURE_NAMES[:41] == FEATURE_NAMES
    assert all(V2_FEATURE_INDEX[n] == i for n, i in V1_FEATURE_INDEX.items())
    assert V2_CARD_IDS[:8] == V1_CARD_IDS
    assert V2_FEATURE_NAMES[51:] == tuple(f'{prefix}_festival' for prefix in
        ('hand', 'in_play', 'owned', 'opponent_owned', 'supply'))
    assert preprocessing_config('raw', 'V2')['divisors'] == [1] * 56
    with pytest.raises(ValueError, match='raw'):
        preprocessing_config('fixed', 'V2')


@pytest.mark.parametrize('seat', [0, 1])
def test_v2_festival_purchase_mask_discard_and_supply_depletion(seat):
    env = DominionEnv(GameConfig(version='V2', max_turns=8), agent_player=seat)
    try:
        env.reset(seed=42)
        assert env.end_buying == 9 and env.action_names[:8] == V1_CARD_IDS
        p = env.game.state.players[seat]
        p.coins, p.buys = 4, 2
        assert not env.action_masks()[8] and env.action_masks()[9]
        with pytest.raises(ValueError):
            env.step(8)
        p.coins = 10
        env.game.state.supply['festival'] = 1
        obs, _, terminated, truncated, _ = env.step(8)
        assert not terminated and not truncated
        assert (p.coins, p.buys) == (5, 1) and p.discard[-1] == 'festival'
        assert obs[V2_FEATURE_INDEX['owned_festival']] == 1
        assert obs[V2_FEATURE_INDEX['supply_festival']] == 0
        assert not env.action_masks()[8] and env.action_masks()[7] and env.action_masks()[9]
        env.step(7)
        np.testing.assert_array_equal(env.action_masks(), [False] * 9 + [True])
    finally:
        env.close()


@pytest.mark.parametrize('seat', [0, 1])
def test_festival_feature_groups_and_hidden_zone_invariance(seat):
    env = DominionEnv(GameConfig(version='V2'), agent_player=seat)
    try:
        env.reset(seed=42)
        state = env.game.state
        p, other = state.players[seat], state.players[1 - seat]
        p.hand.append('festival'); p.in_play.append('festival'); p.discard.append('festival')
        other.deck.append('festival'); other.hand.append('festival')
        state.supply['festival'] -= 5
        obs = encode_observation(state, seat)
        assert [obs[V2_FEATURE_INDEX[f'{prefix}_festival']] for prefix in
                ('hand', 'in_play', 'owned', 'opponent_owned', 'supply')] == [1, 1, 3, 2, 5]
        old = deepcopy(state); old.version = 'V1'
        np.testing.assert_array_equal(obs[:51], encode_observation(old, seat))
        env.game.draw(other, 1)  # Private draw changes no public ownership.
        np.testing.assert_array_equal(obs, encode_observation(state, seat))
        changed = deepcopy(state)
        other = changed.players[1 - seat]
        cards = other.deck + other.hand + other.discard + other.in_play
        other.deck, other.hand, other.discard, other.in_play = [], cards[:1], cards[1:], []
        other.coins, other.buys, other.actions = 999, 99, 99
        changed.players[seat].deck.reverse()
        np.testing.assert_array_equal(obs, encode_observation(changed, seat))
        assert set(env._info(obs)) == {'action_mask', 'turn_number', 'current_player', 'agent_player',
            'episode_opponent', 'scores', 'turns_taken', 'winner', 'end_reason'}
    finally:
        env.close()


@pytest.mark.parametrize('seat', [0, 1])
def test_p8_counts_festival_as_zero_treasure_for_both_players(seat):
    env = DominionEnv(GameConfig(version='V2'), agent_player=seat)
    try:
        env.reset(seed=42)
        for who in (0, 1):
            grant(env.game, who, 'gold'); grant(env.game, who, 'silver')
        before = encode_observation(env.game.state, seat)
        assert ownership_p8(before) == ownership_p8(before[:51]) == ownership_p8(before[:41])
        original = ownership_p8(before)
        grant(env.game, seat, 'festival')
        after = encode_observation(env.game.state, seat)
        assert ownership_p8(after) == five_card_p8(7, 1, 1, 4) < original
        assert ownership_p8(after[:51]) == original  # Prefix deliberately excludes Festival.
        assert ownership_p8(after) == ownership_p8(encode_observation(env.game.state, 1 - seat), opponent=True)
        # Actual turn coins (including Action-generated coins) do not enter P8.
        changed = after.copy(); changed[0] += 2
        assert ownership_p8(after) == ownership_p8(changed)
    finally:
        env.close()


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('objective', ['win_loss', 'vp_margin'])
@pytest.mark.parametrize('design', ['terminal_only', 'vp', 'vp_economy'])
@pytest.mark.parametrize('gamma', [.99, 1.])
@pytest.mark.parametrize('true_terminal', [False, True])
def test_v2_reward_endpoints_seats_and_truncation(seat, objective, design, gamma, true_terminal):
    reward = RewardConfig(mode='terminal_only' if design == 'terminal_only' else 'potential_based',
        potential='vp_economy' if design == 'vp_economy' else 'vp', gamma=gamma, terminal_objective=objective)
    env = DominionEnv(GameConfig(version='V2', max_turns=2), reward, agent_player=seat)
    try:
        env.reset(seed=42)
        for card in ('province', 'festival', 'gold', 'silver'):
            grant(env.game, seat, card)
        before = encode_observation(env.game.state, seat)
        if true_terminal:
            env.game.state.supply['province'] = 0
        obs, actual, terminated, truncated, info = env.step(env.end_buying)
        assert terminated == true_terminal and truncated != true_terminal
        winner = None if info['winner'] is None else int(info['winner'] != seat)
        expected_base = 0 if truncated else (info['scores'][0] - info['scores'][1]) / 48 if objective == 'vp_margin' else 1
        assert terminal_reward(winner, reward, terminated=terminated, truncated=truncated, observation=obs) == expected_base
        shaped = 0 if design == 'terminal_only' else gamma * potential(obs, reward, terminated=terminated) - potential(before, reward)
        assert actual == pytest.approx(expected_base + shaped)
        assert actual == calculate_reward(before, obs, reward, terminated=terminated, truncated=truncated, winner=winner)
        assert potential(obs, reward, terminated=True) == 0
        assert not env.action_masks().any()
        if truncated:
            assert info['winner'] is None and potential(obs, reward) > 0
        else:
            assert info['winner'] == seat
    finally:
        env.close()


def test_v2_frozen_opponents_rejected_before_loading():
    # Deliberately nonexistent pool: reject the schema before opening it or
    # importing optional neural dependencies.
    for mode in ('historical', 'frozen'):
        with pytest.raises(ValueError, match='V1'):
            DominionEnv(GameConfig(version='V2'), opponent=mode, opponent_pool='unused.yaml',
                        frozen_opponent_id='ppo_seed54' if mode == 'frozen' else None)
    with pytest.raises(ValueError, match='V2'):
        DominionEnv(GameConfig(version='V1'), opponent='festival-money')
