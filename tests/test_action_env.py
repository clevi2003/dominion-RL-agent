"""Manual phases, tactical decisions, fixed purchases and information boundaries."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from dominion_rl.agents.action_rich import action_rich_purchase, buying_config
from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.action_env import (ACTION_CARDS, ACTION_FEATURES, ACTION_NAMES, END_ACTION,
    DominionActionEnv, action_contract)
from dominion_rl.env.observations import V3_FEATURE_NAMES, encode_observation
from dominion_rl.game import DominionGame
from test_v1_engine import arrange, total_cards


def scene(hand, draws=(), *, seat=0, max_turns=20):
    env = DominionActionEnv(GameConfig(version='V3', max_turns=max_turns),
                            agent_player=seat, record_decisions=True)
    env.reset(seed=39000)
    player = arrange(env.game, list(hand), list(draws))
    env.game._start_turn()
    return env, player


def test_manual_initialization_and_strict_phase_transitions():
    game = DominionGame(GameConfig(version='V3'), automatic_actions=False)
    p = game.state.players[0]
    assert game.phase == 'action' and (p.actions, p.buys, p.coins) == (1, 1, 0)
    assert len(p.hand) == 5 and not p.in_play and not game.legal_purchases()
    before = deepcopy(game.state)
    with pytest.raises(ValueError): game.buy_card('copper')
    with pytest.raises(ValueError): game.end_buying()
    assert game.state == before
    coins = p.hand.count('copper')
    game.end_action_phase()
    assert game.phase == 'buy' and p.coins == coins
    with pytest.raises(ValueError): game.end_action_phase()
    p.hand.append('moat'); p.actions = 1
    with pytest.raises(ValueError): game.play_action('moat')
    game.end_buying()
    assert game.phase == 'action' and game.state.current_player == 1


@pytest.mark.parametrize('version', ['V0', 'V1', 'V2', 'V3'])
def test_automatic_default_matches_manual_historical_priority_including_rng(version):
    auto = DominionGame(GameConfig(version=version, seed=72))
    manual = DominionGame(GameConfig(version=version, seed=72), automatic_actions=False)
    hand = ['village', 'festival', 'council_room', 'militia', 'moat', 'gold']
    hand = [c for c in hand if c in auto.config.cards]
    for game in (auto, manual):
        arrange(game, hand, ['silver', 'copper', 'estate', 'gold', 'estate', 'copper', 'estate'])
        game._start_turn()
    while (card := manual.automatic_action_choice()) is not None:
        manual.play_action(card)
    manual.end_action_phase()
    assert auto.phase == manual.phase == 'buy'
    assert asdict(auto.state) == asdict(manual.state)
    assert auto._rng.getstate() == manual._rng.getstate()
    assert auto.militia_resolutions == manual.militia_resolutions


@pytest.mark.parametrize('seat', [0, 1])
def test_contract_masks_raw_observations_and_actions_remaining(seat):
    env, p = scene(['village', 'militia', 'moat'], ['festival'], seat=seat)
    try:
        obs = env._observation()
        assert ACTION_FEATURES == (*V3_FEATURE_NAMES, 'actions_remaining')
        assert ACTION_NAMES == ('village', 'council_room', 'festival', 'militia', 'moat', 'end_action')
        assert len(action_contract()['features']) == 67 and env.action_space.n == 6
        assert env.observation_space.contains(obs) and obs[-1] == 1
        np.testing.assert_array_equal(obs[:66], encode_observation(env.game.state, seat))
        np.testing.assert_array_equal(env.action_masks(), [True, False, False, True, True, True])
        obs, reward, terminated, truncated, _ = env.step(0)
        assert obs[-1] == 2 and p.hand.count('festival') == 1
        assert env.action_masks()[2] and reward == 0 and not (terminated or truncated)
        env.step(2)
        assert p.actions == 3 and p.buys == 2 and p.coins == 2
    finally: env.close()


def test_duplicates_are_played_one_copy_at_a_time():
    env, p = scene(['village', 'village', 'moat'], ['estate', 'estate'])
    try:
        total = total_cards(env.game)
        env.step(0)
        assert p.hand.count('village') == p.in_play.count('village') == 1 and p.actions == 2
        env.step(0)
        assert p.hand.count('village') == 0 and p.in_play.count('village') == 2 and p.actions == 3
        assert not env.action_masks()[0] and env.action_masks()[4]
        assert total_cards(env.game) == total
    finally: env.close()


@pytest.mark.parametrize('invalid', [-1, 6, True, np.bool_(False), 3.5, 1])
def test_invalid_masked_decisions_never_coerce_to_end(invalid):
    env, _ = scene(['militia', 'moat'])
    try:
        before = asdict(env.game.state)
        with pytest.raises(ValueError): env.step(invalid)
        assert asdict(env.game.state) == before and env.metrics['action_decisions'] == 0
    finally: env.close()


def test_voluntary_end_plays_treasures_buys_fixed_and_advances():
    env, p = scene(['militia', 'moat', 'gold', 'silver'], max_turns=1)
    try:
        _, reward, term, trunc, info = env.step(END_ACTION)
        assert trunc and not term and reward == 0
        assert env.metrics['voluntary_end_with_playable'] == 1
        assert env.metrics['militia_moat_choices'] == env.metrics['multiple_type_decisions'] == 1
        assert env.metrics['plays_militia'] == env.metrics['plays_moat'] == 0
        assert env.metrics['purchases_silver'] == 1 and env.metrics['buy_phase_coins'] == 5
        assert p.turns_taken == 1 and not env.game.militia_resolutions
        assert info['winner'] is None and not env.action_masks().any()
        with pytest.raises(RuntimeError): env.step(END_ACTION)
    finally: env.close()


def test_council_draws_immediately_then_fixed_buyer_uses_multiple_buys():
    env, p = scene(['festival', 'council_room', 'gold'], ['gold', 'silver', 'copper', 'estate'], max_turns=1)
    try:
        other = env.game.state.players[1]
        other_size = len(other.hand)
        env.step(2)  # Festival keeps room eligible.
        assert p.actions == 2 and p.buys == 2 and p.coins == 2
        _, _, _, truncated, _ = env.step(1)  # No other Actions; automatic Treasure/Buy/cleanup.
        assert truncated and len(other.hand) == other_size+1
        assert env.metrics['extra_buys_generated'] == 2
        assert env.metrics['buy_phase_coins'] == 11
        assert env.metrics['purchases_province'] == env.metrics['purchases_silver'] == 1
        assert env.metrics['extra_buys_used'] == env.metrics['multiple_purchase_turns'] == 1
        assert env.metrics['multiple_action_turns'] == 1
    finally: env.close()


@pytest.mark.parametrize('protected', [False, True])
def test_private_militia_discards_and_automatic_moat_reactions(protected):
    env, p = scene(['festival', 'militia', 'moat'])
    try:
        game = env.game
        game.state.current_player = 1
        defender = arrange(game, ['moat' if protected else 'estate', 'copper', 'copper', 'silver', 'gold'], [])
        game.state.current_player = 0
        before = Counter(defender.hand+defender.discard+defender.deck+defender.in_play)
        env.step(2); env.step(3)  # Another Moat remains playable, so do not advance opponent.
        assert game.phase == 'action' and env.action_masks()[4]
        event = game.militia_resolutions[-1]
        assert event == dict(attacker=0, defender=1, moat_revealed=protected, discard_count=0 if protected else 2)
        assert len(defender.hand) == (5 if protected else 3)
        assert Counter(defender.hand+defender.discard+defender.deck+defender.in_play) == before
        assert 'opponent_hand' not in env._info() and 'discarded_cards' not in str(event)
    finally: env.close()


@pytest.mark.parametrize('seat', [0, 1])
def test_no_private_zones_or_future_draws_in_observation_masks_info_or_context(seat):
    env, _ = scene(['festival', 'militia', 'moat'], ['gold', 'silver'], seat=seat)
    try:
        hidden = deepcopy(env)
        other = hidden.game.state.players[1-seat]
        # Preserve public ownership while redistributing hidden zones arbitrarily.
        other.deck += other.hand; other.hand = []; other.deck.reverse()
        hidden.game.state.players[seat].deck.reverse()
        np.testing.assert_array_equal(env._observation(), hidden._observation())
        np.testing.assert_array_equal(env.action_masks(), hidden.action_masks())
        left, right = env._info(), hidden._info()
        left.pop('action_mask'); right.pop('action_mask'); assert left == right
        env.step(2); hidden.step(2)
        assert env.decisions == hidden.decisions
        assert set(env.decisions[0]) == {'turn', 'actions', 'coins', 'buys', 'playable_types', 'legal_types', 'hand_counts', 'choice'}
    finally: env.close()


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('ending', ['province_empty', 'three_piles_empty', 'turn_limit'])
def test_episode_endings_terminal_margin_and_both_seats(seat, ending):
    env, p = scene(['estate'], seat=seat, max_turns=2)
    try:
        depleted = ['province'] if ending == 'province_empty' else ['village', 'festival', 'moat'] if ending == 'three_piles_empty' else []
        for card in depleted:
            p.discard.extend([card]*env.game.state.supply[card]); env.game.state.supply[card] = 0
        obs, reward, term, trunc, info = env.step(END_ACTION)
        assert term == (ending != 'turn_limit') and trunc != term
        assert info['end_reason'] == ending
        assert reward == (0 if trunc else (info['scores'][0]-info['scores'][1])/48)
        assert not env.action_masks().any() and env.observation_space.contains(obs)
        if trunc: assert info['winner'] is None
    finally: env.close()


@pytest.mark.parametrize('seat', [0, 1])
def test_no_action_turns_are_forced_and_seed_replay_is_exact(seat):
    env = DominionActionEnv(GameConfig(version='V3', max_turns=6), agent_player=seat)
    try:
        traces = []
        for repeat in range(2):
            obs, _ = env.reset(seed=39000); history = []
            while True:
                np.testing.assert_array_equal(env.action_masks(), [False]*5+[True])
                obs, reward, term, trunc, info = env.step(END_ACTION)
                history.append((obs.tolist(), reward, term, trunc, info['scores'], dict(env.metrics)))
                if term or trunc: break
            assert env.metrics['action_decisions'] == env.metrics['forced_end_decisions'] > 0
            traces.append(history)
        assert traces[0] == traces[1]
    finally: env.close()


def test_buyer_is_bounded_economic_and_saved_config_cannot_mutate_rules():
    legal = list(GameConfig(version='V3').cards)
    owned = Counter()
    assert action_rich_purchase(legal, 8, owned) == 'province'
    legal.remove('province')
    assert action_rich_purchase(legal, 8, owned) == 'gold'
    owned.update(gold=2, silver=3)
    for _ in range(7):
        choice = action_rich_purchase(legal, 8, owned)
        assert choice in ACTION_CARDS
        owned[choice] += 1
    assert {card:owned[card] for card in ACTION_CARDS} == buying_config()['limits']
    assert action_rich_purchase(legal, 8, owned) == 'gold'
    assert action_rich_purchase(legal, 4, owned) == 'duchy'
    assert action_rich_purchase(['copper'], 8, owned) is None
    rules = buying_config(); rules['limits']['militia'] = 900
    assert buying_config()['limits']['militia'] == 1


def test_action_environment_rejects_changed_rewards_and_other_versions():
    with pytest.raises(ValueError): DominionActionEnv(GameConfig(version='V2'))
    with pytest.raises(ValueError): DominionActionEnv(reward_config=RewardConfig())
    with pytest.raises(ValueError): DominionActionEnv(opponent='random')
    with pytest.raises(ValueError): DominionActionEnv(GameConfig(version='V3', max_turns=1), agent_player=1)
    old = DominionEnv(GameConfig(version='V3'))
    assert old.observation_space.shape == (66,) and old.action_space.n == 12
