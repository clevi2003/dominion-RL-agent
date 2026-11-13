"""Local V3 PPO save/load, public attack metrics, CLI and private demo checks."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.observations import encode_observation
from dominion_rl.env.preprocessing import preprocessing_config
from dominion_rl.ppo import game_contract, model_game_contract, model_preprocessing, play_episode, train_ppo
from test_v1_engine import arrange
from test_v1_usage import actor, serialized


def cli(name):
    path = Path(__file__).resolve().parents[1]/f'scripts/{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('protected', [False, True])
def test_v3_action_timing_plays_public_reactions_and_discards(monkeypatch, seat, protected):
    env = DominionEnv(GameConfig(version='V3', max_turns=2), agent_player=seat)
    reset = env.reset
    def arranged_reset(**kwargs):
        reset(**kwargs); game = env.game
        if protected:
            game.state.supply['moat'] -= 1; game.state.players[1-seat].hand.append('moat')
        arrange(game, ['festival', 'militia', 'gold'], [])
        game._start_turn()
        obs = encode_observation(game.state, seat)
        return obs, env._info(obs)
    monkeypatch.setattr(env, 'reset', arranged_reset)
    try:
        row = play_episode(actor(env, ['militia', 'moat']), env, 37000)
        assert row['militia_plays'] == row['turns_with_militia_plays'] == row['militia_attacks_by_agent'] == 1
        assert row['militia_coins_generated'] == 2 and row['moat_plays'] == 0
        assert row['moat_reactions_by_opponent'] == int(protected)
        assert row['militia_discards_by_opponent'] == (0 if protected else 2)
        assert row['militia_attacks_by_opponent'] == row['militia_discards_by_agent'] == 0
        assert row['purchases_militia'] == row['purchases_moat'] == 1
        assert row['first_militia_turn'] == row['first_moat_turn'] == 1
        assert row['final_owned_militia'] == 2 and row['final_owned_moat'] == 1
        assert row['turns_with_multiple_purchases'] == row['purchases_after_first'] == 1
        assert row['game_version'] == 'V3' and row['outcome'] == 'truncated'
    finally:
        env.close()


def test_no_attack_or_defense_purchase_has_blank_timing_and_zero_public_counts():
    env = DominionEnv(GameConfig(version='V3', max_turns=2))
    try:
        row = play_episode(actor(env, []), env, 37000)
        assert row['first_militia_turn'] is None and row['first_moat_turn'] is None
        assert serialized(row)['first_militia_turn'] == serialized(row)['first_moat_turn'] == ''
        assert row['militia_attacks_by_agent'] == row['militia_attacks_by_opponent'] == 0
        assert row['moat_reactions_by_agent'] == row['moat_reactions_by_opponent'] == 0
    finally:
        env.close()


def test_v3_short_train_periodic_final_loading_gamma_and_masking(tmp_path):
    reward = RewardConfig(gamma=1., terminal_objective='vp_margin')
    path = train_ppo(tmp_path/'run', game_config=GameConfig(version='V3', max_turns=4),
        reward_config=reward, seed=71, opponent='militia-money', total_timesteps=16, checkpoint_freq=8,
        ppo_kwargs=dict(gamma=1., n_steps=8, batch_size=8, n_epochs=1,
                        policy_kwargs=dict(net_arch=[64, 64], activation_fn='tanh')))
    metadata = yaml.safe_load((path.parent/'run.yaml').read_text())
    assert metadata['game_contract'] == game_contract('V3')
    assert metadata['reward']['terminal_objective'] == 'vp_margin' and metadata['training_gamma'] == 1.
    assert metadata['by_opponent']['militia-money']['decision_steps'] == 16
    for archive in (path, *sorted((path.parent/'checkpoints').glob('*.zip'))):
        loaded = MaskablePPO.load(archive, device='cpu')
        assert model_game_contract(loaded, metadata) == game_contract('V3')
        assert model_preprocessing(loaded) == preprocessing_config('raw', 'V3')
        for seat in (0, 1):
            env = DominionEnv(GameConfig(version='V3', max_turns=2), agent_player=seat, opponent='militia-money')
            try:
                obs, _ = env.reset(seed=37000)
                mask = env.action_masks()
                action, _ = loaded.predict(obs, deterministic=True, action_masks=mask)
                assert mask[int(action.item())]
                assert play_episode(loaded, env, 37000)['game_version'] == 'V3'
            finally:
                env.close()


def test_v3_gamma_mismatch_rejected_before_training(tmp_path):
    with pytest.raises(ValueError, match='gamma must match'):
        train_ppo(tmp_path/'must_not_exist', game_config=GameConfig(version='V3'),
            reward_config=RewardConfig(gamma=1.), ppo_kwargs=dict(gamma=.99), total_timesteps=16)
    assert not (tmp_path/'must_not_exist').exists()


def test_v3_training_cli_keeps_existing_ppo_architecture(monkeypatch, tmp_path):
    module = cli('train_ppo'); captured = {}
    monkeypatch.setattr(module, 'train_ppo', lambda path, **kw: captured.update(kw) or path)
    monkeypatch.setattr(sys, 'argv', ['train_ppo.py', '--game-config', 'configs/game_v3.yaml',
        '--opponent', 'militia-money', '--mode', 'terminal_only', '--terminal-objective', 'vp_margin',
        '--gamma', '1.0', '--seed', '71', '--timesteps', '1024', '--output', str(tmp_path/'run')])
    module.main()
    assert captured['game_config'].version == 'V3' and captured['opponent'] == 'militia-money'
    assert captured['ppo_kwargs']['policy_kwargs'] == dict(net_arch=[64, 64], activation_fn='tanh')
    assert captured['reward_config'].gamma == captured['ppo_kwargs']['gamma'] == 1.


def test_v3_evaluation_cli_multiple_checkpoints_both_seats_and_opponents(monkeypatch, tmp_path):
    module = cli('evaluate_ppo'); env = DominionEnv(GameConfig(version='V3', max_turns=2))
    model = actor(env, []); model.gamma = 1.; model.seed = 71; model.num_timesteps = 1024
    model.training_reward_config = asdict(RewardConfig(gamma=1., terminal_objective='vp_margin'))
    model.policy = SimpleNamespace(activation_fn=type('Tanh', (), {}))
    paths = [tmp_path/'periodic.zip', tmp_path/'final.zip']
    for path in paths:
        path.write_bytes(b'hash sentinel; no policy loaded')
    monkeypatch.setattr(module.MaskablePPO, 'load', lambda *a, **k: model)
    output = tmp_path/'games.csv'
    monkeypatch.setattr(sys, 'argv', ['evaluate_ppo.py', '--model', *map(str, paths),
        '--game-config', 'configs/game_v3.yaml', '--opponent', 'big-money', 'militia-money',
        '--agent-player', '0', '1', '--max-turns', '2', '--games', '1', '--output', str(output)])
    try:
        module.main()
        import csv
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 8 and all(r['game_version'] == 'V3' for r in rows)
        assert all(r['first_militia_turn'] == r['first_moat_turn'] == '' for r in rows)
        assert {r['opponent'] for r in rows} == {'big-money', 'militia-money'}
        assert all(r['moat_reactions_by_agent'] == '0' for r in rows)
    finally:
        env.close()
