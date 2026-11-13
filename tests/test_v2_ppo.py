"""Local V2 native archive, metric and CLI checks; no training on import."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.preprocessing import preprocessing_config
from dominion_rl.ppo import (game_contract, model_game_contract, model_preprocessing,
    model_reward_config, play_episode, train_ppo)
from test_v1_usage import actor, short_game, serialized


@pytest.mark.parametrize('seat', [0, 1])
def test_festival_usage_purchases_timing_and_multiple_buy_metrics(monkeypatch, seat):
    env = short_game(monkeypatch, seat, ['festival', 'council_room'],
                     ['gold', 'silver', 'copper', 'estate'], version='V2')
    try:
        row = play_episode(actor(env, ['festival', 'silver', 'copper']), env, 35000)
        assert row['festival_plays'] == row['turns_with_festival_plays'] == row['festival_extra_buys_granted'] == 1
        assert row['council_room_plays'] == row['council_room_extra_buys_granted'] == 1
        assert row['turns_with_multiple_purchases'] == row['turns_with_non_copper_extra_purchase_opportunity'] == 1
        assert row['purchases_festival'] == 1 and row['first_festival_turn'] == 1
        assert row['first_council_room_turn'] is None and row['village_plays'] == 0
        assert row['observed_agent_turns'] == 1 and row['outcome'] == 'truncated'
        assert row['festival_coins_generated'] == 2
        assert row['turns_with_festival_and_council_room_plays'] == 1
        assert row['purchases_after_first'] == 2 and row['non_copper_purchases_after_first'] == 1
        assert row['owned_card_turns_festival'] == row['turns_owning_festival'] == 1
        assert row['turns_owning_without_play_festival'] == 0
        assert row['final_owned_festival'] == 2  # Fixture's played copy plus the purchase.

        csv_row = serialized(row)
        assert csv_row['first_council_room_turn'] == '' and csv_row['purchases_festival'] == '1'
    finally:
        env.close()


def test_festival_never_bought_has_missing_timing(monkeypatch):
    env = short_game(monkeypatch, 0, ['festival'], [], version='V2')
    try:
        row = play_episode(actor(env, []), env, 35000)
        assert row['festival_plays'] == 1 and row['purchases_festival'] == 0
        assert row['first_festival_turn'] is None and serialized(row)['first_festival_turn'] == ''
    finally:
        env.close()


@pytest.mark.parametrize('version', ['V0', 'V1', 'V2', 'V3'])
def test_native_archives_preserve_own_version_and_reject_other_versions(tmp_path, version):
    env = DominionEnv(GameConfig(version=version, max_turns=2))
    try:
        model = MaskablePPO('MlpPolicy', env, seed=60, n_steps=2, batch_size=2, device='cpu')
        model.game_contract = env.game_contract
        model.observation_preprocessing = preprocessing_config('raw', version)
        model.save(tmp_path / 'model.zip')
        loaded = MaskablePPO.load(tmp_path / 'model.zip', device='cpu')
        assert model_game_contract(loaded) == game_contract(version)
        assert model_preprocessing(loaded) == preprocessing_config('raw', version)
        assert play_episode(loaded, env, 35000)['game_version'] == version
        for other in ('V0', 'V1', 'V2', 'V3'):
            if other == version:
                continue
            other_env = DominionEnv(GameConfig(version=other, max_turns=2))
            try:
                with pytest.raises(ValueError, match='schema'):
                    play_episode(loaded, other_env, 35000)
            finally:
                other_env.close()
    finally:
        env.close()


def test_v2_short_train_save_load_checkpoint_and_masking(tmp_path):
    reward = RewardConfig(gamma=1., terminal_objective='vp_margin')
    path = train_ppo(tmp_path / 'run', game_config=GameConfig(version='V2', max_turns=4),
        reward_config=reward, seed=60, opponent='festival-money', total_timesteps=16, checkpoint_freq=8,
        ppo_kwargs=dict(gamma=1., n_steps=8, batch_size=8, n_epochs=1,
                        policy_kwargs=dict(net_arch=[64, 64], activation_fn='tanh')))
    metadata = yaml.safe_load((path.parent / 'run.yaml').read_text())
    assert metadata['game_contract'] == game_contract('V2')
    assert metadata['by_opponent']['festival-money']['decision_steps'] == 16
    for archive in (path, *sorted((path.parent / 'checkpoints').glob('*.zip'))):
        loaded = MaskablePPO.load(archive, device='cpu')
        assert model_game_contract(loaded, metadata) == game_contract('V2')
        assert model_reward_config(loaded, metadata) == reward
        env = DominionEnv(GameConfig(version='V2', max_turns=2), opponent='festival-money')
        try:
            obs, _ = env.reset(seed=35000)
            action, _ = loaded.predict(obs, deterministic=True, action_masks=env.action_masks())
            assert env.action_masks()[int(action.item())]
            assert play_episode(loaded, env, 35000)['game_version'] == 'V2'
        finally:
            env.close()


def cli(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_v2_training_cli_uses_existing_config_and_fixed_heuristic(monkeypatch, tmp_path):
    module = cli('train_ppo'); captured = {}
    monkeypatch.setattr(module, 'train_ppo', lambda output, **kw: captured.update(kw) or output)
    monkeypatch.setattr(sys, 'argv', ['train_ppo.py', '--game-config', 'configs/game_v2.yaml',
        '--opponent', 'festival-money', '--mode', 'terminal_only', '--terminal-objective', 'vp_margin',
        '--gamma', '1.0', '--timesteps', '1024', '--output', str(tmp_path / 'run')])
    module.main()
    assert captured['game_config'].version == 'V2' and captured['opponent'] == 'festival-money'
    assert captured['reward_config'].gamma == captured['ppo_kwargs']['gamma'] == 1.
    assert captured['reward_config'].terminal_objective == 'vp_margin'


def test_v2_multiple_checkpoint_evaluation_cli_has_festival_fields(monkeypatch, tmp_path):
    module = cli('evaluate_ppo')
    env = DominionEnv(GameConfig(version='V2', max_turns=2))
    model = actor(env, [])
    model.gamma = 1.; model.seed = 60; model.num_timesteps = 1024
    model.training_reward_config = asdict(RewardConfig(gamma=1., terminal_objective='vp_margin'))
    from types import SimpleNamespace
    model.policy = SimpleNamespace(activation_fn=type('Tanh', (), {}))
    paths = [tmp_path / 'step0.zip', tmp_path / 'final.zip']
    for path in paths:
        path.write_bytes(b'archive hash sentinel')
    monkeypatch.setattr(module.MaskablePPO, 'load', lambda *a, **k: model)
    output = tmp_path / 'games.csv'
    monkeypatch.setattr(sys, 'argv', ['evaluate_ppo.py', '--model', *map(str, paths),
        '--game-config', 'configs/game_v2.yaml', '--max-turns', '2', '--games', '1',
        '--opponent', 'big-money', 'festival-money', '--agent-player', '0', '1', '--output', str(output)])
    try:
        module.main()
        import csv
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 8
        assert all(r['game_version'] == 'V2' and r['purchases_festival'] == '0' and
                   r['first_festival_turn'] == '' and r['training_seed'] == '60' for r in rows)
        assert {r['opponent'] for r in rows} == {'big-money', 'festival-money'}
    finally:
        env.close()


def test_v2_rejects_reward_ppo_gamma_mismatch_before_training(tmp_path):
    output = tmp_path / 'must_not_exist'
    with pytest.raises(ValueError, match='gamma must match'):
        train_ppo(output, game_config=GameConfig(version='V2'),
            reward_config=RewardConfig(gamma=1.), ppo_kwargs=dict(gamma=.99), total_timesteps=16)
    assert not output.exists()
