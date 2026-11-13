"""Focused local persistence/CLI checks, never full experiment training."""
import csv
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
import torch
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.ppo import model_reward_config, model_training_gamma, train_ppo


def cli(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name+'_objective_test',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('version',['V0','V1'])
@pytest.mark.parametrize('objective',['win_loss','vp_margin'])
def test_saved_models_record_and_recover_objective_and_gamma(tmp_path,version,objective):
    config = RewardConfig(gamma=1.,terminal_objective=objective)
    path = train_ppo(tmp_path/'run',game_config=GameConfig(version=version,max_turns=2),reward_config=config,seed=54,
        opponent='random',total_timesteps=16,checkpoint_freq=16,
        ppo_kwargs=dict(gamma=1.,gae_lambda=.95,n_steps=16,batch_size=8,n_epochs=1,policy_kwargs={'net_arch':[16,16]}))
    snapshot = yaml.safe_load((path.parent/'run.yaml').read_text())
    assert snapshot['terminal_objective'] == objective and snapshot['training_gamma'] == 1.
    assert snapshot['reward'] == asdict(config)
    for archive in (path,*sorted((path.parent/'checkpoints').glob('*.zip'))):
        model = MaskablePPO.load(archive,device='cpu')
        assert model_reward_config(model,snapshot) == model_reward_config(model) == config
        assert model_training_gamma(model,snapshot) == 1.
        assert model.training_reward_config['terminal_objective'] == objective


@pytest.mark.parametrize('version',['V0','V1'])
def test_historical_archive_missing_objective_defaults_to_win_loss(tmp_path,version):
    env = DominionEnv(GameConfig(version=version,max_turns=2))
    try:
        model = MaskablePPO('MlpPolicy',env,seed=51,n_steps=2,batch_size=2,device='cpu')
        legacy = asdict(RewardConfig());legacy.pop('terminal_objective')
        model.training_reward_config = legacy;model.game_contract = env.game_contract
        model.save(tmp_path/'old.zip')
        loaded = MaskablePPO.load(tmp_path/'old.zip',device='cpu')
        assert model_reward_config(loaded,{'reward':legacy}) == RewardConfig()
        assert model_reward_config(loaded).terminal_objective == 'win_loss'
        assert model_training_gamma(loaded) == .99
        del loaded.training_reward_config
        assert model_reward_config(loaded,{'reward':legacy}).terminal_objective == 'win_loss'
        assert model_reward_config(loaded) is None  # No reward metadata: remain unknown.
    finally:env.close()


def test_conflicting_objective_metadata_is_rejected():
    model = SimpleNamespace(gamma=1.,training_reward_config=asdict(RewardConfig(gamma=1.,terminal_objective='vp_margin')))
    with pytest.raises(ValueError,match='objective disagree'):
        model_reward_config(model,{'terminal_objective':'win_loss'})
    with pytest.raises(ValueError,match='configuration disagree'):
        model_reward_config(model,{'reward':asdict(RewardConfig(gamma=1.))})


@pytest.mark.parametrize('objective',['win_loss','vp_margin'])
def test_training_cli_selects_objective_and_both_gamma_values(monkeypatch,tmp_path,objective):
    module = cli('train_ppo');captured = {}
    def capture(output,**kwargs):captured.update(kwargs);return Path(output)/'model.zip'
    monkeypatch.setattr(module,'train_ppo',capture)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys,'argv',['train_ppo.py','--terminal-objective',objective,'--mode','terminal_only','--gamma','1.0',
        '--output',str(tmp_path/'run')])
    module.main()
    assert captured['reward_config'].terminal_objective == objective
    assert captured['reward_config'].gamma == captured['ppo_kwargs']['gamma'] == 1.
    assert captured['reward_config'].mode == 'terminal_only' and captured['ppo_kwargs']['gae_lambda'] == .95


@pytest.mark.parametrize('objective',['win_loss','vp_margin'])
@pytest.mark.parametrize('source',['terminal','training'])
def test_evaluation_recovers_objective_gamma_and_labels_common_outcomes(monkeypatch,tmp_path,objective,source):
    module = cli('evaluate_ppo');config = RewardConfig(gamma=1.,terminal_objective=objective)
    env = DominionEnv(GameConfig(version='V1'))
    model = SimpleNamespace(gamma=1.,num_timesteps=16,seed=54,training_reward_config=asdict(config),game_contract=env.game_contract,
        policy=SimpleNamespace(activation_fn=torch.nn.Tanh),observation_space=env.observation_space,action_space=env.action_space)
    if objective == 'win_loss':model.training_reward_config.pop('terminal_objective')  # Legacy native contract.
    env.close();monkeypatch.setattr(module.MaskablePPO,'load',lambda *args,**kwargs:model)
    captured = []
    def episode(policy,env,seed,**kwargs):
        captured.append((env.reward_config,kwargs['return_gamma']));return dict(outcome='loss',evaluation_seed=seed)
    monkeypatch.setattr(module,'play_episode',episode)
    archive,output = tmp_path/'moved.zip',tmp_path/'games.csv';archive.write_bytes(b'archive hash sentinel')
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys,'argv',['evaluate_ppo.py','--model',str(archive),'--game-config','configs/game_v1.yaml','--games','1',
        '--opponent','engine-money','--agent-player','1','--reward-source',source,'--output',str(output)])
    module.main()
    expected = config if source == 'training' else RewardConfig(gamma=1.)
    assert captured == [(expected,1.)]
    m = yaml.safe_load(output.with_suffix('.yaml').read_text())['models'][0]
    assert m['training_terminal_objective'] == objective and m['evaluation_terminal_objective'] == expected.terminal_objective
    assert m['training_gamma'] == m['evaluation_return_gamma'] == 1.
    with output.open() as stream:row = next(csv.DictReader(stream))
    assert row['training_terminal_objective'] == objective and row['evaluation_terminal_objective'] == expected.terminal_objective
