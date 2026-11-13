"""Small local persistence/CLI checks; no full experiment jobs."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
import torch
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.ppo import model_reward_config, train_ppo


@pytest.mark.parametrize('design', ['terminal_only','vp','vp_economy'])
def test_full_reward_contract_survives_every_archive_and_relocation(tmp_path, design):
    config = RewardConfig(mode='terminal_only' if design=='terminal_only' else 'potential_based',
                          potential='vp_economy' if design=='vp_economy' else 'vp')
    path = train_ppo(tmp_path/'run',game_config=GameConfig(max_turns=2),reward_config=config,seed=47,
        total_timesteps=16,checkpoint_freq=16,opponent='random',
        ppo_kwargs=dict(gamma=.99,n_steps=16,batch_size=8,n_epochs=1,policy_kwargs={'net_arch':[16,16]}))
    metadata = yaml.safe_load((path.parent/'run.yaml').read_text())
    assert metadata['reward']==asdict(config) and metadata['reward_design']==design
    for archive in (path,*sorted((path.parent/'checkpoints').glob('*.zip'))):
        model = MaskablePPO.load(archive,device='cpu')
        assert model.training_reward_config==asdict(config)
        assert model_reward_config(model,metadata)==model_reward_config(model)==config
    moved = tmp_path/'moved.zip'
    shutil.copyfile(path,moved)
    assert model_reward_config(MaskablePPO.load(moved,device='cpu'))==config


def test_legacy_reward_recovery_and_conflicting_metadata():
    legacy = dict(mode='potential_based',gamma=.99,shaping_strength=1.,vp_normalization=48.,
                  win_reward=1.,tie_reward=0.,loss_reward=-1.)
    model = SimpleNamespace(gamma=.99)
    assert model_reward_config(model) is None  # Unrecorded legacy rewards stay unknown.
    assert model_reward_config(model,{'reward':legacy})==RewardConfig(mode='potential_based')
    model.training_reward_config = asdict(RewardConfig(mode='potential_based',potential='vp_economy'))
    with pytest.raises(ValueError,match='configuration disagree'):
        model_reward_config(model,{'reward':legacy})
    model.gamma = 1.
    with pytest.raises(ValueError,match='gamma disagree'): model_reward_config(model)


def load_cli(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name+'_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_cli_selects_economy_without_changing_ppo(monkeypatch,tmp_path):
    module = load_cli('train_ppo')
    captured = {}
    def capture(output,**kwargs):
        captured.update(kwargs)
        return Path(output)/'model.zip'
    monkeypatch.setattr(module,'train_ppo',capture)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys,'argv',['train_ppo.py','--mode','potential_based','--potential','vp_economy',
                                  '--gamma','.99','--output',str(tmp_path/'run')])
    module.main()
    assert captured['reward_config']==RewardConfig(mode='potential_based',potential='vp_economy')
    assert captured['ppo_kwargs']['gamma']==.99
    assert captured['ppo_kwargs']['policy_kwargs']==dict(net_arch=[64,64],activation_fn='tanh')


@pytest.mark.parametrize('source',['terminal','training'])
def test_evaluation_recovers_native_rewards_and_labels_them_without_run_yaml(monkeypatch,tmp_path,source):
    module = load_cli('evaluate_ppo')
    config = RewardConfig(mode='potential_based',potential='vp_economy')
    spaces = DominionEnv()
    model = SimpleNamespace(gamma=.99,num_timesteps=16,seed=47,policy=SimpleNamespace(activation_fn=torch.nn.Tanh),
                            observation_space=spaces.observation_space,action_space=spaces.action_space,
                            training_reward_config=asdict(config))
    spaces.close()
    monkeypatch.setattr(module.MaskablePPO,'load',lambda *args,**kwargs:model)
    captured = []
    def episode(policy,env,seed,**kwargs):
        captured.append(env.reward_config)
        return dict(outcome='win',evaluation_seed=seed)
    monkeypatch.setattr(module,'play_episode',episode)
    archive,output = tmp_path/'moved.zip',tmp_path/'games.csv'
    archive.write_bytes(b'native archive sentinel')
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys,'argv',['evaluate_ppo.py','--model',str(archive),'--games','1','--opponent','random',
                                  '--agent-player','1','--reward-source',source,'--output',str(output)])
    module.main()
    assert captured==[config if source=='training' else RewardConfig()]
    snapshot = yaml.safe_load(output.with_suffix('.yaml').read_text())
    assert snapshot['models'][0]['training_reward_config']==asdict(config)
    assert snapshot['models'][0]['reward_design']=='vp_economy'
    assert snapshot['models'][0]['evaluation_reward_config']==asdict(captured[0])
