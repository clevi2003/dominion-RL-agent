"""Short local V1 PPO integration checks and explicit archive compatibility."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
import torch
import yaml

from dominion_rl.config import GameConfig,RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.preprocessing import preprocessing_config
from dominion_rl.ppo import game_contract, model_game_contract, model_preprocessing, play_episode, train_ppo


@pytest.mark.parametrize('design',['terminal_only','vp','vp_economy'])
def test_v1_smoke_train_checkpoints_reload_masks_and_both_seats(tmp_path,design):
    config=GameConfig(version='V1',max_turns=8)
    reward=RewardConfig(mode='terminal_only' if design=='terminal_only' else 'potential_based',
                        potential='vp_economy' if design=='vp_economy' else 'vp')
    path=train_ppo(tmp_path/'run',game_config=config,reward_config=reward,seed=47,opponent='engine-money',
        total_timesteps=32,checkpoint_freq=16,ppo_kwargs=dict(n_steps=16,batch_size=8,n_epochs=1,
            gamma=.99,policy_kwargs={'net_arch':[16,16]}))
    metadata=yaml.safe_load((path.parent/'run.yaml').read_text())
    assert metadata['game']['version']=='V1' and len(metadata['features'])==51 and len(metadata['actions'])==9
    assert metadata['game_contract']==game_contract('V1')
    assert metadata['by_opponent']['engine-money']['decision_steps']==32
    for archive in (path,*sorted((path.parent/'checkpoints').glob('*.zip'))):
        loaded=MaskablePPO.load(archive,device='cpu')
        assert model_game_contract(loaded,metadata)==game_contract('V1')
        assert model_preprocessing(loaded)==preprocessing_config('raw','V1')
    for seat in (0,1):
        env=DominionEnv(config,agent_player=seat,opponent='engine-money')
        try:
            result=play_episode(loaded,env,20000)
            assert result['game_version']=='V1' and result['terminated']!=result['truncated']
            assert result['purchases_village']>=0 and result['purchases_council_room']>=0
            obs,_=env.reset(seed=42)
            with torch.no_grad():
                loaded.policy.action_net.weight.zero_()
                loaded.policy.action_net.bias.zero_()
                loaded.policy.action_net.bias[5]=1000  # Unaffordable Province.
                loaded.policy.action_net.bias[8]=500
            mask=env.action_masks()
            action,_=loaded.predict(obs,deterministic=True,action_masks=mask)
            assert mask[int(action.item())] and int(action.item())!=5
            with pytest.raises(ValueError,match='schema'): play_episode(loaded,DominionEnv(),42)
        finally: env.close()


def test_legacy_v0_archive_remains_evaluable_but_cannot_be_used_as_v1(tmp_path):
    env=DominionEnv(GameConfig(max_turns=2))
    other=DominionEnv(GameConfig(version='V1',max_turns=2))
    try:
        old=MaskablePPO('MlpPolicy',env,n_steps=2,batch_size=2,seed=42,device='cpu')
        old.save(tmp_path/'legacy.zip')  # No custom game or preprocessing attributes.
        loaded=MaskablePPO.load(tmp_path/'legacy.zip',device='cpu')
        assert model_game_contract(loaded)==game_contract('V0')
        assert play_episode(loaded,env,42)['outcome']=='truncated'
        with pytest.raises(ValueError,match='snapshot'):
            model_game_contract(loaded,{'game':{'version':'V1'}})
        with pytest.raises(ValueError,match='schema'): play_episode(loaded,other,42)
        # Space sizes alone cannot legitimize a V1 archive missing its mapping.
        loaded.observation_space,loaded.action_space=other.observation_space,other.action_space
        with pytest.raises(ValueError,match='legacy'): model_game_contract(loaded)
    finally:
        env.close();other.close()


def test_v1_native_mapping_corruption_is_rejected(tmp_path):
    env=DominionEnv(GameConfig(version='V1',max_turns=2))
    try:
        model=MaskablePPO('MlpPolicy',env,n_steps=2,batch_size=2,seed=42,device='cpu')
        model.game_contract=deepcopy(env.game_contract)
        model.game_contract['actions'][6:8]=reversed(model.game_contract['actions'][6:8])
        with pytest.raises(ValueError,match='mapping'): model_game_contract(model)
    finally: env.close()
