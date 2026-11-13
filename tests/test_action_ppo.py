"""Native Action checkpoints, role isolation, and deterministic baseline pairing."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
pytest.importorskip('sb3_contrib')
import torch
from sb3_contrib import MaskablePPO
from dominion_rl.action_ppo import play_action_episode, validate_action_model
from dominion_rl.agents.action_rich import buying_config
from dominion_rl.env import DominionEnv
from dominion_rl.env.action_env import DominionActionEnv, action_contract, action_preprocessing, action_reward
from dominion_rl.config import GameConfig
from dominion_rl.ppo import model_game_contract, train_ppo


def native_stub():
    env = DominionActionEnv()
    return SimpleNamespace(policy_role='action', game_contract=action_contract(),
        observation_space=env.observation_space, action_space=env.action_space,
        observation_preprocessing=action_preprocessing(), fixed_buying_controller=buying_config(),
        action_policy_settings=dict(net_arch=[64,64], activation='tanh', observation_scaling='raw', gae_lambda=.95),
        num_timesteps=0, gae_lambda=.95, policy_kwargs=dict(net_arch=[64,64], activation_fn=torch.nn.Tanh),
        gamma=1., training_reward_config=asdict(action_reward()), training_seed=72, seed=72)


@pytest.mark.parametrize('field,value', [
    ('policy_role', 'purchase'), ('game_contract', {}), ('observation_preprocessing', {}),
    ('fixed_buying_controller', {}), ('action_policy_settings', {}),
    ('gamma', .99), ('training_reward_config', {}), ('training_seed', None)])
def test_native_role_and_contract_mismatches_fail_closed(field, value):
    model = native_stub(); validate_action_model(model)
    setattr(model, field, value)
    with pytest.raises(ValueError): validate_action_model(model)


def test_role_rejection_and_backward_compatible_purchase_contracts():
    with pytest.raises(ValueError, match='purchasing model'): model_game_contract(native_stub())
    for version in ('V0', 'V1', 'V2', 'V3'):
        env = DominionEnv(GameConfig(version=version))
        legacy = SimpleNamespace(observation_space=env.observation_space, action_space=env.action_space)
        if version != 'V0': legacy.game_contract = env.game_contract
        assert model_game_contract(legacy) == env.game_contract
        with pytest.raises(ValueError, match='Action model'): validate_action_model(legacy)
        env.close()


def test_action_training_saves_checkpoints_metadata_and_masked_load(tmp_path):
    path = train_ppo(tmp_path/'run', policy_role='action', game_config=GameConfig(version='V3', max_turns=4),
        reward_config=action_reward(), seed=72, total_timesteps=16, checkpoint_freq=8, opponent='militia-money',
        ppo_kwargs=dict(gamma=1., gae_lambda=.95, n_steps=8, batch_size=8, n_epochs=1,
                        policy_kwargs=dict(net_arch=[64,64], activation_fn='tanh')))
    metadata = yaml.safe_load((path.parent/'run.yaml').read_text())
    assert metadata['policy_role'] == 'action' and metadata['actual_timesteps'] == 16
    assert metadata['features'] == action_contract()['features'] and metadata['actions'] == action_contract()['actions']
    assert metadata['fixed_buying_controller'] == buying_config()
    assert metadata['action_exposure']['total']['action_decisions'] == 16
    assert metadata['completed_episodes'] > 0 and metadata['by_opponent']['militia-money']['decision_steps'] == 16
    for archive in (path, *sorted((path.parent/'checkpoints').glob('*.zip'))):
        loaded = MaskablePPO.load(archive, device='cpu')
        validate_action_model(loaded, metadata)
        with pytest.raises(ValueError): model_game_contract(loaded)
        for seat in (0, 1):
            env = DominionActionEnv(GameConfig(version='V3', max_turns=4), agent_player=seat)
            try:
                obs, _ = env.reset(seed=38000)
                action, _ = loaded.predict(obs, deterministic=True, action_masks=env.action_masks())
                assert env.action_masks()[int(action)]
                result = play_action_episode(loaded, env, 38000)
                assert result['training_seed'] == 72 and result['checkpoint_step'] == loaded.num_timesteps
                assert result['truncated'] and result['outcome'] == 'truncation'
                assert result['tie_adjusted_score'] is None
            finally: env.close()


@pytest.mark.parametrize('change', ['scaling', 'architecture', 'reward', 'gamma'])
def test_invalid_training_settings_fail_before_output(tmp_path, change):
    kwargs = dict(gamma=1., observation_scaling='raw', policy_kwargs=dict(net_arch=[64,64]))
    reward = action_reward()
    if change == 'scaling': kwargs['observation_scaling'] = 'fixed'
    elif change == 'architecture': kwargs['policy_kwargs']['net_arch'] = [16,16]
    elif change == 'reward': reward = type(reward)(gamma=1., terminal_objective='win_loss')
    else: kwargs['gamma'] = .99
    with pytest.raises(ValueError):
        train_ppo(tmp_path/'absent', policy_role='action', game_config=GameConfig(version='V3'),
                  reward_config=reward, ppo_kwargs=kwargs)
    assert not (tmp_path/'absent').exists()


def test_baseline_and_scripted_equivalent_action_policy_share_exact_decisions():
    from dominion_rl.env.action_env import ACTION_CARDS, END_ACTION
    model = native_stub(); model.num_timesteps = 0
    def predict(obs, **kw):
        card = env.game.automatic_action_choice()
        return np.asarray(END_ACTION if card is None else ACTION_CARDS.index(card)), None
    model.predict = predict
    env = DominionActionEnv(GameConfig(version='V3', max_turns=80), record_decisions=True)
    try:
        baseline = play_action_episode(None, env, 39000)
        decisions = deepcopy(env.decisions)
        learned = play_action_episode(model, env, 39000)
        assert env.decisions == decisions
        for key in baseline:
            if key not in ('condition', 'training_seed', 'checkpoint_step'):
                assert baseline[key] == learned[key]
    finally: env.close()
