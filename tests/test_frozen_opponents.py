"""Frozen checkpoint contracts, private perspectives and episode-fixed inference."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
import torch
import yaml

from dominion_rl.agents import frozen_policy as frozen
from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.observations import FEATURE_INDEX, encode_observation
from dominion_rl.env.preprocessing import preprocessing_config


@pytest.fixture
def archive(tmp_path):
    env = DominionEnv(GameConfig(version='V1'))
    reward = RewardConfig(gamma=1.,terminal_objective='vp_margin')
    model = MaskablePPO('MlpPolicy',env,seed=54,gamma=1.,n_steps=2,batch_size=2,
        policy_kwargs=dict(net_arch=[64,64],activation_fn=torch.nn.Tanh),device='cpu')
    with torch.no_grad():
        model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
        model.policy.action_net.bias[5] = 1000  # Illegal Province must lose to masked End.
        model.policy.action_net.bias[8] = 500
    model.game_contract = frozen.v1_contract();model.observation_preprocessing = preprocessing_config('raw','V1')
    model.training_reward_config = asdict(reward);model.save(tmp_path/'model.zip');env.close()
    metadata = dict(seed=54,training_gamma=1.,terminal_objective='vp_margin',actual_timesteps=0,game=asdict(GameConfig(version='V1')),
        game_contract=frozen.v1_contract(),features=frozen.v1_contract()['features'],actions=frozen.v1_contract()['actions'],
        observation_preprocessing=preprocessing_config('raw','V1'),observation_scaling='raw',activation='tanh',reward=asdict(reward),
        ppo=dict(gamma=1.,policy_kwargs=dict(net_arch=[64,64],activation_fn='tanh')))
    (tmp_path/'run.yaml').write_text(yaml.safe_dump(metadata))
    data = dict(format_version=1,root='.',policies=[dict(id='test_policy',role='training',path='model.zip',sha256=frozen.file_hash(tmp_path/'model.zip'),
        training_metadata='run.yaml',training_metadata_sha256=frozen.file_hash(tmp_path/'run.yaml'),training_seed=54,checkpoint_step=0)],
        training_mixture=[dict(opponent='engine-money',probability=.5),dict(opponent='test_policy',probability=.5)])
    path = tmp_path/'pool.yaml';path.write_text(yaml.safe_dump(data,sort_keys=False))
    return path,data


def test_valid_checkpoint_loads_once_freezes_parameters_and_is_deterministic(archive):
    path,_ = archive;pool,models = frozen.load_frozen_policies(path)
    model = models['test_policy'];env = DominionEnv(GameConfig(version='V1'))
    try:
        env.reset(seed=4)
        assert not model.policy.training and all(not p.requires_grad and p.grad is None for p in model.policy.parameters())
        before = {k:v.clone() for k,v in model.policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        assert frozen.frozen_action(model,env.game) == frozen.frozen_action(model,env.game) == env.end_buying
        assert torch.equal(rng,torch.get_rng_state())
        learner = MaskablePPO('MlpPolicy',env,seed=57,n_steps=2,batch_size=2,
            policy_kwargs=dict(net_arch=[64,64],activation_fn=torch.nn.Tanh),device='cpu')
        own = [p.detach().clone() for p in learner.policy.parameters()]
        learner.policy.optimizer.zero_grad();sum(p.sum() for p in learner.policy.parameters()).backward();learner.policy.optimizer.step()
        assert any(not torch.equal(a,b) for a,b in zip(own,learner.policy.parameters()))
        assert all(torch.equal(before[k],v) for k,v in model.policy.state_dict().items())
    finally:env.close()


@pytest.mark.parametrize('key',['sha256','training_metadata_sha256'])
@pytest.mark.parametrize('damage',['missing','changed'])
def test_missing_or_altered_hash_rejected_before_loading(archive,key,damage,monkeypatch):
    path,data = archive
    if damage == 'missing':data['policies'][0].pop(key)
    else:data['policies'][0][key] = '0'*64
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(MaskablePPO,'load',lambda *a,**k:pytest.fail('Must reject before network load'))
    with pytest.raises(ValueError,match='hash|sha256'):frozen.load_frozen_policies(path)


@pytest.mark.parametrize('corruption',['V0','features','actions','preprocessing','architecture','metadata'])
def test_incompatible_native_or_saved_contract_rejected(archive,corruption):
    path,data = archive;zip_path = path.parent/'model.zip'
    with zipfile.ZipFile(zip_path) as stream:contents = {n:stream.read(n) for n in stream.namelist()}
    native = json.loads(contents['data'])
    if corruption == 'V0':native['game_contract']['version'] = 'V0'
    elif corruption == 'features':native['game_contract']['features'][:2] = reversed(native['game_contract']['features'][:2])
    elif corruption == 'actions':native['game_contract']['actions'][6:8] = reversed(native['game_contract']['actions'][6:8])
    elif corruption == 'preprocessing':native['observation_preprocessing']['mode'] = 'fixed'
    elif corruption == 'architecture':native['policy_kwargs']['net_arch'] = [32,32]
    else:
        metadata = yaml.safe_load((path.parent/'run.yaml').read_text());metadata['seed'] = 55
        (path.parent/'run.yaml').write_text(yaml.safe_dump(metadata));data['policies'][0]['training_metadata_sha256'] = frozen.file_hash(path.parent/'run.yaml')
    contents['data'] = json.dumps(native).encode()
    with zipfile.ZipFile(zip_path,'w') as stream:
        for name,value in contents.items():stream.writestr(name,value)
    data['policies'][0]['sha256'] = frozen.file_hash(zip_path);path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):frozen.load_frozen_policies(path)


def test_held_out_role_cannot_enter_training_mixture(archive):
    path,data = archive;data['policies'][0]['role'] = 'evaluation_only';path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='evaluation-only'):frozen.verify_manifest(path)


@pytest.fixture
def fake_pool(monkeypatch,tmp_path):
    path = tmp_path/'pool.yaml';path.write_text('fixture: true\n');calls = [];observations = []
    def predict(obs,deterministic,action_masks):
        assert deterministic and torch.is_inference_mode_enabled()
        observations.append((obs.copy(),action_masks.copy()))
        # Buy Silver repeatedly if affordable; then End buying.
        return np.asarray(1 if action_masks[1] else 8),None
    model = SimpleNamespace(predict=predict)
    pool = dict(policies=[dict(id='test_policy',role='training')],training_mixture=[dict(opponent='engine-money',probability=.5),
        dict(opponent='test_policy',probability=.5)])
    def load(path,identifiers=None,training=False):
        calls.append((identifiers,training));return pool,{'test_policy':model}
    monkeypatch.setattr(frozen,'load_frozen_policies',load)
    return path,calls,observations,model


def arrange_opponent(env):
    game = env.game;game.state.current_player = 1-env.agent_player
    player = game.state.players[game.state.current_player]
    stock = Counter(player.deck+player.hand+player.discard+player.in_play)
    hand,draws = ['village','council_room'],['gold','silver','copper','estate','estate']
    for card in hand+draws:
        if stock[card]:stock[card] -= 1
        else:game.state.supply[card] -= 1
    player.hand,player.deck,player.discard,player.in_play = hand,list(reversed(draws)),list(stock.elements()),[]
    player.actions,player.buys,player.coins = 1,1,0;game._start_turn()
    return player


@pytest.mark.parametrize('seat',[0,1])
def test_own_perspective_masks_automatic_actions_multiple_buys_and_end(fake_pool,seat):
    path,calls,observations,model = fake_pool
    env = DominionEnv(GameConfig(version='V1'),opponent='frozen',opponent_pool=path,frozen_opponent_id='test_policy',agent_player=seat)
    try:
        env.reset(seed=7);observations.clear();player = arrange_opponent(env)
        expected = encode_observation(env.game.state,1-seat);before = Counter(player.deck+player.hand+player.discard+player.in_play)['silver'];turns = player.turns_taken
        env._advance_opponent()
        assert len(observations) == 3
        np.testing.assert_array_equal(observations[0][0],expected)
        assert observations[0][1][1] and not observations[0][1][5] and observations[0][1][8]
        np.testing.assert_array_equal(observations[-1][1],[False]*8+[True])
        assert [int(o[FEATURE_INDEX['buys']]) for o,m in observations] == [2,1,0]
        assert player.turns_taken == turns+1 and env.game.state.current_player == seat
        assert len(calls) == 1  # No model load per decision, turn or episode.
        assert sum(c == 'silver' for zone in (player.deck,player.hand,player.discard,player.in_play) for c in zone) == before+2
    finally:env.close()


def test_end_without_buy_and_invalid_policy_action(fake_pool):
    path,calls,observations,model = fake_pool
    env = DominionEnv(GameConfig(version='V1'),opponent='frozen',opponent_pool=path,frozen_opponent_id='test_policy')
    try:
        env.reset(seed=5);player = arrange_opponent(env);owned = Counter(player.deck+player.hand+player.discard+player.in_play)
        model.predict = lambda obs,**kwargs:(np.asarray(8),None)
        env._advance_opponent()
        assert Counter(player.deck+player.hand+player.discard+player.in_play) == owned
        assert env.game.state.current_player == 0
        arrange_opponent(env);model.predict = lambda obs,**kwargs:(np.asarray(5),None)
        with pytest.raises(ValueError,match='illegal'):env._advance_opponent()
    finally:env.close()


def test_episode_selection_reproducible_fixed_and_no_reload(fake_pool):
    path,calls,observations,model = fake_pool
    a = DominionEnv(GameConfig(version='V1',max_turns=8),opponent='historical',opponent_pool=path)
    b = DominionEnv(GameConfig(version='V1',max_turns=8),opponent='historical',opponent_pool=path)
    try:
        def sequence(env):
            env.reset(seed=57);result = [env.episode_opponent]
            for _ in range(30):env.reset();result.append(env.episode_opponent)
            return result
        assert sequence(a) == sequence(b)
        selected = a.episode_opponent
        for _ in range(3):
            a.step(a.end_buying);assert a.episode_opponent == selected
        assert set(a.opponent_step_counts) == {'engine-money','test_policy'} and sum(a.opponent_step_counts.values()) == 3
        assert sum(a.opponent_reset_counts.values()) == 31
        assert len(calls) == 2
    finally:a.close();b.close()


def test_hidden_zones_orders_and_identity_not_in_observation(fake_pool):
    path,calls,observations,model = fake_pool
    env = DominionEnv(GameConfig(version='V1'),opponent='frozen',opponent_pool=path,frozen_opponent_id='test_policy')
    try:
        env.reset(seed=9);arrange_opponent(env)
        frozen.frozen_action(model,env.game);first = observations[-1][0]
        other = env.game.state.players[env.agent_player]
        cards = other.deck+other.hand+other.discard+other.in_play
        other.deck,other.hand,other.discard,other.in_play = cards[:1],cards[1:3],cards[3:],[]
        other.coins,other.buys = 999,99
        env.game.state.players[1-env.agent_player].deck.reverse()
        frozen.frozen_action(model,env.game)
        np.testing.assert_array_equal(first,observations[-1][0])
        learner = encode_observation(env.game.state,env.agent_player)
        teacher = env.game.state.players[1-env.agent_player]
        cards = teacher.deck+teacher.hand+teacher.discard+teacher.in_play
        teacher.hand,teacher.deck,teacher.discard,teacher.in_play = cards[:1],cards[1:3],cards[3:],[]
        env.episode_opponent = 'another_identity'
        np.testing.assert_array_equal(learner,encode_observation(env.game.state,env.agent_player))
        assert learner.shape == (51,) and len(env.feature_names) == 51
    finally:env.close()


@pytest.mark.parametrize('mode',['random','big-money','mixed','curriculum','engine-money'])
@pytest.mark.parametrize('version',['V0','V1'])
def test_fixed_modes_never_load_frozen_models_and_keep_seeded_behavior(monkeypatch,mode,version):
    monkeypatch.setattr(frozen,'load_frozen_policies',lambda *a,**k:pytest.fail('Fixed modes must not load PPO'))
    if mode == 'engine-money' and version == 'V0':
        with pytest.raises(ValueError):DominionEnv(opponent=mode)
        return
    kwargs = dict(curriculum_total_timesteps=256) if mode == 'curriculum' else {}
    a = DominionEnv(GameConfig(version=version,max_turns=4),opponent=mode,**kwargs)
    b = DominionEnv(GameConfig(version=version,max_turns=4),opponent=mode,**kwargs)
    try:
        x,_ = a.reset(seed=12);y,_ = b.reset(seed=12)
        for _ in range(2):
            np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(a.action_masks(),b.action_masks())
            x,*_ = a.step(a.end_buying);y,*_ = b.step(b.end_buying)
            assert a.game.state == b.game.state
    finally:a.close();b.close()


def test_training_loader_never_instantiates_evaluation_only_policy(archive,monkeypatch):
    path,data = archive
    extra = deepcopy(data['policies'][0]);extra.update(id='held_out',role='evaluation_only',path='held_out.zip')
    (path.parent/'held_out.zip').write_bytes((path.parent/'model.zip').read_bytes())
    data['policies'].append(extra);path.write_text(yaml.safe_dump(data,sort_keys=False))
    calls = [];original = MaskablePPO.load
    def load(filename,**kwargs):calls.append(Path(filename).name);return original(filename,**kwargs)
    monkeypatch.setattr(MaskablePPO,'load',load)
    pool,models = frozen.load_frozen_policies(path,training=True)
    assert list(models) == ['test_policy'] and calls == ['model.zip']
    provenance = frozen.pool_provenance(path,pool,list(models))
    assert [p['id'] for p in provenance['policies']] == ['test_policy']


@pytest.mark.parametrize('filename',['model.zip','run.yaml'])
def test_modified_artifact_bytes_rejected_before_network_load(archive,filename,monkeypatch):
    path,_ = archive;artifact = path.parent/filename
    artifact.write_bytes(artifact.read_bytes()+b'\nchanged')
    monkeypatch.setattr(MaskablePPO,'load',lambda *a,**k:pytest.fail('Corrupt files must never be loaded'))
    with pytest.raises(ValueError,match='hash mismatch'):frozen.load_frozen_policies(path)
