"""Fixed V1 checkpoint validation and deterministic purchase inference.

No manager, optimizer or sampling state. Neural imports occur only when loading
or using a frozen policy, so fixed-heuristic environments keep optional RL deps.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np
import yaml

from dominion_rl.config import RewardConfig
from dominion_rl.env.observations import encode_observation, feature_names
from dominion_rl.env.preprocessing import preprocessing_config
from dominion_rl.game.cards import card_ids_for_version

RESERVED = {'random','big-money','engine-money','festival-money','militia-money','mixed','curriculum','historical','frozen'}


def require(condition: bool, message: str) -> None:
    if not condition:raise ValueError(message)


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def v1_contract() -> dict[str, Any]:
    return dict(version='V1',features=list(feature_names('V1')),actions=[*card_ids_for_version('V1'),'end_buying'])


def verify_manifest(path: str | Path) -> dict[str, Any]:
    """Validate hashes/native JSON and saved metadata without loading networks."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    require(isinstance(data,dict) and data.get('format_version') == 1,'Invalid frozen pool format')
    root = (path.parent/data['root']).resolve();data['root'] = str(root)
    entries = data.get('policies')
    require(isinstance(entries,list) and bool(entries),'Explicit frozen policy list required')
    ids,paths = set(),set()
    for entry in entries:
        require(isinstance(entry,dict),"Each frozen policy needs an explicit metadata mapping")
        identifier = entry.get('id')
        require(isinstance(identifier,str) and bool(re.fullmatch(r'[A-Za-z0-9_-]+',identifier)) and identifier not in RESERVED and identifier not in ids,
                'Duplicate/reserved/invalid frozen opponent identifier')
        require(entry.get('role') in ('training','evaluation_only'),'Each frozen policy needs an explicit role')
        ids.add(identifier)
        for key,hash_key in (('path','sha256'),('training_metadata','training_metadata_sha256')):
            require(isinstance(entry.get(hash_key),str) and bool(re.fullmatch(r'[0-9a-f]{64}',entry[hash_key])),f'Missing/invalid {hash_key}')
            resolved = (root/entry[key]).resolve();entry[key] = str(resolved)
            require(file_hash(resolved) == entry[hash_key],f'Frozen artifact hash mismatch: {resolved}')
        require(entry['path'] not in paths,'Duplicate frozen checkpoint path');paths.add(entry['path'])
        metadata = yaml.safe_load(Path(entry['training_metadata']).read_text())
        with zipfile.ZipFile(entry['path']) as archive:native = json.loads(archive.read('data'))
        contract = v1_contract()
        require(native.get('game_contract') == metadata.get('game_contract') == contract and
            metadata.get('features') == contract['features'] and metadata.get('actions') == contract['actions'] and
            metadata['game']['version'] == 'V1','Frozen policy V1 observation/action contract mismatch')
        require(native.get('observation_preprocessing') == metadata.get('observation_preprocessing') == preprocessing_config('raw','V1') and
            metadata.get('observation_scaling') == 'raw','Frozen preprocessing mismatch; V1 requires raw inputs')
        require(native['policy_kwargs']['net_arch'] == [64,64] and
            native['policy_kwargs']['activation_fn'].endswith(".Tanh'>") and metadata['activation'] == 'tanh' and
            metadata['ppo']['policy_kwargs'] == dict(net_arch=[64,64],activation_fn='tanh'),'Frozen architecture mismatch')
        require(native['seed'] == metadata['seed'] == entry['training_seed'] and
            native['num_timesteps'] == entry['checkpoint_step'] and entry['checkpoint_step'] <= metadata['actual_timesteps'],
            'Frozen seed/checkpoint metadata mismatch')
        reward = RewardConfig(**native['training_reward_config'])
        require(reward == RewardConfig(**metadata['reward']) and native['gamma'] == metadata['training_gamma'] ==
            metadata['ppo']['gamma'] == reward.gamma and metadata.get('terminal_objective',reward.terminal_objective) == reward.terminal_objective,
            'Frozen gamma/reward metadata mismatch')
    mixture = data.get('training_mixture')
    require(isinstance(mixture,list) and bool(mixture),'Explicit per-episode mixture required')
    seen = set();total = 0.
    by_id = {p['id']:p for p in entries}
    for row in mixture:
        name,p = row['opponent'],row['probability']
        require(name not in seen and (name == 'engine-money' or (name in by_id and by_id[name]['role'] == 'training')),
                'Training mixture contains unknown, duplicated or evaluation-only opponent')
        require(type(p) in (int,float) and isfinite(p) and p > 0,'Selection probabilities must be positive finite numbers')
        seen.add(name);total += p
    require(abs(total-1.) < 1e-12,'Selection probabilities must sum to one')
    return data


def pool_provenance(path: str | Path, pool: dict, identifiers: list[str] | None = None) -> dict:
    entries = [p for p in pool['policies'] if identifiers is None or p['id'] in identifiers]
    return dict(path=str(Path(path).resolve()),sha256=file_hash(path),policies=entries,training_mixture=pool['training_mixture'])


def load_frozen_policies(path: str | Path, identifiers: list[str] | None = None, *, training: bool = False) -> tuple[dict,dict]:
    pool = verify_manifest(path)
    entries = {p['id']:p for p in pool['policies']}
    if training:
        require(identifiers is None,'Training IDs come only from the frozen mixture')
        identifiers = [r['opponent'] for r in pool['training_mixture'] if r['opponent'] != 'engine-money']
    elif identifiers is None:identifiers = list(entries)
    require(bool(identifiers) and len(set(identifiers)) == len(identifiers) and all(i in entries for i in identifiers),'Unknown/empty frozen policy selection')
    import torch
    from sb3_contrib import MaskablePPO
    from dominion_rl.ppo import model_game_contract, model_preprocessing, model_reward_config
    models = {}
    for identifier in identifiers:
        entry = entries[identifier];metadata = yaml.safe_load(Path(entry['training_metadata']).read_text())
        model = MaskablePPO.load(entry['path'],device='cpu')
        require(model_game_contract(model,metadata) == v1_contract() and model_preprocessing(model,'raw') == preprocessing_config('raw','V1'),
                'Loaded frozen model contract mismatch')
        require(model.policy.net_arch == [64,64] and model.policy.activation_fn is torch.nn.Tanh,'Loaded frozen policy architecture mismatch')
        require(asdict(model_reward_config(model,metadata)) == asdict(RewardConfig(**metadata['reward'])),'Loaded frozen reward mismatch')
        model.policy.set_training_mode(False)
        for parameter in model.policy.parameters():parameter.requires_grad_(False)
        models[identifier] = model
    return pool,models


def frozen_action(model: Any, game: Any) -> int:
    """Only the acting opponent's own encoded perspective and public legal mask."""
    require(game.state.version == 'V1' and not (game.state.terminated or game.state.truncated),'Frozen inference requires an active V1 Buy phase')
    observation = encode_observation(game.state,game.state.current_player)
    cards = card_ids_for_version('V1')
    legal = game.legal_purchases()
    mask = np.asarray([card in legal for card in cards]+[True],dtype=np.bool_)
    import torch
    with torch.inference_mode():
        action,_ = model.predict(observation,deterministic=True,action_masks=mask)
    selected = int(np.asarray(action).item())
    require(0 <= selected < len(mask) and bool(mask[selected]),'Frozen policy returned an illegal purchase')
    return selected
