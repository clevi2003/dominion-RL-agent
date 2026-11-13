"""Small MaskablePPO training and masked game-evaluation helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.observations import FEATURE_INDEX, V1_FEATURE_INDEX, V2_FEATURE_INDEX, V3_FEATURE_INDEX, feature_names
from dominion_rl.game.cards import card_ids_for_version
from dominion_rl.env.preprocessing import PolicyObservation, preprocessing_config


# Evaluation-only counters from the agent's public in-play cards and Buy decisions.
V1_USAGE_FIELDS = ("observed_agent_turns", "village_plays", "council_room_plays",
    "turns_with_village_plays", "turns_with_council_room_plays", "council_room_extra_buys_granted",
    "turns_with_legal_extra_purchase_opportunity", "turns_with_non_copper_extra_purchase_opportunity", "turns_with_multiple_purchases")

V2_USAGE_FIELDS = (*V1_USAGE_FIELDS, "festival_plays", "turns_with_festival_plays",
                   "festival_extra_buys_granted", "festival_coins_generated",
    "turns_with_festival_and_council_room_plays", "purchases_after_first",
    "non_copper_purchases_after_first", *(
        f"{prefix}_{card}" for card in ("village", "council_room", "festival")
        for prefix in ("owned_card_turns", "turns_owning", "turns_owning_without_play")))

V3_USAGE_FIELDS = (*V2_USAGE_FIELDS, "militia_plays", "moat_plays",
    "turns_with_militia_plays", "turns_with_moat_plays", "militia_coins_generated")


def militia_statistics(game, agent_player: int) -> dict[str, int]:
    """Public resolution counts only; private discarded identities are absent."""
    result = {}
    for who, seat in (("agent", agent_player), ("opponent", 1 - agent_player)):
        attacked = [e for e in game.militia_resolutions if e["attacker"] == seat]
        defended = [e for e in game.militia_resolutions if e["defender"] == seat]
        result[f"militia_attacks_by_{who}"] = len(attacked)
        result[f"moat_reactions_by_{who}"] = sum(int(e["moat_revealed"]) for e in defended)
        result[f"militia_discards_by_{who}"] = sum(int(e["discard_count"]) for e in defended)
    return result


def game_contract(version: str = "V0") -> dict[str, Any]:
    """Exact schema/action meanings, independent of configured pile ordering."""
    return dict(version=version, features=list(feature_names(version)),
                actions=[*card_ids_for_version(version), "end_buying"])


def model_game_contract(model: MaskablePPO, training: dict[str, Any] | None = None) -> dict[str, Any]:
    """Old 41/7 archives explicitly mean V0; newer versions need saved contracts."""
    if getattr(model, "policy_role", "purchase") != "purchase":
        raise ValueError("A purchasing model is required; Action policies have a separate contract")
    if training is not None and training.get("policy_role", "purchase") != "purchase":
        raise ValueError("Training snapshot has the wrong policy role")
    saved = getattr(model, "game_contract", None)
    if saved is None:
        if model.observation_space.shape != (41,) or getattr(model.action_space, "n", None) != 7:
            raise ValueError("A legacy model without game schema metadata must use V0 spaces (41/7)")
        saved = game_contract("V0")
    if not isinstance(saved, dict) or saved != game_contract(saved.get("version")):
        raise ValueError("Incompatible model game schema/action mapping metadata")
    if (model.observation_space.shape != (len(saved["features"]),)
            or getattr(model.action_space, "n", None) != len(saved["actions"])):
        raise ValueError("Model spaces disagree with its saved game schema")
    if training is not None:
        if (training.get("game_contract", saved) != saved
                or training.get("features", saved["features"]) != saved["features"]
                or training.get("actions", saved["actions"]) != saved["actions"]
                or training.get("game", {}).get("version", saved["version"]) != saved["version"]):
            raise ValueError("Training snapshot and model game schema/action mapping disagree")
    return saved


def model_preprocessing(model: MaskablePPO, expected_mode: str | None = None) -> dict[str, Any]:
    """Recover exact input units without reinterpreting historical V0 features."""
    saved = getattr(model, "observation_preprocessing", None)
    contract = getattr(model, "game_contract", None)
    if contract is not None:
        contract = model_game_contract(model)
    if saved is None:
        saved = preprocessing_config("raw", contract["version"] if contract is not None else "V0")
    if not isinstance(saved, dict):
        raise ValueError("Incompatible model observation preprocessing metadata")
    version = next((v for v in ("V0", "V1", "V2", "V3")
                    if saved.get("features") == list(feature_names(v))), None)
    if version is None:
        raise ValueError("Incompatible model observation preprocessing metadata")
    if saved != preprocessing_config(saved.get("mode"), version):
        raise ValueError("Incompatible model observation preprocessing metadata")
    if contract is not None and saved["features"] != model_game_contract(model)["features"]:
        raise ValueError("Model preprocessing and game schema disagree")
    if expected_mode is not None and saved["mode"] != expected_mode:
        raise ValueError(f"Model requires {saved['mode']} observations, not {expected_mode}")
    return saved


def model_training_gamma(model: MaskablePPO, training: dict[str, Any] | None = None) -> float:
    """Read native SB3 gamma (also present in legacy archives), never override it.

    The optional adjacent training snapshot is a consistency check, not a fallback
    for missing archive data. SB3 already saves gamma in every model/checkpoint.
    """
    gamma = getattr(model, "gamma", None)
    if type(gamma) not in (int, float) or not 0 <= gamma <= 1:
        raise ValueError("Model must contain a finite training gamma between 0 and 1")
    if training is not None:
        recorded = [training.get("training_gamma"), training.get("ppo", {}).get("gamma"),
                    training.get("reward", {}).get("gamma")]
        if any(value is not None and value != gamma for value in recorded):
            raise ValueError("Training snapshot and model gamma disagree")
    return float(gamma)


def model_reward_config(
    model: MaskablePPO, training: dict[str, Any] | None = None,
) -> RewardConfig | None:
    """Recover native rewards; normalize legacy YAML to VP-only defaults.

    New archives carry the full contract even after relocation. Legacy archives
    need their run.yaml; absent both, the original reward is explicitly unknown.
    Evaluation may still use common terminal outcomes, never guess economy mode.
    """
    native = getattr(model, "training_reward_config", None)
    snapshot = training.get("reward") if training is not None else None
    recovered = []
    for raw in (native, snapshot):
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("Reward configuration metadata must be a mapping")
            recovered.append(RewardConfig(**raw))
    if not recovered:
        return None
    if any(config != recovered[0] for config in recovered[1:]):
        raise ValueError("Training snapshot and model reward configuration disagree")
    if training is not None and training.get("terminal_objective", recovered[0].terminal_objective) != recovered[0].terminal_objective:
        raise ValueError("Training snapshot and model terminal objective disagree")
    if recovered[0].gamma != model_training_gamma(model, training):
        raise ValueError("Model and reward gamma disagree")
    return recovered[0]


def train_ppo(
    output_dir: str | Path,
    *,
    game_config: GameConfig,
    reward_config: RewardConfig,
    ppo_kwargs: dict[str, Any],
    seed: int = 0,
    total_timesteps: int = 100_000,
    opponent: str = "big-money",
    checkpoint_freq: int = 10_000,
    curriculum: list[dict[str, float]] | None = None,
    opponent_pool: str | Path | None = None,
    policy_role: str = "purchase",
) -> Path:
    """Train on CPU, saving the final model, periodic checkpoints and CSV logs.

    SB3 wraps Monitor in DummyVecEnv. This preserves terminal_observation and
    TimeLimit.truncated, which MaskablePPO uses to bootstrap only truncations.
    No extra TimeLimit wrapper or manual episode reset is needed.
    """
    if policy_role not in ("purchase", "action"):
        raise ValueError("policy_role must be purchase or action")
    if policy_role == "action":
        from dominion_rl.env.action_env import (DominionActionEnv, EXPOSURE_FIELDS, OPPONENTS,
                                                    action_preprocessing, action_reward)
        if opponent not in OPPONENTS:
            raise ValueError("Action learning requires a fixed deterministic V3 opponent")
        if curriculum is not None or opponent_pool is not None:
            raise ValueError("Action learning requires fixed opponents, no curriculum/pool")
        if game_config.version != "V3" or reward_config != action_reward():
            raise ValueError("Action learning requires V3 and terminal-only VP margin gamma=1")
    wall_start = perf_counter()
    if seed < 0 or total_timesteps <= 0 or checkpoint_freq < 0:
        raise ValueError("seed/checkpoint_freq must be nonnegative; timesteps must be positive")
    kwargs = dict(ppo_kwargs)
    gamma = kwargs.get("gamma", 0.99)
    if gamma != reward_config.gamma:
        raise ValueError("PPO gamma must match reward gamma")
    if type(gamma) not in (int, float) or gamma not in (0.99, 1.0):
        raise ValueError("Training gamma must be 0.99 or 1.0")
    kwargs["gamma"] = float(gamma)
    scaling = kwargs.pop("observation_scaling", "raw")
    if policy_role == "action" and scaling != "raw":
        raise ValueError("Action policies require raw observations")
    preprocessing = action_preprocessing() if policy_role == "action" else preprocessing_config(scaling, game_config.version)
    policy_kwargs = dict(kwargs.get("policy_kwargs", {}))
    policy_kwargs.setdefault("net_arch", [64, 64])
    activation = policy_kwargs.get("activation_fn", "tanh")
    if activation not in ("tanh", "relu"):
        raise ValueError("policy_kwargs.activation_fn must be 'tanh' or 'relu'")
    if policy_role == "action" and (activation != "tanh" or policy_kwargs["net_arch"] != [64, 64]
                                     or kwargs.get("gae_lambda", 0.95) != 0.95):
        raise ValueError("Action study fixes Tanh, [64,64] and gae_lambda=0.95")
    # Preserve YAML names in the snapshot; pass actual PyTorch classes to SB3.
    policy_kwargs["activation_fn"] = activation
    snapshot_kwargs = {**kwargs, "policy_kwargs": policy_kwargs,
                       "observation_scaling": preprocessing["mode"]}
    kwargs["policy_kwargs"] = {
        **policy_kwargs, "activation_fn": torch.nn.Tanh if activation == "tanh" else torch.nn.ReLU,
    }
    output = Path(output_dir)
    # A distinct directory per run prevents accidental checkpoint/log replacement.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    raw_env = DominionActionEnv(game_config, reward_config, opponent=opponent) if policy_role == "action" else DominionEnv(game_config, reward_config, opponent=opponent,
        curriculum_total_timesteps=total_timesteps if opponent == "curriculum" else None,
        curriculum=curriculum,opponent_pool=opponent_pool)
    curriculum_fields = ("curriculum_stage", "episode_start_step", "curriculum_random_probability") if opponent == "curriculum" else ()
    action_fields = EXPOSURE_FIELDS if policy_role == "action" else ()
    env = Monitor(
        raw_env if policy_role == "action" else PolicyObservation(raw_env, mode=preprocessing["mode"]),
        filename=str(output / "episodes.monitor.csv"),
        info_keywords=("winner", "end_reason", "scores", "turns_taken", "episode_opponent") + curriculum_fields + action_fields,
    )
    logger = configure(str(output), ["stdout", "csv"])
    try:
        model = MaskablePPO("MlpPolicy", env, seed=seed, device="cpu", verbose=1, **kwargs)
        model.set_logger(logger)
        # SB3 saves this ordinary model attribute in every .zip, including periodic
        # checkpoints, so preprocessing survives moving a model without run.yaml.
        model.observation_preprocessing = preprocessing
        # Ordinary SB3 data attribute: retained by step-zero, periodic and final zips.
        model.training_reward_config = asdict(reward_config)
        model.game_contract = raw_env.game_contract
        model.policy_role = policy_role
        model.training_seed = seed
        if policy_role == "action":
            model.fixed_buying_controller = raw_env.fixed_buying_controller
            model.action_policy_settings = dict(net_arch=policy_kwargs["net_arch"], activation=activation,
                                                observation_scaling=scaling, gae_lambda=0.95)
        metadata = {
            "seed": seed,
            "policy_role": policy_role,
            "training_gamma": float(gamma),
            "terminal_objective": reward_config.terminal_objective,
            "requested_timesteps": total_timesteps,
            "checkpoint_freq": checkpoint_freq,
            "opponent": opponent,
            "opponent_distribution": raw_env.historical_probabilities if opponent == "historical" else {name: 0.5 if opponent == "mixed" else float(name == opponent)
                                      for name in raw_env.opponent_step_counts} if opponent != "curriculum" else None,
            "agent_player": 0,
            "device": "cpu",
            "torch_threads": 1,
            "game": asdict(game_config),
            "reward": asdict(reward_config),
            "reward_design": reward_config.mode if reward_config.mode == "terminal_only" else reward_config.potential,
            "ppo": snapshot_kwargs,
            "activation": activation,
            "observation_scaling": preprocessing["mode"],
            "observation_preprocessing": preprocessing,
            "actions": list(raw_env.action_names),
            "features": list(raw_env.feature_names),
            "game_contract": raw_env.game_contract,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in (
                "sb3-contrib", "stable-baselines3", "torch", "gymnasium", "numpy", "PyYAML",
            )},
        }
        if policy_role == "action":
            metadata["fixed_buying_controller"] = raw_env.fixed_buying_controller
            metadata["decision_unit"] = "Action decisions including forced End"
        if opponent == "historical":
            metadata["frozen_opponent_pool"] = raw_env.frozen_pool_metadata
        if raw_env.curriculum is not None:
            metadata["curriculum"] = raw_env.curriculum
            metadata["curriculum_total_timesteps"] = total_timesteps
        config_path = output / "run.yaml"
        config_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        callback = CheckpointCallback(
            save_freq=checkpoint_freq, save_path=str(output / "checkpoints"), name_prefix="ppo",
        ) if checkpoint_freq else None
        # Retain the untrained policy for a true step-zero evaluation baseline.
        (output / "checkpoints").mkdir(exist_ok=True)
        model.save(output / "checkpoints" / "ppo_0_steps.zip")
        training_start = perf_counter()
        model.learn(total_timesteps=total_timesteps, callback=callback)
        training_seconds = perf_counter() - training_start
        model_path = output / "model.zip"
        model.save(model_path)
        # PPO completes whole rollouts, so this may exceed requested_timesteps.
        with (output / "episodes.monitor.csv").open(encoding="utf-8") as stream:
            next(stream)  # SB3 JSON comment preceding the CSV header.
            episodes = list(csv.DictReader(stream))
        truncated_episodes = sum(row["end_reason"] == "turn_limit" for row in episodes)
        completed_steps = sum(int(row["l"]) for row in episodes)
        partial_steps = model.num_timesteps - completed_steps
        started_episodes = len(episodes) + int(partial_steps > 0)
        by_opponent = {}
        for name, decision_steps in env.unwrapped.opponent_step_counts.items():
            rows = [row for row in episodes if row["episode_opponent"] == name]
            started = len(rows) + int(partial_steps > 0 and env.unwrapped.episode_opponent == name)
            by_opponent[name] = {
                "completed_episodes": len(rows), "started_episodes": started,
                "wins": sum(row["winner"] == "0" for row in rows),
                "losses": sum(row["winner"] == "1" for row in rows),
                "ties": sum(row["winner"] == "" and row["end_reason"] != "turn_limit" for row in rows),
                "truncations": sum(row["end_reason"] == "turn_limit" for row in rows),
                "decision_steps": decision_steps,
                "completed_episode_decision_steps": sum(int(row["l"]) for row in rows),
                "episode_fraction": started / started_episodes if started_episodes else None,
                "step_fraction": decision_steps / model.num_timesteps,
            }
        if opponent == "historical":
            metadata["opponent_reset_selections"] = dict(raw_env.opponent_reset_counts)
            metadata["partial_episode"] = dict(episode_opponent=raw_env.episode_opponent,decision_steps=partial_steps) if partial_steps else None
            resets = sum(raw_env.opponent_reset_counts.values())
            for name,count in raw_env.opponent_reset_counts.items():
                by_opponent[name]["reset_selections"] = count
                by_opponent[name]["reset_selection_fraction"] = count/resets if resets else None
        if raw_env.curriculum is not None:
            metadata["partial_episode"] = dict(
                episode_start_step=completed_steps, episode_opponent=raw_env.episode_opponent,
                curriculum_stage=raw_env.curriculum_stage_index, decision_steps=partial_steps,
            ) if partial_steps else None
            metadata["by_curriculum_stage"] = []
            for index, stage in enumerate(raw_env.curriculum):
                stage_rows = [row for row in episodes if int(row["curriculum_stage"]) == index]
                exposure = {}
                for name, count in raw_env.curriculum_step_counts[index].items():
                    rows = [row for row in stage_rows if row["episode_opponent"] == name]
                    partial = int(partial_steps > 0 and raw_env.curriculum_stage_index == index
                                  and raw_env.episode_opponent == name)
                    exposure[name] = dict(completed_episodes=len(rows), started_episodes=len(rows)+partial,
                        decision_steps=count, completed_episode_decision_steps=sum(int(row["l"]) for row in rows))
                metadata["by_curriculum_stage"].append(dict(stage=index, **stage, by_opponent=exposure))
        if policy_role == "action":
            completed_exposure = {key:sum(int(row[key]) for row in episodes) for key in EXPOSURE_FIELDS}
            partial_exposure = {key:raw_env.metrics[key] if partial_steps else 0 for key in EXPOSURE_FIELDS}
            total_exposure = {key:completed_exposure[key]+partial_exposure[key] for key in EXPOSURE_FIELDS}
            metadata["action_exposure"] = dict(completed_episodes=completed_exposure,
                partial_episode=partial_exposure, total=total_exposure,
                multi_type_fraction=total_exposure["multiple_type_decisions"]/model.num_timesteps,
                forced_end_fraction=total_exposure["forced_end_decisions"]/model.num_timesteps)
        metadata.update(
            by_opponent=by_opponent,
            started_episodes=started_episodes,
            partial_episode_steps=partial_steps,
            actual_timesteps=model.num_timesteps,
            training_seconds=training_seconds,
            wall_seconds=perf_counter() - wall_start,
            steps_per_second=model.num_timesteps / training_seconds,
            completed_episodes=len(episodes),
            truncated_episodes=truncated_episodes,
            truncation_rate=truncated_episodes / len(episodes) if episodes else None,
        )
        config_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        return model_path
    finally:
        env.close()
        logger.close()


def play_episode(
    model: MaskablePPO, env: DominionEnv, seed: int, *, deterministic: bool = True,
    expected_scaling: str | None = None, return_gamma: float | None = None,
) -> dict[str, Any]:
    """Evaluate a complete masked game without updating the model.

    The supplied environment determines the opponent, reward and turn limit.
    A truncated game has no winner and is never counted as a tie or loss.
    Discount reported returns with the model's training gamma unless explicitly
    overridden. Scripted policies without gamma retain the environment discount.
    Shaped evaluation rewards require the same shaping and return discounts.
    """
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not isinstance(env, DominionEnv):
        raise ValueError("Pass a raw DominionEnv; play_episode applies model preprocessing automatically")
    training_gamma = model_training_gamma(model) if hasattr(model, "gamma") else None
    if return_gamma is None:
        return_gamma = training_gamma if training_gamma is not None else env.reward_config.gamma
    if type(return_gamma) not in (int, float) or not 0 <= return_gamma <= 1:
        raise ValueError("return_gamma must be a finite number between 0 and 1")
    if env.reward_config.mode == "potential_based" and return_gamma != env.reward_config.gamma:
        raise ValueError("Evaluation return gamma must match shaping gamma for shaped rewards")
    preprocessing = model_preprocessing(model, expected_scaling)
    divisors = np.asarray(preprocessing["divisors"], dtype=np.float32)
    contract = model_game_contract(model)
    if contract != env.game_contract:
        raise ValueError(f"Incompatible game schema: model {contract['version']}, environment {env.game_config.version}")
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise ValueError("Model and environment observation/action spaces must match")
    card_ids, end_buying = env.card_ids, env.end_buying
    # Also seed policy sampling when stochastic evaluation is requested.
    model.set_random_seed(seed)
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    steps = 0
    discounted_episode_return = 0.0
    purchase_decisions = 0
    province_opportunities = 0
    province_declines = 0
    copper_despite_silver_or_gold = 0
    copper_despite_non_copper = 0
    purchases = {card: 0 for card in card_ids}
    first_turn = {card: None for card in ("province", "duchy", "estate", "silver", "gold")}
    v1 = env.game_config.version in ("V1", "V2", "V3")
    v2 = env.game_config.version in ("V2", "V3")
    v3 = env.game_config.version == "V3"
    if v1:
        first_turn.update(village=None, council_room=None)
    if v2:
        first_turn["festival"] = None
    if v3:
        first_turn.update(militia=None, moat=None)
    usage = {name: 0 for name in (V3_USAGE_FIELDS if v3 else V2_USAGE_FIELDS if v2 else V1_USAGE_FIELDS)}
    last_turn, purchases_this_turn = None, 0
    extra_seen = non_copper_extra_seen = False
    early_victory = {card: 0 for card in ("estate", "duchy", "province")}
    bought_this_turn = False
    no_purchase_turns = 0
    passes_with_affordable_non_copper = 0
    while True:
        mask = env.action_masks()
        if v1:
            own_turn = int(observation[FEATURE_INDEX["agent_turns_taken"]]) + 1
            if own_turn != last_turn:
                last_turn, purchases_this_turn = own_turn, 0
                extra_seen = non_copper_extra_seen = False
                village = int(observation[V1_FEATURE_INDEX["in_play_village"]])
                rooms = int(observation[V1_FEATURE_INDEX["in_play_council_room"]])
                usage["observed_agent_turns"] += 1
                usage["village_plays"] += village
                usage["council_room_plays"] += rooms
                usage["turns_with_village_plays"] += int(village > 0)
                usage["turns_with_council_room_plays"] += int(rooms > 0)
                usage["council_room_extra_buys_granted"] += rooms
                if v2:
                    festivals = int(observation[V2_FEATURE_INDEX["in_play_festival"]])
                    usage["festival_plays"] += festivals
                    usage["turns_with_festival_plays"] += int(festivals > 0)
                    usage["festival_extra_buys_granted"] += festivals
                    usage["festival_coins_generated"] += 2 * festivals
                    usage["turns_with_festival_and_council_room_plays"] += int(festivals > 0 and rooms > 0)
                    for card, plays in (("village", village), ("council_room", rooms), ("festival", festivals)):
                        owned = int(observation[V2_FEATURE_INDEX[f"owned_{card}"]])
                        usage[f"owned_card_turns_{card}"] += owned
                        usage[f"turns_owning_{card}"] += int(owned > 0)
                        usage[f"turns_owning_without_play_{card}"] += int(owned > 0 and plays == 0)
                if v3:
                    for card in ("militia", "moat"):
                        plays = int(observation[V3_FEATURE_INDEX[f"in_play_{card}"]])
                        usage[f"{card}_plays"] += plays
                        usage[f"turns_with_{card}_plays"] += int(plays > 0)
                    usage["militia_coins_generated"] += 2 * int(observation[V3_FEATURE_INDEX["in_play_militia"]])
            # A remaining Buy with a legal card after the first purchase is an
            # opportunity, even if the agent declines it. Track non-Copper too.
            if purchases_this_turn >= 1:
                if not extra_seen and mask[:end_buying].any():
                    usage["turns_with_legal_extra_purchase_opportunity"] += 1
                    extra_seen = True
                if not non_copper_extra_seen and mask[1:end_buying].any():
                    usage["turns_with_non_copper_extra_purchase_opportunity"] += 1
                    non_copper_extra_seen = True
        policy_observation = observation / divisors if preprocessing["mode"] == "fixed" else observation
        action, _ = model.predict(policy_observation, deterministic=deterministic, action_masks=mask)
        action_id = int(action.item())
        purchase_decisions += int(observation[FEATURE_INDEX["buys"]] > 0)
        province_opportunities += int(mask[5])
        province_declines += int(mask[5] and action_id != 5)
        copper_despite_silver_or_gold += int(action_id == 0 and (mask[1] or mask[2]))
        copper_despite_non_copper += int(action_id == 0 and mask[1:end_buying].any())
        if action_id != end_buying:
            card = card_ids[action_id]
            purchases[card] += 1
            if v1:
                purchases_this_turn += 1
                usage["turns_with_multiple_purchases"] += int(purchases_this_turn == 2)
                if v2 and purchases_this_turn > 1:
                    usage["purchases_after_first"] += 1
                    usage["non_copper_purchases_after_first"] += int(card != "copper")
            bought_this_turn = True
            own_turn = int(observation[FEATURE_INDEX["agent_turns_taken"]]) + 1
            if card in early_victory and own_turn <= 10:
                early_victory[card] += 1
            if card in first_turn and first_turn[card] is None:
                first_turn[card] = int(observation[FEATURE_INDEX["agent_turns_taken"]]) + 1
        else:
            if not bought_this_turn:
                no_purchase_turns += 1
                passes_with_affordable_non_copper += int(mask[1:end_buying].any())
            bought_this_turn = False
        observation, reward, terminated, truncated, info = env.step(action_id)
        episode_return += reward
        discounted_episode_return += return_gamma ** steps * reward
        steps += 1
        if terminated or truncated:
            outcome = (
                "truncated" if truncated else "tie" if info["winner"] is None
                else "win" if info["winner"] == env.agent_player else "loss"
            )
            return {
                "seed": seed, "evaluation_seed": seed, "outcome": outcome,
                "game_version": env.game_config.version,
                **(dict(**usage, first_village_turn=first_turn["village"],
                        first_council_room_turn=first_turn["council_room"]) if v1 else {}),
                **({"first_festival_turn": first_turn["festival"], **{
                    f"final_owned_{card}": int(observation[V2_FEATURE_INDEX[f"owned_{card}"]])
                    for card in ("village", "council_room", "festival")}} if v2 else {}),
                **({**{f"first_{card}_turn": first_turn[card] for card in ("militia", "moat")},
                    **{f"final_owned_{card}": int(observation[V3_FEATURE_INDEX[f"owned_{card}"]])
                       for card in ("militia", "moat")},
                    **militia_statistics(env.game, env.agent_player)} if v3 else {}),
                "agent_player": env.agent_player, "scores": info["scores"],
                "agent_vp": info["scores"][0], "opponent_vp": info["scores"][1],
                "first_silver_turn": first_turn["silver"], "first_gold_turn": first_turn["gold"],
                "first_estate_turn": first_turn["estate"], "early_victory_turn_cutoff": 10,
                **{f"purchases_{card}_first10_turns": count for card, count in early_victory.items()},
                "vp_differential": info["scores"][0] - info["scores"][1],
                "turn_count": sum(info["turns_taken"]),
                "agent_turns": info["turns_taken"][env.agent_player],
                "opponent_turns": info["turns_taken"][1 - env.agent_player],
                **{f"purchases_{card}": count for card, count in purchases.items()},
                "first_province_turn": first_turn["province"],
                "first_duchy_turn": first_turn["duchy"],
                "turns_without_purchase": no_purchase_turns,
                "passes_with_affordable_non_copper": passes_with_affordable_non_copper,
                "evaluation_reward_mode": env.reward_config.mode,
                "training_gamma": training_gamma, "evaluation_return_gamma": float(return_gamma),
                "winner": info["winner"], "terminated": terminated, "truncated": truncated,
                "end_reason": info["end_reason"], "turns_taken": info["turns_taken"],
                "steps": steps, "episode_return": episode_return,
                "discounted_episode_return": discounted_episode_return,
                "tie_adjusted_score": None if truncated else {"win": 1.0, "tie": 0.5, "loss": 0.0}[outcome],
                "purchase_decisions": purchase_decisions,
                "province_opportunities": province_opportunities,
                "province_declines": province_declines,
                "copper_despite_silver_or_gold": copper_despite_silver_or_gold,
                "copper_despite_non_copper": copper_despite_non_copper,
            }
