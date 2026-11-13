"""Role validation and deterministic evaluation for the separate Action policy."""

import torch

from dominion_rl.agents.action_rich import buying_config
from dominion_rl.env.action_env import (ACTION_CARDS, END_ACTION, DominionActionEnv,
                                       action_contract, action_preprocessing, action_reward)
from dominion_rl.ppo import model_reward_config, model_training_gamma, militia_statistics


def validate_action_model(model, training=None):
    """Never infer an Action role from spaces; reject absent/mismatched metadata."""
    if getattr(model, "policy_role", "purchase") != "action":
        raise ValueError("An Action model is required; purchase models cannot select Actions")
    if getattr(model, "game_contract", None) != action_contract():
        raise ValueError("Incompatible Action observation schema/action mapping")
    if model.observation_space.shape != (67,) or getattr(model.action_space, "n", None) != 6:
        raise ValueError("Action model spaces must be 67/6")
    if getattr(model, "observation_preprocessing", None) != action_preprocessing():
        raise ValueError("Action preprocessing must be raw with exact feature order")
    if getattr(model, "fixed_buying_controller", None) != buying_config():
        raise ValueError("Fixed buying-controller rules changed or are missing")
    settings = dict(net_arch=[64, 64], activation="tanh", observation_scaling="raw", gae_lambda=0.95)
    if getattr(model, "action_policy_settings", None) != settings:
        raise ValueError("Action architecture/settings metadata mismatch")
    native_policy = getattr(model, "policy_kwargs", {})
    if (native_policy.get("net_arch") != [64, 64] or native_policy.get("activation_fn") is not torch.nn.Tanh
            or getattr(model, "gae_lambda", None) != 0.95):
        raise ValueError("Native Action network/settings disagree with saved metadata")
    if type(getattr(model, "num_timesteps", None)) is not int or model.num_timesteps < 0:
        raise ValueError("Action model requires a nonnegative checkpoint step")
    if model_training_gamma(model, training) != 1 or model_reward_config(model, training) != action_reward():
        raise ValueError("Action reward must be terminal VP margin, gamma=1, no shaping")
    if type(getattr(model, "training_seed", None)) is not int or model.training_seed < 0 or model.seed != model.training_seed:
        raise ValueError("Action model requires its original training seed")
    if training is not None:
        expected = dict(policy_role="action", game_contract=action_contract(),
                        fixed_buying_controller=buying_config(), seed=model.training_seed,
                        features=action_contract()["features"], actions=action_contract()["actions"],
                        observation_preprocessing=action_preprocessing())
        if (training.get("game", {}).get("version") != "V3"
                or training.get("activation") != "tanh"
                or training.get("ppo", {}).get("policy_kwargs") != dict(net_arch=[64,64], activation_fn="tanh")
                or training.get("ppo", {}).get("gae_lambda", 0.95) != 0.95):
            raise ValueError("Training snapshot and native Action settings disagree")
        if any(training.get(key) != value for key, value in expected.items()):
            raise ValueError("Training snapshot and native Action metadata disagree")
    return action_contract()


def play_action_episode(model, env: DominionActionEnv, seed):
    """None selects the historical priority in the same manual env and fixed buyer.

    Both baseline and learned policy see the same forced decisions and seeds.
    The baseline always plays its highest-priority legal Action, then Ends.
    """
    if not isinstance(env, DominionActionEnv):
        raise ValueError("Action evaluation requires DominionActionEnv")
    if model is not None:
        validate_action_model(model)
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError("Action model/environment spaces disagree")
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        mask = env.action_masks()
        if model is None:
            card = env.game.automatic_action_choice()
            action = END_ACTION if card is None else ACTION_CARDS.index(card)
        else:
            predicted, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(predicted)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    agent_vp, opponent_vp = info["scores"]
    turns = info["turns_taken"][env.agent_player]
    decisions = env.metrics["action_decisions"]
    outcome = ("truncation" if truncated else "tie" if info["winner"] is None else
               "win" if info["winner"] == env.agent_player else "loss")
    successful_attacks = {f"militia_successful_attacks_by_{who}":
        sum(e["attacker"] == seat and e["discard_count"] > 0 for e in env.game.militia_resolutions)
        for who, seat in (("agent", env.agent_player), ("opponent", 1-env.agent_player))}
    return dict(condition="deterministic" if model is None else "learned", policy_role="action",
        reward_condition="terminal_only", terminal_objective="vp_margin", gamma=1.0,
        training_seed=None if model is None else model.training_seed,
        checkpoint_step=None if model is None else model.num_timesteps,
        evaluation_seed=seed, opponent=env.opponent, agent_player=env.agent_player,
        buyer="action-rich-v1", agent_vp=agent_vp, opponent_vp=opponent_vp,
        vp_differential=agent_vp-opponent_vp, outcome=outcome,
        tie_adjusted_score=None if truncated else 1.0 if outcome == "win" else 0.5 if outcome == "tie" else 0.0,
        terminated=terminated, truncated=truncated, end_reason=info["end_reason"],
        agent_turns_taken=turns, opponent_turns_taken=info["turns_taken"][1-env.agent_player],
        total_turns=sum(info["turns_taken"]), episode_return=total_reward,
        **env.metrics, **militia_statistics(env.game, env.agent_player), **successful_attacks,
        multi_type_fraction=env.metrics["multiple_type_decisions"]/decisions,
        forced_end_fraction=env.metrics["forced_end_decisions"]/decisions,
        mean_buy_phase_coins=env.metrics["buy_phase_coins"]/turns if turns else None,
        mean_action_decisions_per_turn=decisions/turns if turns else None)
