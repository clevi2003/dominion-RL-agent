"""Train a masked purchasing policy using simple YAML experiment settings."""

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from dominion_rl.config import load_game_config, load_reward_config
from dominion_rl.ppo import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("terminal_only", "potential_based"))
    parser.add_argument("--terminal-objective", choices=("win_loss", "vp_margin"),
                        help="Terminal target, independent of shaping; default: reward YAML or legacy win_loss")
    parser.add_argument("--potential", choices=("vp", "vp_economy"), help="Potential for potential_based mode; default: reward YAML")
    parser.add_argument("--gamma", type=float, choices=(0.99, 1.0),
                        help="Override BOTH PPO and reward-shaping gamma (default: YAML values)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--opponent", choices=("big-money", "random", "mixed", "curriculum", "engine-money", "festival-money", "militia-money", "historical"), default="big-money")
    parser.add_argument("--opponent-pool", type=Path, help="Fixed historical pool YAML; historical training only")
    parser.add_argument("--curriculum-config", type=Path, help="YAML mapping with curriculum stage list; curriculum mode only")
    parser.add_argument("--game-config", type=Path, default=Path("configs/game.yaml"))
    parser.add_argument("--reward-config", type=Path, default=Path("configs/rewards.yaml"))
    parser.add_argument("--ppo-config", type=Path, default=Path("configs/ppo.yaml"))
    parser.add_argument("--activation", choices=("tanh", "relu"), help="Override policy_kwargs.activation_fn")
    parser.add_argument("--observation-scaling", choices=("raw", "fixed"), help="Override policy input scaling")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000, help="Decision steps; 0 disables periodic saves")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    args = parser.parse_args()
    if (args.opponent == "historical") != (args.opponent_pool is not None):
        parser.error("--opponent-pool is required only with --opponent historical")
    curriculum = None
    if args.curriculum_config is not None:
        if args.opponent != "curriculum":
            parser.error("--curriculum-config requires --opponent curriculum")
        with args.curriculum_config.open(encoding="utf-8") as stream:
            curriculum = yaml.safe_load(stream)["curriculum"]
    game = load_game_config(args.game_config)
    reward = load_reward_config(args.reward_config)
    if args.terminal_objective is not None:
        reward = replace(reward, terminal_objective=args.terminal_objective)
    if args.mode is not None:
        reward = replace(reward, mode=args.mode)
    if args.potential is not None:
        reward = replace(reward, potential=args.potential)
    with args.ppo_config.open(encoding="utf-8") as stream:
        ppo_kwargs = yaml.safe_load(stream)
    if not isinstance(ppo_kwargs, dict):
        parser.error("PPO configuration must be a YAML mapping of MaskablePPO hyperparameters")
    if args.gamma is not None:
        reward = replace(reward, gamma=args.gamma)
        ppo_kwargs["gamma"] = args.gamma
    if args.activation is not None:
        ppo_kwargs.setdefault("policy_kwargs", {})["activation_fn"] = args.activation
    if args.observation_scaling is not None:
        ppo_kwargs["observation_scaling"] = args.observation_scaling
    path = train_ppo(
        args.output, game_config=game, reward_config=reward, ppo_kwargs=ppo_kwargs,
        seed=args.seed, total_timesteps=args.timesteps, opponent=args.opponent,
        checkpoint_freq=args.checkpoint_freq, curriculum=curriculum, opponent_pool=args.opponent_pool,
    )
    print(f"Saved model: {path}")


if __name__ == "__main__":
    main()
