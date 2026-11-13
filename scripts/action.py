"""Train or evaluate an independent Action policy with a fixed heuristic buyer."""

import argparse
import csv
from pathlib import Path

import yaml

from dominion_rl.config import load_game_config
from dominion_rl.env.action_env import DominionActionEnv, OPPONENTS, action_reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train one Action policy from scratch")
    evaluate = commands.add_parser("evaluate", help="Evaluate one saved Action policy")
    for command in (train, evaluate):
        command.add_argument("--game-config", type=Path, default=Path("configs/game_v3.yaml"))
        command.add_argument("--opponent", choices=OPPONENTS, default="militia-money")
        command.add_argument("--seed", type=int, default=0 if command is train else 20_000)
        command.add_argument("--output", type=Path, required=True,
                             help="New run directory" if command is train else "New outcome CSV")
    train.add_argument("--ppo-config", type=Path, default=Path("configs/action_ppo.yaml"))
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--checkpoint-freq", type=int, default=10_000,
                       help="Decision steps; 0 disables periodic saves")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--games", type=int, default=10, help="Games per seat")
    evaluate.add_argument("--agent-player", type=int, nargs="+", choices=(0, 1), default=[0])
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    game = load_game_config(args.game_config)
    if game.version != "V3":
        parser.error("Action policies require the eleven-card V3 game")
    if args.command == "train":
        if args.timesteps <= 0 or args.checkpoint_freq < 0:
            parser.error("timesteps must be positive; checkpoint-freq must be nonnegative")
        from dominion_rl.ppo import train_ppo

        with args.ppo_config.open(encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)
        if not isinstance(settings, dict):
            parser.error("PPO configuration must be a YAML mapping")
        path = train_ppo(
            args.output, game_config=game, reward_config=action_reward(), ppo_kwargs=settings,
            seed=args.seed, total_timesteps=args.timesteps, opponent=args.opponent,
            checkpoint_freq=args.checkpoint_freq, policy_role="action",
        )
        print(f"Saved model: {path}")
        return
    if args.games < 1 or (1 in args.agent_player and game.max_turns < 2):
        parser.error("games must be positive; player 1 requires max_turns >= 2")
    if args.output.exists():
        parser.error("evaluation output must be a new CSV file")
    from sb3_contrib import MaskablePPO
    import torch

    from dominion_rl.action_ppo import play_action_episode, validate_action_model

    torch.set_num_threads(1)
    model = MaskablePPO.load(args.model, device="cpu")
    metadata = args.model.parent / "run.yaml"
    if not metadata.exists():
        metadata = args.model.parent.parent / "run.yaml"
    training = yaml.safe_load(metadata.read_text(encoding="utf-8")) if metadata.exists() else None
    validate_action_model(model, training)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = None
        for seat in args.agent_player:
            env = DominionActionEnv(game, action_reward(), args.opponent, seat)
            try:
                for seed in range(args.seed, args.seed + args.games):
                    row = play_action_episode(model, env, seed)
                    row.update(model=str(args.model), max_turns=game.max_turns)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
            finally:
                env.close()
    print(f"Saved game outcomes: {args.output}")


if __name__ == "__main__":
    main()
