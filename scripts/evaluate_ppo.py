"""Evaluate one or more checkpoints on identical seeds, opponents and seats."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

from sb3_contrib import MaskablePPO
import torch
import yaml

from dominion_rl.config import RewardConfig, load_game_config
from dominion_rl.env import DominionEnv
from dominion_rl.ppo import model_game_contract, model_preprocessing, model_reward_config, model_training_gamma, play_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, nargs="+", required=True)
    parser.add_argument("--games", type=int, default=10, help="Games per checkpoint/opponent/seat")
    parser.add_argument("--seed", type=int, default=20_000, help="First held-out seed; consecutive seeds follow")
    parser.add_argument("--opponent", nargs="+", default=["big-money"], help="Fixed heuristic names or IDs from --opponent-pool")
    parser.add_argument("--opponent-pool", type=Path, help="Frozen policy manifest for learned opponent IDs")
    parser.add_argument("--agent-player", type=int, nargs="+", choices=(0, 1), default=[0])
    parser.add_argument("--game-config", type=Path, default=Path("configs/game.yaml"))
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--observation-scaling", choices=("raw", "fixed"), help="Require this model preprocessing; never override it")
    parser.add_argument("--reward-source", choices=("terminal", "training"), default="terminal",
                        help="Default: common terminal rewards. training recovers exact saved shaping for return diagnostics")
    parser.add_argument("--return-gamma", type=float, choices=(0.99, 1.0),
                        help="Discount evaluation returns at this gamma; default: each model's training gamma")
    parser.add_argument("--stochastic", action="store_true", help="Sample masked actions instead of taking the most likely")
    parser.add_argument("--output", type=Path, required=True, help="New CSV file for game outcomes")
    args = parser.parse_args()
    if args.seed < 0 or args.games < 1:
        parser.error("seed must be nonnegative and games must be positive")
    game = load_game_config(args.game_config)
    if args.max_turns is not None:
        game = replace(game, max_turns=args.max_turns)
    if 1 in args.agent_player and game.max_turns < 2:
        parser.error("player 1 requires max_turns >= 2")
    if "engine-money" in args.opponent and game.version not in ("V1", "V2", "V3"):
        parser.error("engine-money requires a V1, V2 or V3 game config")
    if "festival-money" in args.opponent and game.version not in ("V2", "V3"):
        parser.error("festival-money requires a V2 or V3 game config")
    if "militia-money" in args.opponent and game.version != "V3":
        parser.error("militia-money requires --game-config configs/game_v3.yaml")
    frozen_ids = [name for name in args.opponent if name not in ("big-money","random","engine-money","festival-money","militia-money")]
    pool_snapshot = None
    if frozen_ids:
        if args.opponent_pool is None or game.version != "V1":
            parser.error("Learned opponents require V1 and --opponent-pool")
        from dominion_rl.agents.frozen_policy import verify_manifest, pool_provenance
        pool = verify_manifest(args.opponent_pool)
        if not set(frozen_ids) <= {p['id'] for p in pool['policies']}:
            parser.error("Unknown frozen opponent ID")
        pool_snapshot = pool_provenance(args.opponent_pool,pool)
    elif args.opponent_pool is not None:
        parser.error("--opponent-pool requires learned opponent IDs")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output.with_suffix(".yaml")
    if args.output.exists() or snapshot_path.exists():
        parser.error("evaluation CSV and YAML snapshot must both be new")
    snapshot = {
        "game": asdict(game), "evaluation_reward_mode": "terminal_only" if args.reward_source == "terminal" else "training",
        "evaluation_reward_source": args.reward_source,
        "evaluation_terminal_objective": "win_loss" if args.reward_source == "terminal" else "training",
        "seed_start": args.seed, "games_per_model_opponent_seat": args.games,
        "opponents": args.opponent, "agent_players": args.agent_player,
        "deterministic": not args.stochastic,
        "evaluation_return_gamma_override": args.return_gamma,
        "evaluation_return_gamma_source": "training_gamma" if args.return_gamma is None else "explicit_override",
        "outcome_metrics": "Win/loss/tie and VP differential are undiscounted; compare these across gammas",
        "models": [],
    }
    if pool_snapshot is not None:snapshot["frozen_opponent_pool"] = pool_snapshot
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = None
        for path in args.model:
            model = MaskablePPO.load(path, device="cpu")
            # Both final models and checkpoints retain their training run metadata.
            metadata_path = path.parent / "run.yaml"
            if not metadata_path.exists():
                metadata_path = path.parent.parent / "run.yaml"
            training = yaml.safe_load(metadata_path.read_text()) if metadata_path.exists() else {}
            contract = model_game_contract(model, training)
            recovered_reward = model_reward_config(model, training)
            condition = recovered_reward.mode if recovered_reward is not None else "unknown"
            design = (condition if condition == "terminal_only" else recovered_reward.potential) if recovered_reward is not None else "unknown"
            training_seed = training.get("seed", model.seed)
            training_gamma = model_training_gamma(model, training)
            return_gamma = training_gamma if args.return_gamma is None else args.return_gamma
            if args.reward_source == "training" and recovered_reward is None:
                raise ValueError("Training reward is unknown: retain legacy run.yaml or use common terminal outcomes")
            if args.reward_source == "training" and return_gamma != recovered_reward.gamma:
                raise ValueError("Training reward evaluation requires matching return/shaping gamma")
            evaluation_reward = recovered_reward if args.reward_source == "training" else RewardConfig(gamma=return_gamma)
            preprocessing = model_preprocessing(model, args.observation_scaling)
            activation = model.policy.activation_fn.__name__.lower()
            if training.get("observation_preprocessing", preprocessing) != preprocessing:
                raise ValueError(f"Training snapshot and model preprocessing disagree: {path}")
            if training.get("activation", activation) != activation:
                raise ValueError(f"Training snapshot and model activation disagree: {path}")
            snapshot["models"].append({
                "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
                "checkpoint_step": model.num_timesteps, "reward_condition": condition,
                "training_seed": training_seed, "training_gamma": training_gamma,
                "evaluation_return_gamma": return_gamma,
                "training_config": training, "activation": activation,
                "training_reward_config": asdict(recovered_reward) if recovered_reward is not None else None,
                "training_terminal_objective": recovered_reward.terminal_objective if recovered_reward is not None else "win_loss",
                "evaluation_terminal_objective": evaluation_reward.terminal_objective,
                "reward_design": design, "evaluation_reward_config": asdict(evaluation_reward),
                "observation_preprocessing": preprocessing,
                "game_contract": contract,
            })
            # Save before playing so even an interrupted evaluation keeps its settings.
            snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
            for opponent in args.opponent:
                for seat in args.agent_player:
                    # Common terminal outcomes; discounted returns retain explicit units.
                    env = DominionEnv(game, evaluation_reward,
                                      opponent="frozen" if opponent in frozen_ids else opponent, agent_player=seat,
                                      opponent_pool=args.opponent_pool if opponent in frozen_ids else None,
                                      frozen_opponent_id=opponent if opponent in frozen_ids else None)
                    outcomes: Counter[str] = Counter()
                    try:
                        for seed in range(args.seed, args.seed + args.games):
                            result = play_episode(model, env, seed, deterministic=not args.stochastic,
                                                  return_gamma=return_gamma)
                            result.update(
                                activation=activation, observation_scaling=preprocessing["mode"],
                                model=str(path), opponent=opponent, deterministic=not args.stochastic,
                                max_turns=game.max_turns, reward_condition=condition,
                                training_seed=training_seed, checkpoint_step=model.num_timesteps,
                                reward_design=design, training_potential=recovered_reward.potential if recovered_reward is not None else "unknown",
                                economy_coefficient=recovered_reward.economy_coefficient if recovered_reward is not None else None,
                                training_opponent=training.get("opponent", "unknown"),
                                training_terminal_objective=recovered_reward.terminal_objective if recovered_reward is not None else "win_loss",
                                evaluation_terminal_objective=evaluation_reward.terminal_objective,
                            )
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(result))
                                writer.writeheader()
                            writer.writerow(result)
                            outcomes[result["outcome"]] += 1
                    finally:
                        env.close()
                    print(f"{path}: {opponent}, seat {seat}, {args.games} games: {dict(sorted(outcomes.items()))}")
    print(f"Saved game outcomes: {args.output}; snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
