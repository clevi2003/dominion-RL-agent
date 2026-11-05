"""shows purchase steps and opponent progression, or batch masked random episodes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from random import Random
from statistics import mean

import numpy as np

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import ACTION_NAMES, END_BUYING
from dominion_rl.env.observations import FEATURE_INDEX


def run_episode(env: DominionEnv, seed: int, *, illustrate: bool, trace: bool) -> tuple[dict, int]:
    observation, info = env.reset(seed=seed)
    rng = Random(seed)
    if illustrate or trace:
        print(f"\nSeed {seed}; reward mode={env.reward_config.mode}; opponent={env.opponent}")
        print("Initial observation:", np.array2string(observation, max_line_width=100))
        print("Action order:", ACTION_NAMES)
        print("Initial mask:", info["action_mask"])
        print(env.render())
    steps = 0
    while True:
        legal = np.flatnonzero(env.action_masks()).tolist()
        # The single-game screenshot demo begins with a legal purchase and an
        # explicit End buying. Batch runs use uniform masked random decisions.
        if illustrate and steps == 0:
            action = 1 if 1 in legal else next(index for index in legal if index != END_BUYING)
        elif illustrate and steps == 1:
            action = END_BUYING
        else:
            action = rng.choice(legal)
        before_supply = env.game.state.supply.copy()
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if trace or (illustrate and steps <= 2):
            print(f"\nStep {steps}: {ACTION_NAMES[action]}; reward={reward:.6f}; "
                  f"terminated={terminated}; truncated={truncated}")
            print("Updated observation:", np.array2string(observation, max_line_width=100))
            print("Updated mask:", info["action_mask"])
            print("Supply changes:", {
                card: (before_supply[card], count)
                for card, count in env.game.state.supply.items() if before_supply[card] != count
            })
            print(env.render())
        if terminated or truncated:
            if illustrate or trace:
                print(f"\nFinal scores: {info['scores']}; winner={info['winner']}; "
                      f"reason={info['end_reason']}; steps={steps}; "
                      f"completed turns={info['turns_taken']}")
                print(f"Final inferable VP: {observation[FEATURE_INDEX['agent_vp']]:g} / "
                      f"{observation[FEATURE_INDEX['opponent_vp']]:g}; "
                      f"terminated={terminated}; truncated={truncated}")
            return info, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("terminal_only", "potential_based"), default="terminal_only")
    parser.add_argument("--opponent", choices=("big-money", "random"), default="big-money")
    parser.add_argument("--games", type=int, default=1, help="Consecutive seeds; batches use fully random legal decisions")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--trace", action="store_true", help="Show every environment step")
    args = parser.parse_args()
    if args.seed < 0 or args.games < 1 or args.max_turns < 1:
        parser.error("seed must be nonnegative; games and max-turns must be positive")
    env = DominionEnv(
        replace(GameConfig(), max_turns=args.max_turns), RewardConfig(mode=args.mode),
        opponent=args.opponent, render_mode="ansi",
    )
    outcomes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    durations: list[int] = []
    step_counts: list[int] = []
    for seed in range(args.seed, args.seed + args.games):
        info, steps = run_episode(env, seed, illustrate=args.games == 1, trace=args.trace)
        result = (
            "truncated" if info["end_reason"] == "turn_limit" else
            "tie" if info["winner"] is None else "win" if info["winner"] == 0 else "loss"
        )
        outcomes[result] += 1
        reasons[info["end_reason"]] += 1
        durations.append(sum(info["turns_taken"]))
        step_counts.append(steps)
    env.close()
    if args.games > 1:
        print(f"{args.games} masked random episodes, seeds {args.seed}..{args.seed + args.games - 1}; "
              f"mode={args.mode}; opponent={args.opponent}")
        print(f"Outcomes: {dict(sorted(outcomes.items()))}")
        print(f"End reasons: {dict(sorted(reasons.items()))}")
        print(f"Mean completed turns={mean(durations):.1f}; mean agent steps={mean(step_counts):.1f}")


if __name__ == "__main__":
    main()
