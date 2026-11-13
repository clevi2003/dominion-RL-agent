"""Run seeded backend games with fixed purchasing heuristics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import mean

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.random_agent import random_purchase
from dominion_rl.agents.engine_money import engine_money_purchase
from dominion_rl.agents.festival_money import festival_money_purchase
from dominion_rl.agents.militia_money import militia_money_purchase
from dominion_rl.config import GameConfig, load_game_config
from dominion_rl.game.engine import DominionGame
from dominion_rl.game.state import GameState


def run_game(
    config: GameConfig, policies: tuple[str, str], seed: int, show: bool, trace: bool, observer_player: int = 0
) -> GameState:
    game = DominionGame(replace(config, seed=seed))
    # Independent policy streams; changing policy randomness cannot consume shuffle draws.
    policy_rngs = (Random(seed * 2), Random(seed * 2 + 1))
    if show:
        print(f"Seed {seed}: player 0 {policies[0]} vs player 1 {policies[1]}")
        if config.version == "V1":
            print("Automatic Actions: Village first (+1 Card, +2 Actions); Council Room (+4 Cards, +1 Buy).")
            print("Each Council Room immediately gives the other player a private draw; its identity is not displayed.")
        if config.version == "V2":
            print("Automatic Actions: Village -> Festival -> Council Room; Festival grants +2 Actions, +1 Buy, +2 coins.")
        if config.version == "V3":
            print(f"V3 public trace; private hand visible only for observer P{observer_player}. Actions: Village -> Festival -> Council Room -> Moat -> Militia.")
    reported = 0
    while not (game.state.terminated or game.state.truncated):
        if show and config.version == "V3":
            for event in game.militia_resolutions[reported:]:
                print(f"  Militia P{event['attacker']} -> P{event['defender']}: Moat revealed={event['moat_revealed']}; discarded={event['discard_count']} (identities private).")
            reported = len(game.militia_resolutions)
        state = game.state
        index = state.current_player
        player = state.players[index]
        legal = game.legal_purchases()
        if policies[index] == "random":
            decision = random_purchase(legal, policy_rngs[index])
        elif policies[index] in ("engine-money", "festival-money", "militia-money"):
            owned = Counter(player.deck + player.hand + player.discard + player.in_play)
            purchase = {"engine-money": engine_money_purchase, "festival-money": festival_money_purchase,
                        "militia-money": militia_money_purchase}[policies[index]]
            decision = purchase(legal, state.supply["province"], owned)
        else:
            decision = big_money_purchase(legal, state.supply["province"])
        verbose = show and (trace or state.turn_number <= 4)
        if verbose:
            print(
                f"Turn {state.turn_number}, P{index}: hand={player.hand if config.version != 'V3' or index == observer_player else '<private>'}, "
                f"played={player.in_play}, coins={player.coins}, buys={player.buys}; "
                f"legal={legal}"
            )
        if decision is None:
            turn = state.turn_number
            game.end_buying()
            if verbose:
                following = (
                    f"next P{state.current_player}, turn {state.turn_number}"
                    if not (state.terminated or state.truncated) else "episode ended"
                )
                print(f"  End buying: completed turn {turn}; {following}.")
        else:
            before = state.supply[decision]
            game.buy_card(decision)
            if verbose:
                print(f"  Buy {decision}: Supply {before} -> {state.supply[decision]}.")
    if show:
        if not trace and game.state.turn_number > 4:
            print("... (use --trace for every decision)")
        outcome = (
            "truncated, no winner" if game.state.truncated else
            "shared tie" if game.state.winner is None else f"player {game.state.winner} wins"
        )
        print(
            f"Final VP: P0={game.score(0)}, P1={game.score(1)}; {outcome}; "
            f"reason={game.state.end_reason}; "
            f"completed turns={sum(p.turns_taken for p in game.state.players)} "
            f"({game.state.players[0].turns_taken}/{game.state.players[1].turns_taken})."
        )
    return game.state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="First game seed (default: 42)")
    parser.add_argument("--games", type=int, default=1, help="Consecutive seeded games")
    parser.add_argument("--max-turns", type=int, help="Override total individual-turn cap")
    parser.add_argument("--config", type=Path, help="Game YAML (otherwise use GameConfig defaults)")
    parser.add_argument("--player0", choices=("big-money", "random", "engine-money", "festival-money", "militia-money"), default="big-money")
    parser.add_argument("--player1", choices=("big-money", "random", "engine-money", "festival-money", "militia-money"), default="random")
    parser.add_argument("--trace", action="store_true", help="Show every decision in every game")
    parser.add_argument("--observer-player", type=int, choices=(0, 1), default=0, help="V3 trace shows only this player's private hand")
    args = parser.parse_args()
    if args.games < 1 or args.seed < 0 or (args.max_turns is not None and args.max_turns < 1):
        parser.error("games and max-turns must be positive; seed must be nonnegative")
    config = load_game_config(args.config) if args.config else GameConfig()
    if args.max_turns is not None:
        config = replace(config, max_turns=args.max_turns)
    if "engine-money" in (args.player0, args.player1) and config.version not in ("V1", "V2", "V3"):
        parser.error("engine-money requires a V1, V2 or V3 config")
    if "festival-money" in (args.player0, args.player1) and config.version not in ("V2", "V3"):
        parser.error("festival-money requires a V2 or V3 config")
    if "militia-money" in (args.player0, args.player1) and config.version != "V3":
        parser.error("militia-money requires --config configs/game_v3.yaml")
    outcomes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    durations: list[int] = []
    for seed in range(args.seed, args.seed + args.games):
        state = run_game(
            config, (args.player0, args.player1), seed,
            show=args.games == 1 or args.trace, trace=args.trace, observer_player=args.observer_player,
        )
        result = (
            "truncated" if state.truncated else
            "tie" if state.winner is None else f"player{state.winner}_wins"
        )
        outcomes[result] += 1
        reasons[state.end_reason] += 1
        durations.append(sum(player.turns_taken for player in state.players))
    if args.games > 1:
        print(f"{args.games} games, seeds {args.seed}..{args.seed + args.games - 1}:")
        print(f"Outcomes: {dict(sorted(outcomes.items()))}")
        print(f"End reasons: {dict(sorted(reasons.items()))}")
        print(f"Completed turns: mean={mean(durations):.1f}, min={min(durations)}, max={max(durations)}")


if __name__ == "__main__":
    main()
