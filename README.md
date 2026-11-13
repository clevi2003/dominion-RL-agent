# Learning Dominion Deck-Building Strategies via Deep Reinforcement Learning

## Overview

Dominion is a deck-building game in which players buy cards to improve their decks
and earn victory points. This project implements a two-player Gymnasium environment
and uses MaskablePPO to learn purchasing and Action-playing strategies. It
investigates how reward design and opponent behavior affect learning. The two
networks are implemented and trained independently; they are not yet jointly
trained as a complete two-network agent.

The current game has eleven cards: Copper, Silver, Gold, Estate, Duchy, Province,
Village, Council Room, Festival, Militia, and Moat. It supports Action chains,
attacks, reactions, legal-action masks, and observations that hide private
opponent information. Fixed heuristic opponents provide training opponents.

## Setup

Use Python 3.11+ and run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## Running the project

Run the tests and demonstrate the eleven-card game:

```bash
python -m pytest -q
python scripts/demo_engine.py --config configs/game_v3.yaml --seed 42 --player0 militia-money --player1 engine-money --trace
```

Train and evaluate a purchasing policy (Actions play automatically):

```bash
python scripts/train_ppo.py --game-config configs/game_v3.yaml --timesteps 256 --checkpoint-freq 0 --seed 0 --output runs/purchase_smoke
python scripts/evaluate_ppo.py --game-config configs/game_v3.yaml --model runs/purchase_smoke/model.zip --games 1 --seed 20000 --output results/purchase_smoke.csv
```

Train and evaluate an Action policy (purchasing uses a fixed heuristic):

```bash
python scripts/action.py train --timesteps 256 --checkpoint-freq 0 --seed 0 --output runs/action_smoke
python scripts/action.py evaluate --model runs/action_smoke/model.zip --games 1 --seed 20000 --output results/action_smoke.csv
```

These small runs demonstrate functionality, not competitive performance. Use
larger training budgets for learning and fresh output names when repeating runs.
Both evaluators accept `--agent-player 0 1` to evaluate both player positions.

Purchasing rewards are configured in `configs/rewards.yaml`, with terminal
win/loss or VP-margin objectives and optional potential shaping. Purchasing PPO
settings live in `configs/ppo.yaml`; its gamma must match the reward gamma.
Action PPO settings live in `configs/action_ppo.yaml`. For checkpoint compatibility,
Action learning uses terminal VP margin divided by 48, gamma 1, no shaping, and a
fixed buyer. Earlier game configs remain available for older purchasing models;
use the game config that matches the checkpoint.

## Project structure

- `src/dominion_rl/game/`: card definitions, game state, and Dominion rules.
- `src/dominion_rl/env/`: Gymnasium environments, observations, masks, and rewards.
- `src/dominion_rl/agents/`: fixed purchasing, discard, and opponent policies.
- `src/dominion_rl/ppo.py` and `action_ppo.py`: shared training and policy validation.
- `configs/`: game, reward, and PPO settings.
- `scripts/`: a game demo and purchasing/Action training and evaluation tools.
- `tests/`: gameplay, environment, reward, and policy regression tests.

Checkpoints, logs, and evaluation results are written to ignored local `runs/`
and `results/` directories. Local research archives live in ignored `scratch/`.
