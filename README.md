# Learning Dominion Deck-Building Strategies via Deep Reinforcement Learning

This project looks at how reward shaping impacts learning efficiency and
opponent specific purchasing in two player Dominion.

## Install and run

Use Python 3.11+ from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python scripts/demo_env.py --seed 42
python scripts/demo_env.py --seed 42 --mode potential_based --trace
python scripts/demo_env.py --seed 0 --games 100 --opponent random
python scripts/demo_engine.py --seed 42
```

The environment demo shows the initial vector/mask, a purchase, updated Supply,
explicit End buying with automatic opponent progression, and a complete result.
Single-game mode starts with that illustrative purchase/end prefix, then samples
legal random decisions. Batch mode samples every decision uniformly from the
mask, including the end buying action; `--trace` shows every step (like a verbose flag, basically). `--mode`, `--opponent`,
`--seed`, `--games`, and `--max-turns` configure runs. Batch seeds increment by 1.
These demos validate the environment; they don't actually train anything yet

## Actions and observations

The canonical mapping is in `ACTION_NAMES` in `env/dominion_env.py`:
`0 Copper`, `1 Silver`, `2 Gold`, `3 Estate`, `4 Duchy`, `5 Province`, `6 End buying`.
It stays fixed even if the configuration's card order changes.

Observations are float32 arrays of 41 features. Every six card block follows the order Copper, Silver, Gold, Estate, Duchy, Province.

| Indices | Features |
|---|---|
| 0–1 | Agent coins, Buys |
| 2–7 | Agent hand counts |
| 8–13 | Agent in-play counts |
| 14–19 | Agent total ownership counts |
| 20–25 | Opponent inferable total ownership counts |
| 26–31 | Supply counts |
| 32–35 | Agent deck size, discard size, agent VP, opponent VP |
| 36–40 | Turn number, agent/opponent completed turns, Provinces remaining, empty piles |

Hidden draw order and the opponent's hand distribution are excluded. Total card
ownership is inferable from known starting cards and observable purchases, so cards
and VP can be calculated.