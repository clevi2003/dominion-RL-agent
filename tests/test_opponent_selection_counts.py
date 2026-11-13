"""Selection counters measure resets, not buying decisions or completed turns."""
from dominion_rl.config import GameConfig
from dominion_rl.env import DominionEnv


def test_multiple_buys_and_end_buying_do_not_select_an_opponent():
    env = DominionEnv(GameConfig(version='V1',max_turns=8),opponent='engine-money')
    try:
        env.reset(seed=57)
        initial = dict(env.opponent_reset_counts)
        assert sum(initial.values()) == 1
        player = env.game.state.players[env.agent_player]
        player.coins,player.buys = 6,2
        env.step(1)  # Silver, one Buy remains.
        assert env.opponent_reset_counts == initial
        env.step(1)  # Another Silver, no Buys remain.
        assert env.opponent_reset_counts == initial
        selected = env.episode_opponent
        env.step(env.end_buying)
        assert env.episode_opponent == selected
        assert env.opponent_reset_counts == initial
        env.reset()
        assert sum(env.opponent_reset_counts.values()) == 2
    finally:
        env.close()
