"""Player-relative observations, rewards and seeded progression for either seat."""

from copy import deepcopy

import numpy as np
import pytest

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import ACTION_NAMES, END_BUYING
from dominion_rl.env.observations import FEATURE_INDEX, encode_observation
from dominion_rl.game import DominionGame


def test_player_one_observation_is_relative_and_excludes_hidden_distribution():
    state = DominionGame().state
    swapped = deepcopy(state)
    swapped.players = (state.players[1], state.players[0])
    np.testing.assert_array_equal(encode_observation(state, 1), encode_observation(swapped))
    original = encode_observation(state, 1)
    opponent = state.players[0]
    cards = opponent.deck + opponent.hand + opponent.discard + opponent.in_play
    opponent.deck, opponent.hand, opponent.discard, opponent.in_play = [], cards[:1], cards[1:], []
    state.players[1].deck.reverse()
    np.testing.assert_array_equal(original, encode_observation(state, 1))


@pytest.mark.parametrize("opponent", ["big-money", "random"])
def test_player_one_reaches_legal_decisions_and_reproduces_trajectories(opponent):
    env = DominionEnv(GameConfig(max_turns=12), opponent=opponent, agent_player=1)
    histories = []
    try:
        for _ in range(2):
            obs, info = env.reset(seed=321)
            assert info["current_player"] == 1 and info["turns_taken"] == (1, 0)
            history = [obs.tolist()]
            while True:
                legal = env.game.legal_purchases()
                mask = env.action_masks()
                np.testing.assert_array_equal(mask, [card in legal for card in ACTION_NAMES[:-1]] + [True])
                # Buy Silver if possible, otherwise end; use only public information.
                action = 1 if mask[1] else END_BUYING
                obs, reward, terminated, truncated, info = env.step(action)
                history.append((obs.tolist(), reward, terminated, truncated, info["winner"]))
                if terminated or truncated:
                    break
                assert info["current_player"] == 1
            assert truncated and not terminated and info["winner"] is None
            assert sum(info["turns_taken"]) == 12
            histories.append(history)
        assert histories[0] == histories[1]
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["terminal_only", "potential_based"])
@pytest.mark.parametrize("outcome", ["win", "loss", "tie"])
def test_player_one_rewards_and_winner_use_the_correct_perspective(mode, outcome):
    env = DominionEnv(agent_player=1, reward_config=RewardConfig(mode=mode))
    try:
        env.reset(seed=12)
        state = env.game.state
        if outcome != "tie":
            state.players[1 if outcome == "win" else 0].discard.append("province")
        state.supply["province"] = 0
        obs, reward, terminated, truncated, info = env.step(END_BUYING)
        expected = {"win": 1.0, "loss": -1.0, "tie": 0.0}[outcome]
        if mode == "potential_based":
            expected -= {"win": 6 / 48, "loss": -6 / 48, "tie": 0}[outcome]
        assert terminated and not truncated
        assert reward == pytest.approx(expected)
        assert info["winner"] == {"win": 1, "loss": 0, "tie": None}[outcome]
        assert info["scores"] == (obs[FEATURE_INDEX["agent_vp"]], obs[FEATURE_INDEX["opponent_vp"]])
        assert not env.action_masks().any()
    finally:
        env.close()


def test_player_one_rejects_a_cap_before_its_first_decision():
    with pytest.raises(ValueError, match="max_turns >= 2"):
        DominionEnv(GameConfig(max_turns=1), agent_player=1)
