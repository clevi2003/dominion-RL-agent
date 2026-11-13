from collections import Counter
from copy import deepcopy
from dataclasses import replace
import random
from unittest.mock import patch

import gymnasium as gym
from gymnasium.utils.env_checker import check_env, data_equivalence
from gymnasium.utils.passive_env_checker import env_render_passive_checker
import numpy as np
import pytest

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import ACTION_NAMES, END_BUYING
from dominion_rl.env.observations import FEATURE_INDEX, encode_observation
from dominion_rl.game import GameState, PlayerState


def constructed_env(agent=None, opponent=None, *, max_turns=200, mode="terminal_only"):
    env = DominionEnv(GameConfig(max_turns=max_turns), RewardConfig(mode=mode), render_mode="ansi")
    env.reset(seed=42)
    env.game.state = GameState(
        players=(agent or PlayerState(), opponent or PlayerState()),
        supply={"copper": 46, "silver": 40, "gold": 30,
                "estate": 8, "duchy": 8, "province": 8},
    )
    return env


def snapshot(env):
    return (deepcopy(env.game.state), env.game._rng.getstate(), env._opponent_rng.getstate(),
            deepcopy(env.np_random.bit_generator.state))


def test_spaces_action_mapping_reset_and_mask_types():
    env = DominionEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(env, gym.Env)
    assert env.action_space == gym.spaces.Discrete(7)
    assert ACTION_NAMES == ("copper", "silver", "gold", "estate", "duchy", "province", "end_buying")
    assert END_BUYING == 6
    assert env.observation_space.shape == obs.shape == (41,)
    assert obs.dtype == env.observation_space.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    mask = env.action_masks()
    assert mask.shape == (7,) and mask.dtype == np.bool_
    np.testing.assert_array_equal(info["action_mask"], mask)
    assert mask[END_BUYING] and mask[0]
    assert env.opponent == "big-money"


def test_masks_delegate_legality_and_remain_canonical_when_config_is_reordered():
    config = GameConfig()
    env = DominionEnv(replace(config, cards=tuple(reversed(config.cards))))
    env.reset(seed=42)
    state = env.game.state
    state.players[0].coins = 3
    state.supply["silver"] = 0
    np.testing.assert_array_equal(env.action_masks(), [True, False, False, True, False, False, True])
    state.players[0].buys = 0
    np.testing.assert_array_equal(env.action_masks(), [False] * 6 + [True])
    state.players[0].buys = 1
    state.players[0].coins = 0
    env.step(0)  # Zero-cost Copper is legal.
    assert state.players[0].discard[-1] == "copper"


def test_purchase_does_not_advance_or_run_opponent_and_values_are_fresh():
    env = constructed_env(PlayerState(coins=3), PlayerState(hand=["gold", "gold", "silver"]))
    state = env.game.state
    old_opponent = deepcopy(state.players[1])
    old_rng = env._opponent_rng.getstate()
    obs, reward, terminated, truncated, info = env.step(np.int64(1))
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert (state.current_player, state.turn_number) == (0, 1)
    assert state.players[0].discard == ["silver"] and state.supply["silver"] == 39
    assert state.players[1] == old_opponent and env._opponent_rng.getstate() == old_rng
    assert obs[FEATURE_INDEX["owned_silver"]] == 1
    assert obs[FEATURE_INDEX["buys"]] == 0
    np.testing.assert_array_equal(info["action_mask"], [False] * 6 + [True])
    obs[:] = -999
    info["action_mask"][:] = True
    assert encode_observation(state)[FEATURE_INDEX["supply_silver"]] == 39
    assert not env.action_masks()[1]


def test_end_buying_runs_opponent_purchase_and_end_then_returns_agent_boundary():
    env = constructed_env(PlayerState(), PlayerState(hand=["copper"] * 3))
    obs, reward, terminated, truncated, info = env.step(END_BUYING)
    assert not terminated and not truncated and reward == 0
    state = env.game.state
    assert state.current_player == 0 and state.turn_number == 3
    assert [p.turns_taken for p in state.players] == [1, 1]
    assert state.supply["silver"] == 39
    assert obs[FEATURE_INDEX["opponent_owned_silver"]] == 1
    assert info["current_player"] == 0 and info["scores"] == (0, 0)


@pytest.mark.parametrize("action", [-1, 7, 1.0, "1", None, True, np.bool_(False), np.array([1])])
def test_invalid_action_types_and_range_do_not_mutate_any_state_or_rng(action):
    env = DominionEnv()
    env.reset(seed=42)
    before = snapshot(env)
    with pytest.raises(ValueError, match="integer"):
        env.step(action)
    assert snapshot(env) == before


@pytest.mark.parametrize("condition", ["unaffordable", "empty", "no_buys"])
def test_masked_actions_raise_without_mutation(condition):
    env = constructed_env(PlayerState(coins=3))
    if condition == "unaffordable":
        action = 2
    else:
        action = 1
        if condition == "empty":
            env.game.state.supply["silver"] = 0
        else:
            env.game.state.players[0].buys = 0
    before = snapshot(env)
    with pytest.raises(ValueError, match="Illegal action"):
        env.step(action)
    assert snapshot(env) == before


def test_true_ending_on_agent_turn_waits_for_explicit_end_and_awards_win():
    env = constructed_env(PlayerState(coins=8))
    env.game.state.supply["province"] = 1
    _, reward, terminated, truncated, _ = env.step(5)
    assert reward == 0 and not terminated and not truncated
    obs, reward, terminated, truncated, info = env.step(END_BUYING)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["winner"] == 0 and info["end_reason"] == "province_empty"
    assert info["turns_taken"] == (1, 0) and info["scores"] == (6, 0)
    assert obs[FEATURE_INDEX["owned_province"]] == 1
    assert not info["action_mask"].any()


@pytest.mark.parametrize("agent_cards,expected,winner", [([], -1.0, 1), (["province"], 0.0, None),
                                                          (["province", "estate"], 1.0, 0)])
def test_opponent_can_end_game_with_loss_tie_or_win(agent_cards, expected, winner):
    env = constructed_env(PlayerState(hand=agent_cards), PlayerState(hand=["gold", "gold", "silver"]))
    env.game.state.supply["province"] = 1
    obs, reward, terminated, truncated, info = env.step(END_BUYING)
    assert (reward, terminated, truncated) == (expected, True, False)
    assert info["winner"] == winner and info["turns_taken"] == (1, 1)
    assert info["current_player"] == 1 and info["end_reason"] == "province_empty"
    assert obs[FEATURE_INDEX["opponent_owned_province"]] == 1


def test_terminal_reward_honors_fewer_turns_tiebreak():
    env = constructed_env()
    env.game.state.supply["province"] = 0
    _, reward, terminated, _, info = env.step(END_BUYING)
    assert terminated and info["scores"] == (0, 0)
    assert info["winner"] == 1 and reward == -1


@pytest.mark.parametrize("max_turns,current_player,turns", [(1, 0, (1, 0)), (2, 1, (1, 1))])
@pytest.mark.parametrize("mode", ["terminal_only", "potential_based"])
def test_truncation_retains_nonterminal_final_potential_and_no_winner(max_turns, current_player, turns, mode):
    env = constructed_env(PlayerState(hand=["province"]), max_turns=max_turns, mode=mode)
    obs, reward, terminated, truncated, info = env.step(END_BUYING)
    assert truncated and not terminated and info["winner"] is None
    assert info["end_reason"] == "turn_limit" and info["scores"] == (6, 0)
    assert info["current_player"] == current_player and info["turns_taken"] == turns
    assert obs[FEATURE_INDEX["agent_vp"]] == 6
    assert reward == pytest.approx(0 if mode == "terminal_only" else 0.99 * (6 / 48) - 6 / 48)
    assert not env.action_masks().any()


def test_opponent_termination_precedes_truncation_and_has_zero_terminal_potential():
    env = constructed_env(PlayerState(hand=["estate"]),
                          PlayerState(hand=["gold", "gold", "silver"]),
                          max_turns=2, mode="potential_based")
    env.game.state.supply["province"] = 1
    obs, reward, terminated, truncated, info = env.step(END_BUYING)
    assert terminated and not truncated and info["winner"] == 1
    assert obs[FEATURE_INDEX["opponent_vp"]] == 6
    assert reward == pytest.approx(-1 - (1 / 48))


def test_shaping_is_computed_once_for_entire_opponent_turn():
    env = constructed_env(PlayerState(hand=["estate"]),
                          PlayerState(hand=["gold", "gold", "silver"]), mode="potential_based")
    env.reward_config = RewardConfig(mode="potential_based", gamma=0.5, shaping_strength=2)
    env.game.state.supply["province"] = 2
    _, reward, terminated, truncated, _ = env.step(END_BUYING)
    assert not terminated and not truncated
    assert reward == pytest.approx(2 * (0.5 * (-5 / 48) - (1 / 48)))


def test_victory_purchase_can_shape_reward_without_advancing_turn():
    env = constructed_env(PlayerState(coins=2), mode="potential_based")
    obs, reward, terminated, truncated, _ = env.step(3)
    assert not terminated and not truncated and env.game.state.turn_number == 1
    assert obs[FEATURE_INDEX["agent_vp"]] == 1
    assert reward == pytest.approx(0.99 / 48)


@pytest.mark.parametrize("normal", [True, False])
def test_episode_end_requires_reset_and_cannot_mutate_state(normal):
    env = constructed_env(max_turns=1)
    if normal:
        env.game.state.supply["province"] = 0
    env.step(END_BUYING)
    before = snapshot(env)
    with pytest.raises(RuntimeError, match="ended"):
        env.step(END_BUYING)
    assert snapshot(env) == before
    env.reset(seed=42)
    assert not env.game.state.terminated and not env.game.state.truncated


def test_reset_required_before_step_masks_and_ansi_render():
    env = DominionEnv(render_mode="ansi")
    for operation in (lambda: env.step(0), env.action_masks, env.render):
        with pytest.raises(gym.error.ResetNeeded):
            operation()
    assert DominionEnv().render() is None


def test_public_info_and_render_do_not_reveal_private_opponent_zones():
    first = DominionEnv(render_mode="ansi")
    second = DominionEnv(render_mode="ansi")
    first.reset(seed=42)
    second.reset(seed=42)
    opponent = second.game.state.players[1]
    opponent.hand.extend(opponent.deck)
    opponent.deck.clear()
    opponent.hand.reverse()
    second.game.state.players[0].deck.reverse()
    output0, output1 = first.step(0), second.step(0)
    assert data_equivalence(output0, output1, exact=True)
    assert first.render() == second.render()
    assert set(output0[-1]) == {"action_mask", "turn_number", "current_player", "scores",
                              "turns_taken", "winner", "end_reason", "agent_player",
                              "episode_opponent"}
    assert output0[-1]["agent_player"] == 0
    assert output0[-1]["episode_opponent"] == "big-money"


def test_explicit_seed_replays_initialization_and_unseeded_resets_advance_stream():
    env = DominionEnv(opponent="random")
    first = env.reset(seed=42)
    initial = snapshot(env)
    env.step(END_BUYING)
    assert data_equivalence(first, env.reset(seed=42), exact=True)
    assert snapshot(env) == initial
    states = []
    rng_states = []
    for _ in range(6):
        env.reset()
        states.append(deepcopy(env.game.state))
        rng_states.append(env.game._rng.getstate())
    assert any(state != states[0] for state in states[1:])
    assert len(set(rng_states)) == 6


def test_configured_seed_initializes_master_stream_once_and_rngs_are_independent():
    first, second = DominionEnv(GameConfig(seed=17)), DominionEnv(GameConfig(seed=17))
    assert data_equivalence(first.reset(), second.reset(), exact=True)
    initial = deepcopy(first.game.state)
    first.reset()
    second.reset()
    assert first.game.state == second.game.state and first.game.state != initial
    engine_rng, gym_rng = first.game._rng.getstate(), deepcopy(first.np_random.bit_generator.state)
    for _ in range(100):
        first._opponent_rng.random()
    assert first.game._rng.getstate() == engine_rng
    assert first.np_random.bit_generator.state == gym_rng


def test_environment_does_not_consume_global_rngs():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    env = DominionEnv(opponent="random")
    env.reset(seed=42)
    for _ in range(10):
        env.step(END_BUYING)
    assert random.getstate() == python_state
    assert data_equivalence(np.random.get_state(), numpy_state, exact=True)


def play_episode(seed, opponent, mode, max_turns=200):
    env = DominionEnv(GameConfig(max_turns=max_turns), RewardConfig(mode=mode), opponent=opponent)
    history = [env.reset(seed=seed)]
    policy = random.Random(seed + 1000)
    steps = 0
    while True:
        action = policy.choice(np.flatnonzero(env.action_masks()).tolist())
        output = env.step(action)
        observation, reward, terminated, truncated, info = output
        history.append((action, output))
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert type(terminated) is bool and type(truncated) is bool
        assert info["current_player"] == 0 or terminated or truncated
        inventory = Counter(env.game.state.supply)
        for player in env.game.state.players:
            inventory.update(player.deck + player.hand + player.discard + player.in_play)
        inventory.update(env.game.state.trash)
        assert inventory == {"copper": 60, "silver": 40, "gold": 30,
                             "estate": 14, "duchy": 8, "province": 8}
        steps += 1
        assert steps <= max_turns * 2
        if terminated or truncated:
            assert terminated != truncated
            assert not env.action_masks().any()
            return history


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("opponent", ["big-money", "random"])
@pytest.mark.parametrize("mode", ["terminal_only", "potential_based"])
def test_complete_masked_random_episodes(seed, opponent, mode):
    play_episode(seed, opponent, mode)


@pytest.mark.parametrize("opponent", ["big-money", "random"])
def test_entire_seeded_episode_reproduces_opponent_decisions_observations_and_rewards(opponent):
    first = play_episode(42, opponent, "potential_based")
    second = play_episode(42, opponent, "potential_based")
    assert data_equivalence(first, second, exact=True)


@pytest.mark.filterwarnings("ignore:.*maximum value is infinity.*:UserWarning")
def test_gymnasium_checker_with_explicit_legal_sampling_adaptation():
    env = DominionEnv(render_mode="ansi")
    # Generic checker samples without masks, including before reset. End buying
    # is the only action guaranteed legal for every checked active state. Adapt
    # sampling only in this test; production validation remains strict. Purchases
    # and full mask-aware random trajectories are covered independently above.
    with patch.object(env.action_space, "sample", return_value=END_BUYING):
        check_env(env, skip_render_check=True)
    assert isinstance(env_render_passive_checker(env), str)
    env.close()
    env.close()


@pytest.mark.parametrize("kwargs", [{"opponent": "unknown"}, {"render_mode": "human"}])
def test_unsupported_constructor_choices_are_clear(kwargs):
    with pytest.raises(ValueError):
        DominionEnv(**kwargs)
