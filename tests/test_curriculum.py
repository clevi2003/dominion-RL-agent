"""Curriculum boundaries, reproducibility and Monitor/auto-reset integration."""
import csv

import numpy as np
import pytest
import yaml

from dominion_rl.config import GameConfig, RewardConfig
from dominion_rl.env import DominionEnv
from dominion_rl.env.dominion_env import DEFAULT_CURRICULUM, END_BUYING, curriculum_stage, validate_curriculum
import dominion_rl.env.dominion_env as env_module


@pytest.mark.parametrize('steps,expected', [(0,0), (51199,0), (51200,1), (127999,1),
                                            (128000,2), (256000,2), (300000,2)])
def test_fractional_schedule_boundaries(steps, expected):
    assert curriculum_stage(steps, 256000, DEFAULT_CURRICULUM) == expected
    assert curriculum_stage(steps, 512000, DEFAULT_CURRICULUM) == curriculum_stage(steps//2, 256000, DEFAULT_CURRICULUM)


@pytest.mark.parametrize('stages', [[], [{'start_fraction':.2, 'random_probability':1}],
    [{'start_fraction':0, 'random_probability':1.01}], [{'start_fraction':0, 'random_probability':float('nan')}],
    [{'start_fraction':0, 'random_probability':1}, {'start_fraction':0, 'random_probability':.5}],
    [{'start_fraction':0, 'random_probability':1, 'big_money_probability':0}]])
def test_invalid_schedules_fail(stages):
    with pytest.raises(ValueError):
        validate_curriculum(stages)


@pytest.mark.parametrize('budget', [None, 0, -1, True, 10.5])
def test_curriculum_requires_explicit_positive_training_budget(budget):
    with pytest.raises(ValueError, match='curriculum_total_timesteps'):
        DominionEnv(opponent='curriculum', curriculum_total_timesteps=budget)


def test_fixed_mode_rejects_accidental_curriculum_settings():
    with pytest.raises(ValueError, match='require'):
        DominionEnv(opponent='random', curriculum_total_timesteps=100)


def test_stage_changes_at_auto_reset_after_counting_terminal_decision(monkeypatch):
    pytest.importorskip('stable_baselines3')
    from stable_baselines3.common.vec_env import DummyVecEnv
    monkeypatch.setattr(env_module, 'random_purchase', lambda legal, rng: None)
    monkeypatch.setattr(env_module, 'big_money_purchase', lambda legal, provinces: None)
    env = DominionEnv(GameConfig(max_turns=4), opponent='curriculum', curriculum_total_timesteps=10)
    vec = DummyVecEnv([lambda: env])
    vec.seed(42)
    try:
        vec.reset()
        starts = [(env.episode_start_step, env.curriculum_stage_index)]
        for step in range(10):
            previous = env.episode_opponent
            _, _, dones, infos = vec.step([END_BUYING])
            assert infos[0]['episode_opponent'] == previous
            assert infos[0]['TimeLimit.truncated'] == bool(dones[0])
            if dones[0]:
                starts.append((env.episode_start_step, env.curriculum_stage_index))
        assert starts == [(0,0), (2,1), (4,1), (6,2), (8,2), (10,2)]
        assert sum(env.opponent_step_counts.values()) == 10
        assert [sum(x.values()) for x in env.curriculum_step_counts] == [2,4,4]
    finally:
        vec.close()


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('mode', ['terminal_only', 'potential_based'])
def test_curriculum_seeded_trace_and_raw_observation_boundaries(seat, mode):
    a, b = [DominionEnv(GameConfig(max_turns=12), RewardConfig(mode=mode), opponent='curriculum',
                       curriculum_total_timesteps=60, agent_player=seat) for _ in range(2)]
    try:
        for episode in range(16):
            seed = 43 if episode == 0 else None
            obs, info = a.reset(seed=seed)
            other, repeated = b.reset(seed=seed)
            np.testing.assert_array_equal(obs, other)
            assert obs.shape == (41,) and obs.dtype == np.float32
            assert info['episode_opponent'] == repeated['episode_opponent']
            # Opponent/stage metadata changes no policy features or engine RNG.
            fixed = DominionEnv(GameConfig(max_turns=12), RewardConfig(mode=mode), opponent=info['episode_opponent'], agent_player=seat)
            if episode == 0:
                np.testing.assert_array_equal(obs, fixed.reset(seed=43)[0])
            fixed.close()
            selected, stage = info['episode_opponent'], info['curriculum_stage']
            while True:
                np.testing.assert_array_equal(a.action_masks(), b.action_masks())
                left, right = a.step(END_BUYING), b.step(END_BUYING)
                np.testing.assert_array_equal(left[0], right[0])
                assert left[1:4] == right[1:4]
                assert left[4]['episode_opponent'] == selected
                assert left[4]['curriculum_stage'] == stage
                if left[2] or left[3]:
                    break
        assert a.opponent_step_counts == b.opponent_step_counts
        assert a.curriculum_step_counts == b.curriculum_step_counts
        before = sum(a.opponent_step_counts.values())
        a.reset(seed=43)
        assert a.episode_start_step == before  # Reseeding is not resuming/restarting training.
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize('mode', ['terminal_only', 'potential_based'])
def test_short_training_records_stage_exposure_including_partial_episode(tmp_path, mode):
    pytest.importorskip('sb3_contrib')
    from dominion_rl.ppo import train_ppo
    model = train_ppo(tmp_path/'model', game_config=GameConfig(max_turns=8), reward_config=RewardConfig(mode=mode),
        ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1, gamma=.99, policy_kwargs={'net_arch':[16,16]}),
        seed=42, total_timesteps=32, checkpoint_freq=16, opponent='curriculum')
    m = yaml.safe_load((model.parent/'run.yaml').read_text())
    assert m['curriculum'] == list(DEFAULT_CURRICULUM)
    assert m['opponent_distribution'] is None
    assert sum(x['decision_steps'] for x in m['by_opponent'].values()) == 32
    with (model.parent/'episodes.monitor.csv').open() as stream:
        next(stream)
        rows = list(csv.DictReader(stream))
    start = 0
    for row in rows:
        assert int(row['episode_start_step']) == start
        assert int(row['curriculum_stage']) == curriculum_stage(start, 32, DEFAULT_CURRICULUM)
        start += int(row['l'])
    assert start+m['partial_episode_steps'] == 32
    for name in ('random', 'big-money'):
        for key in ('decision_steps', 'started_episodes', 'completed_episodes', 'completed_episode_decision_steps'):
            assert sum(s['by_opponent'][name][key] for s in m['by_curriculum_stage']) == m['by_opponent'][name][key]


def test_random_warmup_preserves_paired_control_game_stream():
    control = DominionEnv(GameConfig(max_turns=8), opponent='random')
    curriculum = DominionEnv(GameConfig(max_turns=8), opponent='curriculum', curriculum_total_timesteps=10000)
    try:
        for episode in range(10):
            seed = 42 if episode == 0 else None
            np.testing.assert_array_equal(control.reset(seed=seed)[0], curriculum.reset(seed=seed)[0])
            assert curriculum.curriculum_stage_index == 0
            while True:
                a, b = control.step(END_BUYING), curriculum.step(END_BUYING)
                np.testing.assert_array_equal(a[0], b[0])
                assert a[1:4] == b[1:4]
                if a[2] or a[3]:
                    break
    finally:
        control.close()
        curriculum.close()
