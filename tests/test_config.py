from pathlib import Path

import pytest
import yaml

from dominion_rl.config import (
    GameConfig, RewardConfig, load_game_config, load_reward_config,
)
from dominion_rl.game.cards import V0_CARD_IDS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_checked_in_game_config():
    config = load_game_config(CONFIGS / "game.yaml")
    assert config == GameConfig()
    assert config.cards == V0_CARD_IDS
    assert (config.num_players, config.seed, config.max_turns) == (2, 0, 200)


def test_checked_in_reward_config():
    config = load_reward_config(str(CONFIGS / "rewards.yaml"))
    assert config == RewardConfig()
    assert config.mode == "terminal_only"
    assert (config.gamma, config.shaping_strength, config.vp_normalization) == (0.99, 1, 48)
    assert (config.win_reward, config.tie_reward, config.loss_reward) == (1, 0, -1)


def test_configuration_overrides(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("seed: null\nmax_turns: 80\n", encoding="utf-8")
    game = load_game_config(path)
    assert game.seed is None and game.max_turns == 80
    path.write_text("mode: potential_based\nshaping_strength: 0.5\n", encoding="utf-8")
    reward = load_reward_config(path)
    assert reward.mode == "potential_based" and reward.shaping_strength == 0.5


@pytest.mark.parametrize(
    "content",
    [
        "version: V99", "num_players: 3", "max_turns: 0", "max_turns: true",
        "seed: -1", "seed: true", "cards: copper", "cards: [copper, gold]",
        "cards: [copper, silver, gold, estate, duchy, unknown]",
        "cards: [copper, silver, gold, estate, duchy, duchy]", "cards: [null]",
    ],
)
def test_invalid_game_settings(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_game_config(path)


@pytest.mark.parametrize(
    "content",
    [
        "mode: unknown", "gamma: 1.1", "shaping_strength: -1",
        "vp_normalization: 0", "gamma: .nan", "win_reward: .inf", "gamma: true",
    ],
)
def test_invalid_reward_settings(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_reward_config(path)


@pytest.mark.parametrize("loader", [load_game_config, load_reward_config])
@pytest.mark.parametrize("content", ["", "[]", "hello", "1: value"])
def test_yaml_must_be_mapping(tmp_path, loader, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        loader(path)


@pytest.mark.parametrize("loader", [load_game_config, load_reward_config])
def test_unknown_settings_fail_instead_of_being_ignored(tmp_path, loader):
    path = tmp_path / "typo.yaml"
    path.write_text("typo: 5", encoding="utf-8")
    with pytest.raises(TypeError, match="typo"):
        loader(path)


def test_yaml_uses_safe_loader(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object:builtins.object {}", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_game_config(path)


def test_missing_config_reports_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_game_config(tmp_path / "missing.yaml")
