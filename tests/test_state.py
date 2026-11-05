import pytest

from dominion_rl.game.cards import CARD_REGISTRY
from dominion_rl.game.state import GameState, PlayerState


def test_player_defaults_and_resources():
    player = PlayerState()
    assert player.deck == player.hand == player.discard == player.in_play == []
    assert (player.actions, player.buys, player.coins, player.turns_taken) == (1, 1, 0, 0)


def test_game_defaults_are_uninitialized_and_unfinished():
    game = GameState()
    assert len(game.players) == 2
    assert all(isinstance(player, PlayerState) for player in game.players)
    assert game.supply == {}
    assert game.trash == []
    assert (game.current_player, game.turn_number) == (0, 1)
    assert not game.terminated and not game.truncated
    assert game.end_reason is None and game.winner is None


def test_mutable_defaults_are_independent():
    first, second = GameState(), GameState()
    first.players[0].deck.append("copper")
    first.supply["copper"] = 46
    first.trash.append("estate")
    assert first.players[1].deck == []
    assert first.players[0].hand == []
    assert first.players[0].discard == []
    assert first.players[0].in_play == []
    assert second.players[0].deck == []
    assert second.supply == {} and second.trash == []


def test_state_accepts_card_ids_and_supplied_players():
    player = PlayerState(deck=["estate", "gold"], hand=["copper"])
    opponent = PlayerState()
    game = GameState(players=(player, opponent), supply={"province": 8})
    assert game.players[0] is player
    assert CARD_REGISTRY[player.deck[-1]].name == "Gold"
    assert game.supply["province"] == 8


@pytest.mark.parametrize("count", [0, 1, 3])
def test_game_requires_two_players(count):
    with pytest.raises(ValueError, match="two players"):
        GameState(players=tuple(PlayerState() for _ in range(count)))


def test_current_player_must_be_valid_index():
    with pytest.raises(ValueError, match="current_player"):
        GameState(current_player=2)
