from dataclasses import FrozenInstanceError

import pytest

from dominion_rl.game.cards import (
    ActionCard, CARD_REGISTRY, Card, Copper, Duchy, Estate, Gold, Province,
    Silver, TreasureCard, VictoryCard,
)
from dominion_rl.game.state import PlayerState


@pytest.mark.parametrize(
    "card_id,card_class,name,cost,value,category",
    [
        ("copper", Copper, "Copper", 0, 1, TreasureCard),
        ("silver", Silver, "Silver", 3, 2, TreasureCard),
        ("gold", Gold, "Gold", 6, 3, TreasureCard),
        ("estate", Estate, "Estate", 2, 1, VictoryCard),
        ("duchy", Duchy, "Duchy", 5, 3, VictoryCard),
        ("province", Province, "Province", 8, 6, VictoryCard),
    ],
)
def test_basic_definitions(card_id, card_class, name, cost, value, category):
    card = CARD_REGISTRY[card_id]
    assert isinstance(card, Card)
    assert type(card) is card_class
    assert isinstance(card, category)
    assert (card.name, card.cost) == (name, cost)
    if isinstance(card, TreasureCard):
        assert card.coin_value == value
        assert card.card_types == frozenset({"treasure"})
    else:
        assert card.victory_points == value
        assert card.card_types == frozenset({"victory"})
    with pytest.raises(FrozenInstanceError):
        card.cost = 999
    with pytest.raises(TypeError):
        card_class(cost=999)


def test_registry_ids_and_shared_definitions():
    assert tuple(CARD_REGISTRY) == (
        "copper", "silver", "gold", "estate", "duchy", "province",
    )
    hand = ["copper", "copper"]
    assert CARD_REGISTRY[hand[0]] is CARD_REGISTRY[hand[1]]
    with pytest.raises(KeyError):
        CARD_REGISTRY["unknown"]


@pytest.mark.parametrize("card_id", ["copper", "silver", "gold"])
def test_treasure_coin_effect(card_id):
    player = PlayerState(hand=[card_id], coins=2)
    card = CARD_REGISTRY[card_id]
    card.play(None, player)  # Treasure effects do not need engine helpers.
    assert player.coins == 2 + card.coin_value
    assert player.hand == [card_id]  # Zone movement is an engine responsibility.
    assert player.buys == 1


def test_action_base_supports_concrete_effect_without_engine_dispatch():
    class ExampleAction(ActionCard):
        def play(self, game, player):
            player.actions += 2

        def react(self):
            return True

    card = ExampleAction(name="Example", cost=3)
    player = PlayerState()
    card.play(None, player)
    assert player.actions == 3
    assert card.react()
    assert isinstance(card, Card)
    assert card.card_types == frozenset({"action"})
