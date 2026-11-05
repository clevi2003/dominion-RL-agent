from copy import deepcopy
from random import Random

import pytest

from dominion_rl.agents.big_money import big_money_purchase
from dominion_rl.agents.random_agent import random_purchase


def test_random_policy_includes_end_buying_and_uses_supplied_rng():
    legal = ["copper", "silver"]
    before = legal.copy()
    first, second = Random(42), Random(42)
    choices = [random_purchase(legal, first) for _ in range(100)]
    assert choices == [random_purchase(legal, second) for _ in range(100)]
    assert set(choices) == {"copper", "silver", None}
    assert legal == before
    assert random_purchase([], first) is None


@pytest.mark.parametrize(
    "legal,provinces,expected",
    [
        (["province", "gold", "duchy", "estate"], 2, "province"),
        (["gold", "duchy", "estate", "silver"], 8, "gold"),
        (["duchy", "estate", "silver"], 8, "silver"),
        (["duchy", "estate", "silver"], 5, "silver"),
        (["gold", "duchy", "estate"], 4, "duchy"),
        (["gold", "estate"], 3, "gold"),
        (["gold", "estate"], 2, "estate"),
        (["copper", "estate"], 8, None),
        (["copper"], 2, None),
        ([], 8, None),
    ],
)
def test_big_money_priorities_and_thresholds(legal, provinces, expected):
    before = deepcopy(legal)
    assert big_money_purchase(legal, provinces) == expected
    assert legal == before
