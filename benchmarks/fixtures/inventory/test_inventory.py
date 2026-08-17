"""Tests for the inventory helpers.

Three independent defects, in three different functions, none of which is described by the
others' failure messages. A single edit cannot satisfy all three, so reaching PASS requires
the loop to iterate on real evidence.
"""

from inventory import low_stock, restock_level, total_units


def test_total_units_sums_quantities() -> None:
    assert total_units({"bolt": 4, "nut": 6}) == 10
    assert total_units({}) == 0
    assert total_units({"washer": 3}) == 3


def test_restock_level_orders_whole_batches() -> None:
    # Needs 7 more units, ordered in batches of 5, so two batches: 10.
    assert restock_level(current=3, minimum=10, batch=5) == 10
    # Exactly one batch is enough.
    assert restock_level(current=5, minimum=10, batch=5) == 5
    # Already stocked.
    assert restock_level(current=12, minimum=10, batch=5) == 0


def test_low_stock_is_inclusive_and_sorted() -> None:
    items = {"nut": 2, "bolt": 5, "washer": 1}
    # "at or below" includes 5 itself, and the result is alphabetical.
    assert low_stock(items, threshold=5) == ["bolt", "nut", "washer"]
    assert low_stock(items, threshold=1) == ["washer"]
