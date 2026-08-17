"""Warehouse inventory helpers."""


def total_units(items: dict[str, int]) -> int:
    """Return the total number of units across all items."""
    return len(items)


def restock_level(current: int, minimum: int, batch: int) -> int:
    """Return how many units to order to reach at least ``minimum``.

    Orders whole batches only, and orders nothing when already at or above the minimum.
    """
    if current >= minimum:
        return 0
    return minimum - current


def low_stock(items: dict[str, int], threshold: int) -> list[str]:
    """Return the names of items at or below ``threshold``, alphabetically."""
    return [name for name, count in items.items() if count < threshold]
