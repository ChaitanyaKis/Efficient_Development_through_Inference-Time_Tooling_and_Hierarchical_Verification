"""Tests for the calculator library."""

from calculator import add, multiply, subtract


def test_add() -> None:
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract() -> None:
    assert subtract(5, 3) == 2
    assert subtract(0, 4) == -4
    assert subtract(10, 10) == 0


def test_multiply() -> None:
    assert multiply(3, 4) == 12
    assert multiply(0, 99) == 0
