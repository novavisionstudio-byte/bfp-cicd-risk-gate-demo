import pytest
from app import calculator as calc


def test_add():
    assert calc.add(2, 3) == 5


def test_subtract():
    assert calc.subtract(5, 3) == 2


def test_multiply():
    assert calc.multiply(4, 3) == 12


def test_divide():
    assert calc.divide(10, 2) == 5


def test_divide_by_zero_raises():
    with pytest.raises(ValueError):
        calc.divide(1, 0)


def test_percentage():
    assert calc.percentage(25, 200) == 12.5


def test_factorial():
    assert calc.factorial(0) == 1
    assert calc.factorial(5) == 120


def test_factorial_negative_raises():
    with pytest.raises(ValueError):
        calc.factorial(-1)
