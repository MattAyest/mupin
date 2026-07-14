import pytest
from hypothesis import given, settings, strategies as st
from src.main import fibonacci


def test_fibonacci_zero_returns_empty():
    assert fibonacci(0) == []


def test_fibonacci_one_returns_zero():
    assert fibonacci(1) == [0]


def test_fibonacci_small_exact_values():
    assert fibonacci(2) == [0, 1]
    assert fibonacci(3) == [0, 1, 1]
    assert fibonacci(5) == [0, 1, 1, 2, 3]
    assert fibonacci(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


@given(n=st.integers(min_value=2, max_value=200))
@settings(max_examples=50)
def test_fibonacci_recurrence(n):
    result = fibonacci(n)
    assert len(result) == n
    assert result[0] == 0
    assert result[1] == 1
    for i in range(2, n):
        assert result[i] == result[i - 1] + result[i - 2]


@given(n=st.integers(min_value=-100, max_value=-1))
@settings(max_examples=50)
def test_fibonacci_negative_raises(n):
    with pytest.raises(ValueError):
        fibonacci(n)
