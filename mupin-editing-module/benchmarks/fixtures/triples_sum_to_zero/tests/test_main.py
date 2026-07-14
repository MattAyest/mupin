import pytest
from hypothesis import given, settings, strategies as st
from src.main import triples_sum_to_zero


def _brute_force_has_zero_triple(numbers: list[int]) -> bool:
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if numbers[i] + numbers[j] + numbers[k] == 0:
                    return True
    return False


@pytest.mark.parametrize(
    "numbers, expected",
    [
        ([1, -1, 0], True),
        ([2, -1, -1], True),
        ([0, 0, 0], True),
        ([1, 2, 3], False),
        ([1, 1, -2], True),
        ([10, 20, 30, -5], False),
        ([0, 0, 5], False),
        ([1] * 100 + [-2], True),
    ],
)
def test_known_cases(numbers, expected):
    before = list(numbers)
    result = triples_sum_to_zero(numbers)
    assert result is expected
    assert numbers == before


@given(st.lists(st.integers(), max_size=2))
@settings(max_examples=50)
def test_short_lists_false(numbers):
    before = list(numbers)
    assert triples_sum_to_zero(numbers) is False
    assert numbers == before


@given(
    st.lists(
        st.integers(min_value=1, max_value=1000),
        min_size=3,
        max_size=50,
    )
)
@settings(max_examples=50)
def test_all_positive_false(numbers):
    before = list(numbers)
    assert triples_sum_to_zero(numbers) is False
    assert numbers == before


@given(
    st.lists(
        st.integers(min_value=-1000, max_value=-1),
        min_size=3,
        max_size=50,
    )
)
@settings(max_examples=50)
def test_all_negative_false(numbers):
    before = list(numbers)
    assert triples_sum_to_zero(numbers) is False
    assert numbers == before


@given(
    st.lists(
        st.integers(min_value=-100, max_value=100),
        min_size=0,
        max_size=12,
    )
)
@settings(max_examples=50)
def test_agrees_with_brute_force(numbers):
    expected = _brute_force_has_zero_triple(numbers)
    before = list(numbers)
    assert triples_sum_to_zero(numbers) is expected
    assert numbers == before


def test_moderate_size_all_positive_false():
    numbers = list(range(1, 501))
    assert triples_sum_to_zero(numbers) is False


def test_moderate_size_with_duplicates_true():
    numbers = [1] * 499 + [-2]
    assert triples_sum_to_zero(numbers) is True
