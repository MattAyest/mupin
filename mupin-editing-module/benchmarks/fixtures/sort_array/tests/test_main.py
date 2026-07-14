from hypothesis import given, settings, strategies as st
from src.main import sort_array


def test_sort_array_odd_sum_ascending():
    numbers = [3, 1, 2]
    result = sort_array(numbers)
    assert result == [1, 2, 3]
    assert numbers == [3, 1, 2]
    assert result is not numbers


def test_sort_array_even_sum_descending():
    numbers = [4, 1, 2]
    result = sort_array(numbers)
    assert result == [4, 2, 1]
    assert numbers == [4, 1, 2]
    assert result is not numbers


def test_sort_array_single_element_unchanged():
    numbers = [7]
    result = sort_array(numbers)
    assert result == [7]
    assert numbers == [7]


def test_sort_array_empty_unchanged():
    numbers = []
    result = sort_array(numbers)
    assert result == []
    assert numbers == []


@given(numbers=st.lists(st.integers(), min_size=2))
@settings(max_examples=50)
def test_sort_array_parity_determines_direction(numbers):
    original = list(numbers)
    result = sort_array(numbers)

    if (numbers[0] + numbers[-1]) % 2 == 1:
        assert result == sorted(numbers)
    else:
        assert result == sorted(numbers, reverse=True)

    assert numbers == original
    assert result is not numbers


@given(numbers=st.lists(st.integers(), max_size=1))
@settings(max_examples=50)
def test_sort_array_trivial_lists_unchanged(numbers):
    original = list(numbers)
    result = sort_array(numbers)
    assert result == numbers
    assert numbers == original
