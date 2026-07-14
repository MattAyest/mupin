from hypothesis import given, settings, strategies as st
from src.main import smallest_change


def _expected_smallest_change(numbers: list[int]) -> int:
    n = len(numbers)
    changes = 0
    for i in range(n // 2):
        if numbers[i] != numbers[n - 1 - i]:
            changes += 1
    return changes


def test_empty_list_requires_zero_changes():
    assert smallest_change([]) == 0


def test_single_element_requires_zero_changes():
    assert smallest_change([42]) == 0


def test_already_palindromic_requires_zero_changes():
    assert smallest_change([1, 2, 3, 2, 1]) == 0
    assert smallest_change([7, 7, 7, 7]) == 0


def test_all_mismatched_pairs_requires_half_length_changes():
    assert smallest_change([1, 2, 3, 4]) == 2
    assert smallest_change([1, 2, 3]) == 1


def test_mixed_list_returns_correct_minimum():
    assert smallest_change([1, 2, 3, 5, 4, 7, 9, 6]) == 4
    assert smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1
    assert smallest_change([1, 2, 3, 2, 1]) == 0


@given(numbers=st.lists(st.integers(), min_size=0, max_size=100))
@settings(max_examples=50)
def test_result_equals_mismatched_symmetric_pairs(numbers):
    assert smallest_change(numbers) == _expected_smallest_change(numbers)


@given(prefix=st.lists(st.integers(), min_size=0, max_size=50))
@settings(max_examples=50)
def test_palindrome_made_by_mirroring_requires_zero_changes(prefix):
    numbers = prefix + prefix[::-1]
    assert smallest_change(numbers) == 0


@given(numbers=st.lists(st.integers(), min_size=0, max_size=100))
@settings(max_examples=50)
def test_result_is_within_valid_bounds(numbers):
    result = smallest_change(numbers)
    assert 0 <= result <= len(numbers) // 2


@given(numbers=st.lists(st.integers(), min_size=0, max_size=100))
@settings(max_examples=50)
def test_function_does_not_mutate_input(numbers):
    original = list(numbers)
    smallest_change(numbers)
    assert numbers == original
