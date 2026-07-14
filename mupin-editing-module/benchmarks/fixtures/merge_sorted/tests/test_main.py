from hypothesis import given, settings, strategies as st
from src.main import merge_sorted


@st.composite
def sorted_int_list(draw):
    return sorted(draw(st.lists(st.integers())))


def test_merge_sorted_basic():
    result = merge_sorted([1, 3, 5], [2, 4, 6])
    assert result == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_empty_inputs():
    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]
    assert merge_sorted([1, 2, 3], []) == [1, 2, 3]
    assert merge_sorted([], []) == []


def test_merge_sorted_duplicates():
    result = merge_sorted([1, 2, 2, 3], [2, 2, 4])
    assert result == [1, 2, 2, 2, 2, 3, 4]


def test_merge_sorted_negatives_and_mixed():
    result = merge_sorted([-5, -1, 0], [-3, 2, 4])
    assert result == [-5, -3, -1, 0, 2, 4]


def test_merge_sorted_single_elements():
    assert merge_sorted([1], [2]) == [1, 2]
    assert merge_sorted([2], [1]) == [1, 2]


@given(a=sorted_int_list(), b=sorted_int_list())
@settings(max_examples=50)
def test_merge_sorted_is_sorted_complete(a, b):
    result = merge_sorted(a, b)
    assert len(result) == len(a) + len(b)
    assert result == sorted(result)
    assert sorted(result) == sorted(a + b)
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]


@given(a=sorted_int_list(), b=sorted_int_list())
@settings(max_examples=50)
def test_merge_sorted_preserves_inputs(a, b):
    a_before = list(a)
    b_before = list(b)
    merge_sorted(a, b)
    assert a == a_before
    assert b == b_before


@given(a=sorted_int_list(), b=sorted_int_list())
@settings(max_examples=50)
def test_merge_sorted_returns_new_list(a, b):
    result = merge_sorted(a, b)
    assert result is not a
    assert result is not b
