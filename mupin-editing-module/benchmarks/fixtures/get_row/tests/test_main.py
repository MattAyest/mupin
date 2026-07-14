from hypothesis import given, settings, strategies as st
from src.main import get_row


def test_get_row_finds_all_occurrences_in_ragged_grid():
    grid = [[1, 2, 3], [2, 4, 2], [5, 2]]
    assert get_row(grid, 2) == [(0, 1), (1, 2), (1, 0), (2, 1)]


def test_get_row_columns_sorted_descending_within_each_row():
    grid = [[7, 7, 7], [7], [7, 7]]
    assert get_row(grid, 7) == [
        (0, 2),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 1),
        (2, 0),
    ]


def test_get_row_returns_empty_list_when_target_missing():
    grid = [[1, 3], [4, 5, 6], []]
    assert get_row(grid, 2) == []


def test_get_row_returns_empty_list_for_empty_grid():
    assert get_row([], 0) == []


def test_get_row_handles_single_element_grid():
    grid = [[42]]
    assert get_row(grid, 42) == [(0, 0)]
    assert get_row(grid, 7) == []


@given(
    grid=st.lists(st.lists(st.integers()), min_size=0, max_size=10),
    target=st.integers(),
)
@settings(max_examples=50)
def test_get_row_matches_full_scan_sorted(grid, target):
    expected = [
        (row, col)
        for row, line in enumerate(grid)
        for col, value in enumerate(line)
        if value == target
    ]
    expected.sort(key=lambda coord: (coord[0], -coord[1]))

    result = get_row(grid, target)

    assert result == expected
    assert all(grid[r][c] == target for r, c in result)
