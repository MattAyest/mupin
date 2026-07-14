import pytest
from hypothesis import given, settings, strategies as st
from src.main import MinHeap


class ComparableItem:
    """A minimal comparable object that only defines __lt__."""

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"ComparableItem({self.value})"


@given(items=st.lists(st.integers()))
@settings(max_examples=50)
def test_pop_returns_smallest_and_removes(items):
    heap = MinHeap()
    for item in items:
        heap.push(item)

    if not items:
        with pytest.raises(IndexError):
            heap.pop()
        return

    expected_smallest = min(items)
    popped = heap.pop()
    assert popped == expected_smallest
    assert len(heap) == len(items) - 1

    remaining = items.copy()
    remaining.remove(expected_smallest)
    if remaining:
        assert heap.peek() == min(remaining)
    else:
        with pytest.raises(IndexError):
            heap.peek()


@given(items=st.lists(st.integers()))
@settings(max_examples=50)
def test_peek_returns_smallest_without_removing(items):
    heap = MinHeap()
    for item in items:
        heap.push(item)

    if not items:
        with pytest.raises(IndexError):
            heap.peek()
        return

    expected_smallest = min(items)
    peeked = heap.peek()
    assert peeked == expected_smallest
    assert len(heap) == len(items)
    assert heap.pop() == expected_smallest


@given(items=st.lists(st.integers()))
@settings(max_examples=50)
def test_repeated_pops_yield_sorted_order(items):
    heap = MinHeap()
    for item in items:
        heap.push(item)

    popped = []
    while len(heap) > 0:
        popped.append(heap.pop())

    assert popped == sorted(items)
    with pytest.raises(IndexError):
        heap.pop()
    with pytest.raises(IndexError):
        heap.peek()


@given(items=st.lists(st.integers()))
@settings(max_examples=50)
def test_len_tracks_push_and_pop(items):
    heap = MinHeap()
    assert len(heap) == 0

    for i, item in enumerate(items, start=1):
        heap.push(item)
        assert len(heap) == i

    for i in range(len(items), 0, -1):
        heap.pop()
        assert len(heap) == i - 1


@st.composite
def valid_operation_sequence(draw):
    length = draw(st.integers(min_value=0, max_value=50))
    ops = []
    count = 0
    for _ in range(length):
        if count == 0:
            op = "push"
        else:
            op = draw(st.sampled_from(["push", "pop"]))
        if op == "push":
            value = draw(st.integers())
            ops.append((op, value))
            count += 1
        else:
            ops.append((op, 0))
            count -= 1
    return ops


@given(ops=valid_operation_sequence())
@settings(max_examples=50)
def test_heap_invariant_after_mixed_operations(ops):
    heap = MinHeap()
    pushed = []

    for op, value in ops:
        if op == "push":
            heap.push(value)
            pushed.append(value)
        else:
            expected = min(pushed)
            popped = heap.pop()
            assert popped == expected
            pushed.remove(expected)

    assert len(heap) == len(pushed)
    popped_rest = []
    while len(heap) > 0:
        popped_rest.append(heap.pop())
    assert popped_rest == sorted(pushed)


@given(values=st.lists(st.integers()))
@settings(max_examples=50)
def test_heap_uses_lt_operator(values):
    heap = MinHeap()
    items = [ComparableItem(v) for v in values]
    for item in items:
        heap.push(item)

    popped_values = []
    while len(heap) > 0:
        item = heap.pop()
        popped_values.append(item.value)

    assert popped_values == sorted(values)


def test_empty_heap_raises_index_error():
    heap = MinHeap()
    assert len(heap) == 0
    with pytest.raises(IndexError):
        heap.pop()
    with pytest.raises(IndexError):
        heap.peek()
