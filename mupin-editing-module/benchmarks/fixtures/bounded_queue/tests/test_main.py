import threading

import pytest
from hypothesis import given, settings, strategies as st

from src.main import BoundedQueue


@pytest.mark.parametrize("maxsize", [0, -1, -5, "5", 3.5, None])
def test_invalid_maxsize_raises_valueerror(maxsize):
    with pytest.raises(ValueError):
        BoundedQueue(maxsize)


@st.composite
def queue_and_items(draw):
    items = draw(st.lists(st.integers(), max_size=50))
    extra = draw(st.integers(min_value=0, max_value=50))
    maxsize = len(items) + 1 + extra
    return maxsize, items


@given(case=queue_and_items())
@settings(max_examples=50)
def test_fifo_order_and_qsize(case):
    maxsize, items = case
    q = BoundedQueue(maxsize)

    assert q.qsize() == 0

    for item in items:
        q.put(item)

    assert q.qsize() == len(items)
    assert q.qsize() <= maxsize

    out = [q.get() for _ in items]
    assert out == items
    assert q.qsize() == 0


@given(maxsize=st.integers(min_value=1, max_value=50))
@settings(max_examples=50)
def test_single_item_put_get(maxsize):
    q = BoundedQueue(maxsize)
    q.put(123)
    assert q.qsize() == 1
    assert q.get() == 123
    assert q.qsize() == 0


def test_get_blocks_until_put():
    q = BoundedQueue(1)
    result = []
    ready = threading.Event()

    def helper():
        ready.set()
        result.append(q.get())

    t = threading.Thread(target=helper)
    t.start()
    ready.wait()
    q.put(42)
    t.join(timeout=1.0)

    assert not t.is_alive()
    assert result == [42]
    assert q.qsize() == 0


def test_put_blocks_until_get():
    q = BoundedQueue(1)
    q.put("first")
    ready = threading.Event()

    def helper():
        ready.set()
        q.put("second")

    t = threading.Thread(target=helper)
    t.start()
    ready.wait()
    first = q.get()
    t.join(timeout=1.0)
    second = q.get()

    assert not t.is_alive()
    assert first == "first"
    assert second == "second"
    assert q.qsize() == 0


def test_fifo_across_threads():
    q = BoundedQueue(100)
    items = list(range(50))
    received = []

    def helper():
        for item in items:
            q.put(item)

    t = threading.Thread(target=helper)
    t.start()
    t.join(timeout=1.0)

    while q.qsize():
        received.append(q.get())

    assert not t.is_alive()
    assert received == items
