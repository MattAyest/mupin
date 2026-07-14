import pytest
from hypothesis import given, settings, strategies as st
from src.main import Stack


def test_new_stack_is_empty_and_size_zero():
    s = Stack()
    assert s.is_empty() is True
    assert s.size() == 0


def test_pop_empty_raises_indexerror():
    s = Stack()
    with pytest.raises(IndexError):
        s.pop()


def test_peek_empty_raises_indexerror():
    s = Stack()
    with pytest.raises(IndexError):
        s.peek()


@given(items=st.lists(st.integers()))
@settings(max_examples=50)
def test_push_then_pop_is_lifo(items):
    s = Stack()
    for item in items:
        s.push(item)
    assert s.size() == len(items)

    popped = []
    while not s.is_empty():
        popped.append(s.pop())
    assert popped == list(reversed(items))


@given(items=st.lists(st.integers(), min_size=1))
@settings(max_examples=50)
def test_peek_returns_last_pushed_and_preserves_stack(items):
    s = Stack()
    for item in items:
        s.push(item)
    assert s.peek() == items[-1]
    assert s.size() == len(items)
    assert s.pop() == items[-1]


@given(items=st.lists(st.integers()))
@settings(max_examples=50)
def test_size_and_empty_track_pushes_and_pops(items):
    s = Stack()
    assert s.size() == 0
    assert s.is_empty() is True

    for i, item in enumerate(items, start=1):
        s.push(item)
        assert s.size() == i
        assert s.is_empty() is False

    for remaining in range(len(items) - 1, -1, -1):
        s.pop()
        assert s.size() == remaining
        assert s.is_empty() == (remaining == 0)


@st.composite
def push_pop_sequence(draw):
    length = draw(st.integers(min_value=0, max_value=50))
    commands = []
    count = 0
    for _ in range(length):
        if count == 0:
            op = "push"
        else:
            op = draw(st.sampled_from(["push", "pop", "peek"]))
        if op == "push":
            value = draw(st.integers())
            commands.append((op, value))
            count += 1
        else:
            commands.append((op,))
            if op == "pop":
                count -= 1
    return commands


@given(commands=push_pop_sequence())
@settings(max_examples=50)
def test_random_operations_match_python_list(commands):
    s = Stack()
    model = []
    for cmd in commands:
        if cmd[0] == "push":
            value = cmd[1]
            s.push(value)
            model.append(value)
        elif cmd[0] == "pop":
            assert s.pop() == model.pop()
        else:
            assert s.peek() == model[-1]
        assert s.size() == len(model)
        assert s.is_empty() == (len(model) == 0)
