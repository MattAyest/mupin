import math
import pytest
from hypothesis import given, settings, strategies as st
from src.main import TokenBucket


class MutableClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_bucket_starts_full():
    clock = MutableClock()
    bucket = TokenBucket(10, 2.0, clock)
    assert bucket.consume(10) is True
    assert bucket.consume(1) is False


def test_consume_default_one_token():
    clock = MutableClock()
    bucket = TokenBucket(5, 1.0, clock)
    assert bucket.consume() is True
    assert bucket.consume(4) is True
    assert bucket.consume(1) is False


def test_consume_false_deducts_nothing():
    clock = MutableClock()
    bucket = TokenBucket(5, 1.0, clock)
    assert bucket.consume(3) is True
    assert bucket.consume(3) is False
    assert bucket.consume(2) is True
    assert bucket.consume(1) is False


def test_continuous_refill():
    clock = MutableClock()
    bucket = TokenBucket(10, 2.0, clock)
    assert bucket.consume(6) is True
    clock.advance(1.5)
    assert bucket.consume(7) is True
    assert bucket.consume(1) is False


def test_refill_never_exceeds_capacity():
    clock = MutableClock()
    bucket = TokenBucket(5, 1.0, clock)
    assert bucket.consume(5) is True
    clock.advance(1000.0)
    assert bucket.consume(5) is True
    clock.advance(1000.0)
    assert bucket.consume(5) is True


def test_default_time_func_starts_full():
    bucket = TokenBucket(5, 1.0)
    assert bucket.consume(5) is True
    assert bucket.consume(1) is False


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        TokenBucket(0, 1.0, MutableClock())
    with pytest.raises(ValueError):
        TokenBucket(-1, 1.0, MutableClock())
    with pytest.raises(ValueError):
        TokenBucket(3.5, 1.0, MutableClock())


def test_invalid_refill_rate_raises():
    with pytest.raises(ValueError):
        TokenBucket(10, 0.0, MutableClock())
    with pytest.raises(ValueError):
        TokenBucket(10, -1.0, MutableClock())


def test_consume_non_positive_raises():
    bucket = TokenBucket(10, 1.0, MutableClock())
    with pytest.raises(ValueError):
        bucket.consume(0)
    with pytest.raises(ValueError):
        bucket.consume(-1)


@st.composite
def valid_bucket_params(draw):
    capacity = draw(st.integers(min_value=1, max_value=1000))
    refill_rate = draw(
        st.floats(
            min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )
    tokens = draw(st.integers(min_value=1, max_value=capacity))
    return capacity, refill_rate, tokens


@given(params=valid_bucket_params())
@settings(max_examples=50)
def test_starts_full_and_exact_deduction(params):
    capacity, refill_rate, tokens = params
    clock = MutableClock()
    bucket = TokenBucket(capacity, refill_rate, clock)
    assert bucket.consume(tokens) is True
    assert bucket.consume(capacity - tokens + 1) is False


@given(params=valid_bucket_params())
@settings(max_examples=50)
def test_false_consume_does_not_deduct(params):
    capacity, refill_rate, tokens = params
    clock = MutableClock()
    bucket = TokenBucket(capacity, refill_rate, clock)
    assert bucket.consume(tokens) is True
    assert bucket.consume(capacity) is False
    clock.advance(tokens / refill_rate + 1e-9)
    assert bucket.consume(capacity) is True


@st.composite
def refill_scenario(draw):
    capacity = draw(st.integers(min_value=1, max_value=1000))
    refill_rate = draw(
        st.floats(
            min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False
        )
    )
    consumed = draw(st.integers(min_value=1, max_value=capacity))
    min_elapsed = max(0.0, (consumed - capacity + 1.001) / refill_rate)
    max_elapsed = (consumed + 1.0) / refill_rate
    elapsed = draw(
        st.floats(
            min_value=min_elapsed,
            max_value=max_elapsed,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return capacity, refill_rate, consumed, elapsed


@given(scenario=refill_scenario())
@settings(max_examples=50)
def test_refill_amount_after_elapsed(scenario):
    capacity, refill_rate, consumed, elapsed = scenario
    clock = MutableClock()
    bucket = TokenBucket(capacity, refill_rate, clock)
    assert bucket.consume(consumed) is True
    clock.advance(elapsed)
    expected = min(capacity, capacity - consumed + refill_rate * elapsed)
    k = math.floor(expected)
    assert k >= 1
    assert k <= capacity
    assert bucket.consume(k) is True
    assert bucket.consume(1) is False
