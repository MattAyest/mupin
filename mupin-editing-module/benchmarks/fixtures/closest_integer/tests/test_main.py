from decimal import Decimal, ROUND_HALF_UP
from hypothesis import given, settings, strategies as st
from src.main import closest_integer


def test_halfway_positive_rounds_away_from_zero():
    assert closest_integer("14.5") == 15
    assert closest_integer("0.5") == 1
    assert closest_integer("2.500") == 3
    assert closest_integer("999.5") == 1000


def test_halfway_negative_rounds_away_from_zero():
    assert closest_integer("-14.5") == -15
    assert closest_integer("-0.5") == -1
    assert closest_integer("-2.500") == -3
    assert closest_integer("-999.5") == -1000


def test_non_halfway_standard_rounding():
    assert closest_integer("14.4") == 14
    assert closest_integer("14.6") == 15
    assert closest_integer("-14.4") == -14
    assert closest_integer("-14.6") == -15
    assert closest_integer("0.1") == 0
    assert closest_integer("-0.1") == 0
    assert closest_integer("0.9") == 1
    assert closest_integer("-0.9") == -1


def test_integer_strings_return_themselves():
    assert closest_integer("42") == 42
    assert closest_integer("-7") == -7
    assert closest_integer("0") == 0
    assert closest_integer("-0") == 0


def test_scientific_notation_rounds_correctly():
    assert closest_integer("2.5e0") == 3
    assert closest_integer("-2.5e0") == -3
    assert closest_integer("1.25e1") == 13
    assert closest_integer("-1.24e1") == -12
    assert closest_integer("5e-1") == 1
    assert closest_integer("-5e-1") == -1


@st.composite
def decimal_string(draw, half: bool = False):
    sign = draw(st.sampled_from(["", "-"]))
    whole = draw(st.integers(min_value=0, max_value=10**6))
    if half:
        trailing_zeros = draw(st.integers(min_value=0, max_value=3))
        fractional = "5" + "0" * trailing_zeros
    else:
        fractional = draw(st.text(alphabet="0123456789", min_size=1, max_size=3))
    return f"{sign}{whole}.{fractional}"


@given(value=decimal_string(half=False))
@settings(max_examples=50)
def test_arbitrary_decimal_matches_round_half_away(value):
    expected = int(Decimal(value).to_integral_value(rounding=ROUND_HALF_UP))
    result = closest_integer(value)
    assert isinstance(result, int)
    assert result == expected


@given(value=decimal_string(half=True))
@settings(max_examples=50)
def test_halfway_decimal_matches_round_half_away(value):
    expected = int(Decimal(value).to_integral_value(rounding=ROUND_HALF_UP))
    result = closest_integer(value)
    assert isinstance(result, int)
    assert result == expected
    # Behavioral property for exact half values: the result is one step
    # away from zero compared to truncating the fractional part.
    truncated = int(Decimal(value))
    assert abs(result) == abs(truncated) + 1
