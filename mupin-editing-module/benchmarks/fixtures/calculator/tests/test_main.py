import math

import pytest
from hypothesis import given, settings, strategies as st

from src.main import calculate


def _expected(expr: str) -> float:
    """Evaluate the expression with Python's parser in a safe scope."""
    return float(eval(expr, {"__builtins__": {}}, {}))


@st.composite
def arithmetic_expression(draw):
    """Generate valid expressions using integers, +, -, *, / and parentheses.

    The divisor of every '/' is constrained to be a non-zero integer literal so
    that the expression never divides by zero.
    """

    def build(depth):
        if depth == 0:
            n = draw(st.integers(min_value=-100, max_value=100))
            return str(n)

        op = draw(st.sampled_from(["+", "-", "*", "/"]))
        left = build(depth - 1)

        if op == "/":
            n = draw(
                st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
            )
            right = str(n)
        else:
            right = build(depth - 1)

        expr = f"({left}{op}{right})"
        if draw(st.booleans()):
            expr = f"({expr})"
        return expr

    depth = draw(st.integers(min_value=0, max_value=4))
    return build(depth)


@given(expr=arithmetic_expression())
@settings(max_examples=50)
def test_valid_expression_matches_python_eval(expr):
    result = calculate(expr)
    expected = _expected(expr)
    assert isinstance(result, float)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_returns_float_for_integer_result():
    result = calculate("2+3")
    assert isinstance(result, float)
    assert result == 5.0


def test_operator_precedence():
    assert calculate("2+3*4") == 14.0
    assert calculate("10-6/2") == 7.0
    assert calculate("8/4/2") == 1.0
    assert calculate("20-5-3") == 12.0


def test_parentheses_override_precedence():
    assert calculate("(2+3)*4") == 20.0
    assert calculate("20/(2+3)") == 4.0
    assert calculate("((2+3)*(4-1))/3") == 5.0


def test_negative_integer_literals():
    assert calculate("(-5)+3") == -2.0
    assert calculate("(-5)*(-3)") == 15.0
    assert calculate("12/(-4)") == -3.0


@pytest.mark.parametrize(
    "expr",
    [
        "",
        "1+",
        "+",
        "1+*2",
        "1+/2",
        "()",
        "(1+2",
        "1+2)",
        "1 & 2",
        "3.14+2",
        "2 ** 3",
        "abc",
        "1//2",
    ],
)
def test_invalid_expression_raises_value_error(expr):
    with pytest.raises(ValueError):
        calculate(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "1/0",
        "0/0",
        "10/(5-5)",
        "(1-1)/(2-2)",
        "100/(10-2*5)",
    ],
)
def test_division_by_zero_raises_zero_division_error(expr):
    with pytest.raises(ZeroDivisionError):
        calculate(expr)
