import math
import pytest
from hypothesis import given, settings, strategies as st
from src.main import evaluate


def _safe_eval(expr: str) -> float:
    return float(eval(expr, {"__builtins__": {}}, {}))


_int_token = st.integers(min_value=-20, max_value=20).map(str)
_decimal_token = st.builds(
    lambda whole, frac: f"{whole}.{frac:02d}",
    st.integers(min_value=0, max_value=20),
    st.integers(min_value=0, max_value=99),
)
_nonzero_int_token = st.one_of(
    st.integers(min_value=-20, max_value=-1),
    st.integers(min_value=1, max_value=20),
).map(str)
_exponent_token = st.integers(min_value=0, max_value=3).map(str)

_expressions = st.recursive(
    st.one_of(_int_token, _decimal_token),
    lambda children: st.one_of(
        st.tuples(children, children).map(lambda ab: f"{ab[0]} + {ab[1]}"),
        st.tuples(children, children).map(lambda ab: f"{ab[0]} - {ab[1]}"),
        st.tuples(children, children).map(lambda ab: f"{ab[0]} * {ab[1]}"),
        st.tuples(children, _nonzero_int_token).map(lambda ab: f"{ab[0]} / {ab[1]}"),
        st.tuples(children, _exponent_token).map(lambda ab: f"{ab[0]} ** {ab[1]}"),
        children.map(lambda a: f"- ( {a} )"),
        children.map(lambda a: f"( {a} )"),
    ),
    max_leaves=10,
)


@given(expr=_expressions)
@settings(max_examples=50)
def test_evaluate_matches_python_eval(expr):
    raw = expr.replace(" ", "")
    expected = _safe_eval(raw)
    result = evaluate(raw)
    assert isinstance(result, float)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12)


@given(expr=_expressions)
@settings(max_examples=50)
def test_evaluate_ignores_whitespace(expr):
    raw = expr.replace(" ", "")
    spaced = expr
    assert math.isclose(evaluate(spaced), evaluate(raw), rel_tol=1e-9, abs_tol=1e-12)


def test_integer_literal():
    assert evaluate("42") == 42.0


def test_decimal_literal():
    assert math.isclose(evaluate("3.5"), 3.5)


def test_decimal_arithmetic():
    assert math.isclose(evaluate("0.1+0.2"), 0.3)


def test_addition_and_multiplication_precedence():
    assert evaluate("2+3*4") == 14.0


def test_parentheses_override_precedence():
    assert evaluate("(2+3)*4") == 20.0


def test_subtraction_and_division_precedence():
    assert evaluate("10-6/2") == 7.0


def test_left_to_right_same_precedence():
    assert evaluate("8/4*2") == 4.0
    assert evaluate("10-3-2") == 5.0


def test_exponentiation_is_right_associative():
    assert evaluate("2**3**2") == 512.0


def test_unary_minus():
    assert evaluate("-3+5") == 2.0
    assert evaluate("-(2+3)*4") == -20.0
    assert evaluate("2*-3") == -6.0
    assert evaluate("2**-3") == 0.125


def test_unary_minus_vs_exponent_precedence():
    assert evaluate("-(2)**2") == -4.0
    assert evaluate("(-2)**2") == 4.0


def test_whitespace_various():
    assert evaluate(" \t3\n+\t4 ") == 7.0
    assert evaluate("  3  + 4 * 2 / ( 1 - 5 ) ** 2 ") == evaluate("3+4*2/(1-5)**2")


def test_division_by_zero_literal():
    with pytest.raises(ZeroDivisionError):
        evaluate("1/0")


def test_division_by_zero_decimal():
    with pytest.raises(ZeroDivisionError):
        evaluate("1/0.0")


def test_division_by_zero_expression():
    with pytest.raises(ZeroDivisionError):
        evaluate("10/(5-5)")


def test_malformed_empty():
    with pytest.raises(ValueError):
        evaluate("")


def test_malformed_invalid_character():
    with pytest.raises(ValueError):
        evaluate("2 + a")


def test_malformed_double_operator():
    with pytest.raises(ValueError):
        evaluate("1 + * 2")


def test_malformed_unmatched_open_paren():
    with pytest.raises(ValueError):
        evaluate("(3+2")


def test_malformed_unmatched_close_paren():
    with pytest.raises(ValueError):
        evaluate("1+2)")


def test_malformed_bad_number():
    with pytest.raises(ValueError):
        evaluate("3..5")


def test_malformed_missing_operand():
    with pytest.raises(ValueError):
        evaluate("2 +")
