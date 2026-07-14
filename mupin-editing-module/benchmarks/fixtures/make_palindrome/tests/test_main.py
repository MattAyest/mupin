import pytest
from hypothesis import given, settings, strategies as st
from src.main import is_palindrome, make_palindrome


def _expected_make_palindrome(s: str) -> str:
    """Reference implementation: append reverse of prefix before longest palindromic suffix."""
    n = len(s)
    for i in range(n + 1):
        suffix = s[i:]
        if suffix == suffix[::-1]:
            return s + s[:i][::-1]
    return s + s[::-1]


@pytest.mark.parametrize(
    "string, expected",
    [
        ("", True),
        ("a", True),
        ("aa", True),
        ("aba", True),
        ("abba", True),
        ("abcba", True),
        ("abc", False),
        ("ab", False),
        ("racecar", True),
        ("hello", False),
    ],
)
def test_is_palindrome_known_values(string, expected):
    assert is_palindrome(string) == expected


@given(s=st.text())
@settings(max_examples=50)
def test_is_palindrome_matches_reverse_check(s):
    assert is_palindrome(s) == (s == s[::-1])


@given(s=st.text())
@settings(max_examples=50)
def test_is_palindrome_same_for_reversed_input(s):
    assert is_palindrome(s) == is_palindrome(s[::-1])


def test_make_palindrome_empty_string():
    assert make_palindrome("") == ""


@pytest.mark.parametrize(
    "string, expected",
    [
        ("a", "a"),
        ("aa", "aa"),
        ("aba", "aba"),
        ("racecar", "racecar"),
        ("cat", "catac"),
        ("race", "racecar"),
        ("aace", "aacecaa"),
        ("abc", "abcba"),
        ("abcd", "abcdcba"),
    ],
)
def test_make_palindrome_known_values(string, expected):
    result = make_palindrome(string)
    assert isinstance(result, str)
    assert result == expected


@given(s=st.text(max_size=30))
@settings(max_examples=50)
def test_make_palindrome_matches_reference_algorithm(s):
    result = make_palindrome(s)
    expected = _expected_make_palindrome(s)
    assert result == expected


@given(s=st.text(max_size=20))
@settings(max_examples=50)
def test_make_palindrome_result_starts_with_input_and_is_palindrome(s):
    result = make_palindrome(s)
    assert result.startswith(s)
    assert is_palindrome(result)


@given(s=st.text(max_size=8))
@settings(max_examples=50)
def test_make_palindrome_is_shortest(s):
    result = make_palindrome(s)
    n = len(s)
    m = len(result)
    for length in range(n, m):
        candidate = result[:length]
        if candidate.startswith(s) and is_palindrome(candidate):
            assert False, f"found shorter palindrome starting with input: {candidate!r}"
