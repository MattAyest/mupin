import itertools
import pytest
from hypothesis import given, settings, strategies as st
from src.main import rle_decode, rle_encode


@st.composite
def no_digit_text(draw):
    return draw(st.text(st.characters(exclude_characters="0123456789")))


def test_rle_empty_string():
    assert rle_encode("") == ""
    assert rle_decode("") == ""


def test_rle_encode_known_examples():
    assert rle_encode("aaabbc") == "a3b2c1"
    assert rle_encode("a") == "a1"
    assert rle_encode("aaaa") == "a4"


def test_rle_decode_known_examples():
    assert rle_decode("a3b2c1") == "aaabbc"
    assert rle_decode("a1") == "a"
    assert rle_decode("a4") == "aaaa"


@given(s=no_digit_text())
@settings(max_examples=50)
def test_rle_encode_matches_groupby(s):
    expected = "".join(
        f"{char}{len(list(group))}" for char, group in itertools.groupby(s)
    )
    assert rle_encode(s) == expected


@given(s=no_digit_text())
@settings(max_examples=50)
def test_rle_decode_inverts_encode(s):
    encoded = rle_encode(s)
    assert rle_decode(encoded) == s


@given(s=no_digit_text())
@settings(max_examples=50)
def test_rle_encode_inverts_decode_for_valid_encoding(s):
    encoded = rle_encode(s)
    assert rle_encode(rle_decode(encoded)) == encoded


@pytest.mark.parametrize(
    "invalid",
    [
        "a",
        "a0",
        "0a",
        "1",
        "a01",
        "a1b",
        "abc",
        "a10b",
        "a1b0c2",
        "a1 ",
    ],
)
def test_rle_decode_invalid_raises(invalid):
    with pytest.raises(ValueError):
        rle_decode(invalid)
