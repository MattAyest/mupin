import pytest
from hypothesis import given, settings, strategies as st
from src.main import encode_cyclic, decode_cyclic


def _reference_encode(s: str) -> str:
    """Reference implementation of encode_cyclic for behavioral comparison."""
    parts = []
    for i in range(0, len(s), 3):
        group = s[i : i + 3]
        if len(group) == 3:
            parts.append(group[1] + group[2] + group[0])
        else:
            parts.append(group)
    return "".join(parts)


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", ""),
        ("a", "a"),
        ("ab", "ab"),
        ("abc", "bca"),
        ("abcd", "bcad"),
        ("abcde", "bcade"),
        ("abcdef", "bcaefd"),
        ("你好世界", "好世你界"),
    ],
)
def test_encode_cyclic_known_values(s, expected):
    assert encode_cyclic(s) == expected


@pytest.mark.parametrize(
    "encoded,original",
    [
        ("", ""),
        ("a", "a"),
        ("ab", "ab"),
        ("bca", "abc"),
        ("bcad", "abcd"),
        ("bcade", "abcde"),
        ("bcaefd", "abcdef"),
        ("好世你界", "你好世界"),
    ],
)
def test_decode_cyclic_known_values(encoded, original):
    assert decode_cyclic(encoded) == original


@given(s=st.text())
@settings(max_examples=50)
def test_encode_matches_reference_implementation(s):
    assert encode_cyclic(s) == _reference_encode(s)


@given(s=st.text())
@settings(max_examples=50)
def test_decode_inverts_encode(s):
    assert decode_cyclic(encode_cyclic(s)) == s


@given(s=st.text())
@settings(max_examples=50)
def test_encode_preserves_length_and_characters(s):
    encoded = encode_cyclic(s)
    assert len(encoded) == len(s)
    assert sorted(encoded) == sorted(s)


@given(s=st.text())
@settings(max_examples=50)
def test_decode_preserves_length_and_characters(s):
    encoded = encode_cyclic(s)
    decoded = decode_cyclic(encoded)
    assert len(decoded) == len(s)
    assert sorted(decoded) == sorted(s)
