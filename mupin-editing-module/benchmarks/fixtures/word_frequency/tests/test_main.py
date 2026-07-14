import string

from hypothesis import given, settings, strategies as st

from src.main import word_frequency


def test_word_frequency_basic():
    text = "Hello, world! Hello..."
    result = word_frequency(text)
    assert result == {"hello": 2, "world": 1}


def test_word_frequency_case_insensitive():
    text = "Hello HELLO hello"
    result = word_frequency(text)
    assert result == {"hello": 3}


def test_word_frequency_empty_and_whitespace():
    assert word_frequency("") == {}
    assert word_frequency("   ") == {}
    assert word_frequency("\t\n\r ") == {}


def test_word_frequency_punctuation_only():
    assert word_frequency("... , ; : ! ?") == {}


def test_word_frequency_does_not_mutate_input():
    text = "Hello, hello!"
    original = text
    word_frequency(text)
    assert text == original


@st.composite
def generated_word_text(draw):
    words = draw(
        st.lists(
            st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10),
            min_size=1,
            max_size=20,
        )
    )

    expected = {}
    tokens = []

    for word in words:
        count = draw(st.integers(min_value=1, max_value=5))
        prefix = draw(st.text(alphabet=string.punctuation, min_size=0, max_size=3))
        suffix = draw(st.text(alphabet=string.punctuation, min_size=0, max_size=3))
        raw_token = prefix + word + suffix
        cased_token = "".join(
            draw(st.sampled_from([str.lower, str.upper]))(ch) for ch in raw_token
        )
        tokens.extend([cased_token] * count)
        expected[word] = expected.get(word, 0) + count

    shuffled = draw(st.permutations(tokens))
    separator = draw(st.text(alphabet=" \t\n", min_size=1, max_size=3))
    text = separator.join(shuffled)
    return text, expected


@given(data=generated_word_text())
@settings(max_examples=50)
def test_word_frequency_matches_expected_counts(data):
    text, expected = data
    result = word_frequency(text)
    assert result == expected


@given(text=st.text(alphabet=" \t\n\r", min_size=0))
@settings(max_examples=50)
def test_word_frequency_whitespace_only_returns_empty(text):
    result = word_frequency(text)
    assert result == {}
