import string


def word_frequency(text: str) -> dict[str, int]:
    """Return a case-insensitive word frequency count with punctuation removed."""
    translator = str.maketrans("", "", string.punctuation)
    cleaned = text.lower().translate(translator)
    words = cleaned.split()

    frequency: dict[str, int] = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    return frequency
