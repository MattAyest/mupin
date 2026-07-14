from hypothesis import given, settings, strategies as st
from src.main import largest_prime_digit_sum


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(n**0.5) + 1
    for divisor in range(3, limit, 2):
        if n % divisor == 0:
            return False
    return True


def _digit_sum(n: int) -> int:
    return sum(int(d) for d in str(n))


def _expected(numbers: list[int]) -> int:
    positive_primes = [n for n in numbers if n > 0 and _is_prime(n)]
    if not positive_primes:
        return 0
    return _digit_sum(max(positive_primes))


@given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=50))
@settings(max_examples=50)
def test_largest_prime_digit_sum_matches_reference(numbers):
    assert largest_prime_digit_sum(numbers) == _expected(numbers)


@given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=50))
@settings(max_examples=50)
def test_largest_prime_digit_sum_does_not_mutate_input(numbers):
    original = list(numbers)
    largest_prime_digit_sum(numbers)
    assert numbers == original


def test_empty_list_returns_zero():
    assert largest_prime_digit_sum([]) == 0


def test_no_positive_primes_returns_zero():
    assert largest_prime_digit_sum([-5, -1, 0, 1, 4, 6, 8, 9, 10]) == 0


def test_ignores_zero_and_negative_values():
    assert largest_prime_digit_sum([-10, 0, 2, 3, 11]) == 2


def test_largest_prime_digit_sum_known_values():
    assert largest_prime_digit_sum([10, 29, 31, 100]) == 4


def test_large_prime_digit_sum():
    assert largest_prime_digit_sum([999983, 997, 100]) == 47
