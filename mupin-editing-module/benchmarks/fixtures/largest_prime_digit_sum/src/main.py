def largest_prime_digit_sum(numbers: list[int]) -> int:
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

    largest_prime = 0
    for number in numbers:
        if number > 0 and _is_prime(number):
            if number > largest_prime:
                largest_prime = number

    if largest_prime == 0:
        return 0

    return sum(int(digit) for digit in str(largest_prime))
