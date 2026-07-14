def sort_array(numbers: list[int]) -> list[int]:
    if len(numbers) <= 1:
        return list(numbers)

    sorted_numbers = sorted(numbers)
    if (numbers[0] + numbers[-1]) % 2 == 0:
        sorted_numbers.reverse()

    return sorted_numbers
