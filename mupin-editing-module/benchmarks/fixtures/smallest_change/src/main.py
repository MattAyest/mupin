def smallest_change(numbers: list[int]) -> int:
    n = len(numbers)
    changes = 0
    for i in range(n // 2):
        if numbers[i] != numbers[n - 1 - i]:
            changes += 1
    return changes
