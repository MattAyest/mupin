def triples_sum_to_zero(numbers: list[int]) -> bool:
    n = len(numbers)
    if n < 3:
        return False

    nums = sorted(numbers)

    for i in range(n - 2):
        # Since the array is sorted, once the first value is positive,
        # no later triple can sum to zero.
        if nums[i] > 0:
            break

        # Skip duplicate starting values to avoid redundant work.
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left = i + 1
        right = n - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                return True
            if total < 0:
                left += 1
            else:
                right -= 1

    return False
