def get_row(grid: list[list[int]], target: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for row_index, row in enumerate(grid):
        matching_columns = [
            col_index for col_index, value in enumerate(row) if value == target
        ]
        matching_columns.sort(reverse=True)
        for col_index in matching_columns:
            result.append((row_index, col_index))
    return result
