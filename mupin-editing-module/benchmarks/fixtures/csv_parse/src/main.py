def parse_csv(text: str) -> list[dict]:
    def _is_int(value: str) -> bool:
        if not value:
            return False
        if value[0] in "+-":
            num = value[1:]
        else:
            num = value
        return num != "" and num.isdigit()

    def _is_float(value: str) -> bool:
        if not value:
            return False
        if value[0] in "+-":
            num = value[1:]
        else:
            num = value
        parts = num.split(".")
        if len(parts) != 2:
            return False
        left, right = parts
        return left.isdigit() and right.isdigit()

    def _infer(value: str):
        if _is_int(value):
            return int(value)
        if _is_float(value):
            return float(value)
        return value

    lines = [line for line in text.splitlines() if line.strip() != ""]
    if not lines:
        return []

    header = lines[0].split(",")
    expected_count = len(header)

    result: list[dict] = []
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) != expected_count:
            raise ValueError("Field count mismatch")
        result.append({key: _infer(value) for key, value in zip(header, fields)})

    return result
