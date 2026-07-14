def rle_encode(data: str) -> str:
    if not data:
        return ""
    result = []
    current = data[0]
    count = 1
    for ch in data[1:]:
        if ch == current:
            count += 1
        else:
            result.append(f"{current}{count}")
            current = ch
            count = 1
    result.append(f"{current}{count}")
    return "".join(result)


def rle_decode(data: str) -> str:
    if not data:
        return ""

    def _is_digit(c: str) -> bool:
        return "0" <= c <= "9"

    n = len(data)
    i = 0
    out = []
    while i < n:
        char = data[i]
        if _is_digit(char):
            raise ValueError("Invalid encoding: expected character, got digit")
        j = i + 1
        if j >= n or not _is_digit(data[j]):
            raise ValueError("Invalid encoding: missing run length")
        if data[j] == "0":
            raise ValueError("Invalid encoding: run length has leading zero")
        while j < n and _is_digit(data[j]):
            j += 1
        count = int(data[i + 1 : j])
        out.append(char * count)
        i = j
    return "".join(out)
