def encode_cyclic(s: str) -> str:
    parts = []
    for i in range(0, len(s), 3):
        group = s[i : i + 3]
        if len(group) == 3:
            parts.append(group[1] + group[2] + group[0])
        else:
            parts.append(group)
    return "".join(parts)


def decode_cyclic(s: str) -> str:
    parts = []
    for i in range(0, len(s), 3):
        group = s[i : i + 3]
        if len(group) == 3:
            parts.append(group[2] + group[0] + group[1])
        else:
            parts.append(group)
    return "".join(parts)
