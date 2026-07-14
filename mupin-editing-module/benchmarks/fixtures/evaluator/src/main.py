def evaluate(expression: str) -> float:
    """Evaluate an arithmetic expression with +, -, *, /, **, unary minus and parentheses."""

    s = expression
    n = len(s)
    i = 0

    # Token types
    NUMBER, PLUS, MINUS, MUL, DIV, POW, LPAREN, RPAREN, EOF = (
        "NUMBER",
        "PLUS",
        "MINUS",
        "MUL",
        "DIV",
        "POW",
        "LPAREN",
        "RPAREN",
        "EOF",
    )

    class Token:
        __slots__ = ("type", "value")

        def __init__(self, type_: str, value: float | None = None):
            self.type = type_
            self.value = value

        def __repr__(self) -> str:
            return f"Token({self.type}, {self.value})"

    tokens: list[Token] = []

    # Tokenizer
    while i < n:
        c = s[i]

        if c.isspace():
            i += 1
            continue

        if c.isdigit() or c == ".":
            start = i
            seen_digit = False

            while i < n and s[i].isdigit():
                i += 1
                seen_digit = True

            if i < n and s[i] == ".":
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
                    seen_digit = True

            if not seen_digit:
                raise ValueError("Malformed number")

            num_str = s[start:i]
            try:
                value = float(num_str)
            except ValueError as exc:
                raise ValueError("Malformed number") from exc
            tokens.append(Token(NUMBER, value))
            continue

        if c == "+":
            tokens.append(Token(PLUS))
            i += 1
            continue

        if c == "-":
            tokens.append(Token(MINUS))
            i += 1
            continue

        if c == "*":
            if i + 1 < n and s[i + 1] == "*":
                tokens.append(Token(POW))
                i += 2
            else:
                tokens.append(Token(MUL))
                i += 1
            continue

        if c == "/":
            tokens.append(Token(DIV))
            i += 1
            continue

        if c == "(":
            tokens.append(Token(LPAREN))
            i += 1
            continue

        if c == ")":
            tokens.append(Token(RPAREN))
            i += 1
            continue

        raise ValueError(f"Invalid character: {c}")

    tokens.append(Token(EOF))

    pos = 0
    current = tokens[0]

    def advance() -> None:
        nonlocal pos, current
        pos += 1
        current = tokens[pos]

    def parse_expression() -> float:
        node = parse_term()
        while current.type in (PLUS, MINUS):
            op = current.type
            advance()
            right = parse_term()
            if op == PLUS:
                node = node + right
            else:
                node = node - right
        return node

    def parse_term() -> float:
        node = parse_factor()
        while current.type in (MUL, DIV):
            op = current.type
            advance()
            right = parse_factor()
            if op == MUL:
                node = node * right
            else:
                node = node / right
        return node

    def parse_factor() -> float:
        if current.type == MINUS:
            advance()
            return -parse_factor()
        if current.type == PLUS:
            advance()
            return parse_factor()
        return parse_power()

    def parse_power() -> float:
        left = parse_atom()
        if current.type == POW:
            advance()
            right = parse_factor()
            return left**right
        return left

    def parse_atom() -> float:
        if current.type == NUMBER:
            value = current.value
            advance()
            if value is None:
                raise ValueError("Malformed expression")
            return value

        if current.type == LPAREN:
            advance()
            value = parse_expression()
            if current.type != RPAREN:
                raise ValueError("Unmatched parenthesis")
            advance()
            return value

        raise ValueError("Malformed expression")

    result = parse_expression()

    if current.type != EOF:
        raise ValueError("Malformed expression")

    return float(result)
