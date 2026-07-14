def calculate(expression: str) -> float:
    """Evaluate a simple arithmetic expression with +, -, *, / and parentheses.

    Supports integer literals, respects operator precedence and parentheses,
    and returns a float. Raises ValueError for malformed expressions and
    ZeroDivisionError for division by zero.
    """
    tokens: list[tuple[str, ...]] = []
    i = 0
    length = len(expression)

    # Tokenize the input string.
    while i < length:
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < length and expression[j].isdigit():
                j += 1
            tokens.append(("NUMBER", expression[i:j]))
            i = j
            continue
        if ch == "+":
            tokens.append(("PLUS",))
        elif ch == "-":
            tokens.append(("MINUS",))
        elif ch == "*":
            tokens.append(("MUL",))
        elif ch == "/":
            tokens.append(("DIV",))
        elif ch == "(":
            tokens.append(("LPAREN",))
        elif ch == ")":
            tokens.append(("RPAREN",))
        else:
            raise ValueError("Invalid character in expression")
        i += 1

    tokens.append(("EOF",))
    pos = 0

    def current():
        return tokens[pos]

    def advance():
        nonlocal pos
        pos += 1

    def parse_expression():
        value = parse_term()
        while True:
            tok = current()
            if tok[0] == "PLUS":
                advance()
                value += parse_term()
            elif tok[0] == "MINUS":
                advance()
                value -= parse_term()
            else:
                break
        return value

    def parse_term():
        value = parse_factor()
        while True:
            tok = current()
            if tok[0] == "MUL":
                advance()
                value *= parse_factor()
            elif tok[0] == "DIV":
                advance()
                # Let ZeroDivisionError propagate naturally.
                value /= parse_factor()
            else:
                break
        return value

    def parse_factor():
        tok = current()
        if tok[0] == "PLUS":
            advance()
            return parse_factor()
        if tok[0] == "MINUS":
            advance()
            return -parse_factor()
        if tok[0] == "NUMBER":
            advance()
            return float(tok[1])
        if tok[0] == "LPAREN":
            advance()
            value = parse_expression()
            if current()[0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            advance()
            return value
        raise ValueError("Unexpected token")

    result = parse_expression()

    if current()[0] != "EOF":
        raise ValueError("Unexpected trailing tokens")

    return float(result)
