def is_palindrome(string: str) -> bool:
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    if string == "":
        return ""
    for i in range(len(string)):
        suffix = string[i:]
        if is_palindrome(suffix):
            return string + string[:i][::-1]
    return string + string[::-1]
