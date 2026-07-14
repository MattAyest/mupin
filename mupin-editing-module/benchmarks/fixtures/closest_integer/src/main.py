def closest_integer(value: str) -> int:
    import decimal

    return int(decimal.Decimal(value).to_integral_value(rounding=decimal.ROUND_HALF_UP))
