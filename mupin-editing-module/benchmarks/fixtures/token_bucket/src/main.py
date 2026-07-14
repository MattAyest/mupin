import time


class TokenBucket:
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        time_func=time.monotonic,
    ) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        if (
            not isinstance(refill_rate, (int, float))
            or isinstance(refill_rate, bool)
            or refill_rate <= 0
        ):
            raise ValueError("refill_rate must be positive")

        self._capacity = capacity
        self._refill_rate = float(refill_rate)
        self._time_func = time_func
        self._tokens = float(capacity)
        self._last_update = time_func()

    def consume(self, tokens: int = 1) -> bool:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise ValueError("tokens must be a positive integer")

        now = self._time_func()
        elapsed = now - self._last_update

        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_update = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        return False
