import collections
import threading
from typing import Any


class BoundedQueue:
    def __init__(self, maxsize: int) -> None:
        if type(maxsize) is not int or maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self._maxsize = maxsize
        self._queue: collections.deque[Any] = collections.deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def put(self, item: Any) -> None:
        with self._not_full:
            while len(self._queue) >= self._maxsize:
                self._not_full.wait()
            self._queue.append(item)
            self._not_empty.notify()

    def get(self) -> Any:
        with self._not_empty:
            while len(self._queue) == 0:
                self._not_empty.wait()
            item = self._queue.popleft()
            self._not_full.notify()
            return item

    def qsize(self) -> int:
        with self._lock:
            return len(self._queue)
