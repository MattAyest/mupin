from typing import Any


class MinHeap:
    def __init__(self) -> None:
        self._heap: list[Any] = []

    def push(self, item: Any) -> None:
        self._heap.append(item)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> Any:
        if not self._heap:
            raise IndexError("pop from empty heap")

        smallest = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return smallest

    def peek(self) -> Any:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def __len__(self) -> int:
        return len(self._heap)

    def _sift_up(self, index: int) -> None:
        item = self._heap[index]
        while index > 0:
            parent_index = (index - 1) // 2
            parent = self._heap[parent_index]
            if item < parent:
                self._heap[index] = parent
                index = parent_index
            else:
                break
        self._heap[index] = item

    def _sift_down(self, index: int) -> None:
        n = len(self._heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index

            if left < n and self._heap[left] < self._heap[smallest]:
                smallest = left
            if right < n and self._heap[right] < self._heap[smallest]:
                smallest = right

            if smallest == index:
                break

            self._heap[index], self._heap[smallest] = (
                self._heap[smallest],
                self._heap[index],
            )
            index = smallest
