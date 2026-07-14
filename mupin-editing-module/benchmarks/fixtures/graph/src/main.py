from typing import Any


class Graph:
    def __init__(self) -> None:
        self._adj: dict[Any, list[Any]] = {}
        self._nodes: set[Any] = set()

    def add_edge(self, u: Any, v: Any) -> None:
        self._nodes.add(u)
        self._nodes.add(v)
        self._adj.setdefault(u, []).append(v)

    def dfs(self, start: Any) -> list[Any]:
        if start not in self._nodes:
            raise KeyError(start)

        visited: set[Any] = set()
        order: list[Any] = []

        def visit(u: Any) -> None:
            visited.add(u)
            order.append(u)
            for neighbor in self._adj.get(u, ()):
                if neighbor not in visited:
                    visit(neighbor)

        visit(start)
        return order

    def bfs(self, start: Any) -> list[Any]:
        if start not in self._nodes:
            raise KeyError(start)

        from collections import deque

        visited: set[Any] = {start}
        queue: deque[Any] = deque([start])
        order: list[Any] = [start]

        while queue:
            u = queue.popleft()
            for neighbor in self._adj.get(u, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    order.append(neighbor)

        return order

    def has_cycle(self) -> bool:
        if not self._nodes:
            return False

        indegree = {node: 0 for node in self._nodes}
        for u in self._adj:
            for v in self._adj[u]:
                indegree[v] += 1

        zero_indegree = [node for node in self._nodes if indegree[node] == 0]
        visited_count = 0

        while zero_indegree:
            u = zero_indegree.pop()
            visited_count += 1
            for v in self._adj.get(u, ()):
                indegree[v] -= 1
                if indegree[v] == 0:
                    zero_indegree.append(v)

        return visited_count < len(self._nodes)

    def shortest_path(self, start: Any, end: Any) -> list[Any]:
        if start not in self._nodes:
            raise KeyError(start)
        if end not in self._nodes:
            raise KeyError(end)

        if start == end:
            return [start]

        from collections import deque

        visited: set[Any] = {start}
        queue: deque[tuple[Any, list[Any]]] = deque([(start, [start])])

        while queue:
            u, path = queue.popleft()
            for neighbor in self._adj.get(u, ()):
                if neighbor == end:
                    return path + [end]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []
