import pytest
from collections import deque
from hypothesis import given, settings, strategies as st
from src.main import Graph


def _ref_bfs(adj, start):
    visited = {start}
    queue = deque([start])
    order = [start]
    while queue:
        u = queue.popleft()
        for v in adj.get(u, ()):
            if v not in visited:
                visited.add(v)
                queue.append(v)
                order.append(v)
    return order


def _ref_dfs(adj, start):
    visited = set()
    order = []

    def visit(u):
        visited.add(u)
        order.append(u)
        for v in adj.get(u, ()):
            if v not in visited:
                visit(v)

    visit(start)
    return order


def _ref_shortest_path(adj, start, end):
    if start == end:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        u, path = queue.popleft()
        for v in adj.get(u, ()):
            if v == end:
                return path + [v]
            if v not in visited:
                visited.add(v)
                queue.append((v, path + [v]))
    return []


def _ref_has_cycle(adj):
    nodes = set(adj.keys())
    for neighbors in adj.values():
        nodes.update(neighbors)
    indeg = {n: 0 for n in nodes}
    for u in adj:
        for v in adj[u]:
            indeg[v] += 1
    zero = [n for n in nodes if indeg[n] == 0]
    visited = 0
    while zero:
        u = zero.pop()
        visited += 1
        for v in adj.get(u, ()):
            indeg[v] -= 1
            if indeg[v] == 0:
                zero.append(v)
    return visited < len(nodes)


@st.composite
def small_directed_graph(draw):
    nodes = draw(
        st.lists(
            st.integers(min_value=-10, max_value=10),
            min_size=1,
            max_size=8,
            unique=True,
        )
    )
    edges = draw(
        st.lists(
            st.tuples(st.sampled_from(nodes), st.sampled_from(nodes)),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
    added = sorted({u for u, v in edges} | {v for u, v in edges})
    start = draw(st.sampled_from(added))
    end = draw(st.sampled_from(added))
    return edges, start, end


def _build_graph_and_adj(edges):
    g = Graph()
    adj = {}
    for u, v in edges:
        g.add_edge(u, v)
        adj.setdefault(u, []).append(v)
    return g, adj


@given(small_directed_graph())
@settings(max_examples=50)
def test_bfs_matches_reference(data):
    edges, start, _ = data
    g, adj = _build_graph_and_adj(edges)
    assert g.bfs(start) == _ref_bfs(adj, start)


@given(small_directed_graph())
@settings(max_examples=50)
def test_dfs_matches_reference(data):
    edges, start, _ = data
    g, adj = _build_graph_and_adj(edges)
    assert g.dfs(start) == _ref_dfs(adj, start)


@given(small_directed_graph())
@settings(max_examples=50)
def test_shortest_path_matches_reference(data):
    edges, start, end = data
    g, adj = _build_graph_and_adj(edges)
    assert g.shortest_path(start, end) == _ref_shortest_path(adj, start, end)


@given(small_directed_graph())
@settings(max_examples=50)
def test_has_cycle_matches_reference(data):
    edges, _, _ = data
    g, adj = _build_graph_and_adj(edges)
    assert g.has_cycle() == _ref_has_cycle(adj)


def test_dfs_and_bfs_visit_neighbors_in_insertion_order():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(2, 5)
    g.add_edge(3, 6)
    assert g.dfs(1) == [1, 2, 4, 5, 3, 6]
    assert g.bfs(1) == [1, 2, 3, 4, 5, 6]


def test_bfs_level_order_with_multiple_roots():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "e")
    assert g.bfs("a") == ["a", "b", "c", "d", "e"]


def test_shortest_path_fewest_edges_prefers_direct_edge():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 4)
    g.add_edge(1, 3)
    g.add_edge(3, 4)
    g.add_edge(1, 4)
    assert g.shortest_path(1, 4) == [1, 4]


def test_shortest_path_same_node_is_zero_length():
    g = Graph()
    g.add_edge(1, 2)
    assert g.shortest_path(1, 1) == [1]


def test_shortest_path_returns_empty_when_no_path():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(3, 4)
    assert g.shortest_path(1, 4) == []


def test_has_cycle_detects_cycle_and_self_loop():
    acyclic = Graph()
    acyclic.add_edge(1, 2)
    acyclic.add_edge(2, 3)
    acyclic.add_edge(1, 3)
    assert acyclic.has_cycle() is False

    cyclic = Graph()
    cyclic.add_edge(1, 2)
    cyclic.add_edge(2, 3)
    cyclic.add_edge(3, 1)
    assert cyclic.has_cycle() is True

    self_loop = Graph()
    self_loop.add_edge(5, 5)
    assert self_loop.has_cycle() is True


def test_empty_graph_has_no_cycle():
    g = Graph()
    assert g.has_cycle() is False


def test_missing_node_raises_keyerror():
    g = Graph()
    g.add_edge(1, 2)
    with pytest.raises(KeyError):
        g.dfs(3)
    with pytest.raises(KeyError):
        g.bfs(3)
    with pytest.raises(KeyError):
        g.shortest_path(3, 1)
    with pytest.raises(KeyError):
        g.shortest_path(1, 3)


def test_add_edge_makes_nodes_traversable():
    g = Graph()
    g.add_edge("x", "y")
    assert g.dfs("x") == ["x", "y"]
    assert g.bfs("x") == ["x", "y"]
    assert g.shortest_path("x", "y") == ["x", "y"]
