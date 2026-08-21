from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass(frozen=True)
class GraphMetrics:
    n_nodes: int
    n_edges: int
    density: float
    mean_degree: float
    max_degree: int
    connected_components: int
    diameter: float
    mean_shortest_path: float
    algebraic_connectivity: float


def normalize_edges(n_nodes, edges):
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    normalized = set()
    for u, v in edges:
        u, v = int(u), int(v)
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError("edge endpoint out of range")
        if u == v:
            raise ValueError("self-loops are not supported")
        normalized.add(tuple(sorted((u, v))))
    return tuple(sorted(normalized))


def topology_edges(n_nodes, topology):
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    name = str(topology).lower()
    if n_nodes == 1:
        return ()
    if name == "none":
        return ()
    if name == "line":
        return tuple((i, i + 1) for i in range(n_nodes - 1))
    if name == "ring":
        if n_nodes == 2:
            return ((0, 1),)
        return tuple((i, i + 1) for i in range(n_nodes - 1)) + ((0, n_nodes - 1),)
    if name == "star":
        return tuple((0, i) for i in range(1, n_nodes))
    if name in {"complete", "all_to_all"}:
        return tuple((i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes))
    raise ValueError("unknown topology")


def adjacency_matrix(n_nodes, edges):
    normalized = normalize_edges(n_nodes, edges)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in normalized:
        adjacency[u, v] = 1.0
        adjacency[v, u] = 1.0
    return adjacency


def _all_pair_distances(adjacency):
    n = adjacency.shape[0]
    distances = np.full((n, n), np.inf, dtype=float)
    for source in range(n):
        distances[source, source] = 0.0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in np.flatnonzero(adjacency[u]):
                if np.isinf(distances[source, v]):
                    distances[source, v] = distances[source, u] + 1.0
                    queue.append(int(v))
    return distances


def graph_metrics(n_nodes, edges):
    adjacency = adjacency_matrix(n_nodes, edges)
    n_edges = int(np.sum(adjacency) // 2)
    degrees = np.sum(adjacency, axis=1)
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = 0.0 if max_edges == 0 else n_edges / max_edges

    distances = _all_pair_distances(adjacency)
    reachability = np.isfinite(distances)
    unseen = set(range(n_nodes))
    components = 0
    while unseen:
        start = next(iter(unseen))
        component = set(np.flatnonzero(reachability[start]).tolist())
        unseen -= component
        components += 1

    upper = distances[np.triu_indices(n_nodes, k=1)]
    finite_upper = upper[np.isfinite(upper)]
    if finite_upper.size == upper.size:
        diameter = float(np.max(finite_upper)) if finite_upper.size else 0.0
        mean_shortest = float(np.mean(finite_upper)) if finite_upper.size else 0.0
    else:
        diameter = float("inf")
        mean_shortest = float("inf")

    laplacian = np.diag(degrees) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    algebraic = float(eigenvalues[1]) if n_nodes > 1 else 0.0
    if abs(algebraic) < 1e-12:
        algebraic = 0.0

    return GraphMetrics(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=float(density),
        mean_degree=float(np.mean(degrees)),
        max_degree=int(np.max(degrees)),
        connected_components=components,
        diameter=diameter,
        mean_shortest_path=mean_shortest,
        algebraic_connectivity=algebraic,
    )
