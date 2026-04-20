"""Benchmark and correctness comparison against the causal-learn PC implementation.

Runs both implementations on synthetic linear-Gaussian data from known random DAGs,
then compares skeleton accuracy, v-structure accuracy, and runtime.

Run with: pytest tests/test_benchmark_causallearn.py -v -s
"""

import importlib.util
import sys
import time
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from causallearn.search.ConstraintBased.PC import pc as cl_pc


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load_module(name: str, filename: str):
    package_dir = Path(__file__).resolve().parents[1] / "mixed-pc"
    if "mixed_pc" not in sys.modules:
        pkg = types.ModuleType("mixed_pc")
        pkg.__path__ = [str(package_dir)]
        sys.modules["mixed_pc"] = pkg
    spec = importlib.util.spec_from_file_location(f"mixed_pc.{name}", package_dir / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"mixed_pc.{name}"] = module
    spec.loader.exec_module(module)
    return module


_graphs_mod = _load_module("graphs", "graphs.py")
_pc_mod = _load_module("pc_algorithm", "pc_algorithm.py")
PC = _pc_mod.PC
dag2cpdag = _graphs_mod.dag2cpdag


# ---------------------------------------------------------------------------
# Random DAG and data generation
# ---------------------------------------------------------------------------

def _random_dag(n_nodes: int, expected_degree: float = 2.0, seed: int = 0) -> nx.DiGraph:
    """Sample a random DAG using the Erdos-Renyi model on a fixed causal order."""
    rng = np.random.default_rng(seed)
    nodes = [f"X{i}" for i in range(n_nodes)]
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    p_edge = expected_degree / (n_nodes - 1)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p_edge:
                dag.add_edge(nodes[i], nodes[j])
    return dag


def _simulate_from_dag(
    dag: nx.DiGraph,
    n_samples: int,
    coef_range: tuple[float, float] = (0.5, 1.5),
    noise_std: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Ancestral sampling from a linear Gaussian DAG. Returns n_samples x n_nodes array."""
    rng = np.random.default_rng(seed)
    order = list(nx.topological_sort(dag))
    node_idx = {n: i for i, n in enumerate(order)}
    n = len(order)
    data = np.zeros((n_samples, n))

    for node in order:
        parents = list(dag.predecessors(node))
        noise = rng.normal(scale=noise_std, size=n_samples)
        if not parents:
            data[:, node_idx[node]] = noise
        else:
            val = noise.copy()
            for par in parents:
                coef = rng.uniform(*coef_range) * rng.choice([-1, 1])
                val += coef * data[:, node_idx[par]]
            data[:, node_idx[node]] = val

    return data, order


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _skeleton_from_dag(dag: nx.DiGraph) -> set[frozenset]:
    return {frozenset(e) for e in dag.edges()}


def _skeleton_from_pdag(pdag) -> set[frozenset]:
    edges = set()
    for e in pdag.dir_edges:
        edges.add(frozenset(e))
    for e in pdag.undir_edges:
        edges.add(frozenset(e))
    return edges


def _skeleton_from_cl_graph(graph: np.ndarray, node_names: list[str]) -> set[frozenset]:
    """Extract undirected skeleton from causal-learn G.graph matrix."""
    edges = set()
    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i, j] != 0 or graph[j, i] != 0:
                edges.add(frozenset([node_names[i], node_names[j]]))
    return edges


def _vstructs_from_pdag(pdag) -> set[tuple]:
    """Collect all v-structures (X -> Z <- Y, X not adjacent to Y) from a PDAG."""
    vstructs = set()
    for node in pdag.nodes:
        parents = list(pdag.parents(node))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                p1, p2 = parents[i], parents[j]
                if not pdag.is_adjacent(p1, p2):
                    vstructs.add((min(p1, p2), node, max(p1, p2)))
    return vstructs


def _vstructs_from_cl_graph(graph: np.ndarray, node_names: list[str]) -> set[tuple]:
    """Extract v-structures from causal-learn G.graph matrix.

    Encoding: graph[i,j]=-1, graph[j,i]=1 means i->j.
    """
    n = len(node_names)
    # Build parent sets
    parents: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if graph[i, j] == -1 and graph[j, i] == 1:
                parents[j].add(i)

    vstructs = set()
    for z in range(n):
        ps = sorted(parents[z])
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                p1, p2 = ps[a], ps[b]
                # check non-adjacent
                if graph[p1, p2] == 0 and graph[p2, p1] == 0:
                    n1, n2 = node_names[p1], node_names[p2]
                    vstructs.add((min(n1, n2), node_names[z], max(n1, n2)))
    return vstructs


def _skeleton_metrics(pred: set[frozenset], true: set[frozenset]) -> dict:
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _shd_skeleton(pred: set[frozenset], true: set[frozenset]) -> int:
    """Skeleton SHD = |pred △ true| (insertions + deletions)."""
    return len(pred.symmetric_difference(true))


# ---------------------------------------------------------------------------
# Fixed test graphs for correctness verification
# ---------------------------------------------------------------------------

class TestCausalLearnCorrectness:
    """Verify our implementation matches causal-learn on known structures."""

    def _run_both(self, dag_edges, n_samples=5000, alpha=0.01, seed=42):
        """Run both implementations on the same data, return metrics."""
        nodes = sorted({n for e in dag_edges for n in e})
        node_idx = {n: i for i, n in enumerate(nodes)}

        rng = np.random.default_rng(seed)
        # Build data via ancestral sampling
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        dag.add_edges_from(dag_edges)
        order = list(nx.topological_sort(dag))

        data_dict: dict[str, np.ndarray] = {}
        for node in order:
            parents = list(dag.predecessors(node))
            noise = rng.normal(scale=0.5, size=(n_samples, 1))
            if not parents:
                data_dict[node] = noise
            else:
                val = noise.copy()
                for par in parents:
                    val += rng.uniform(0.8, 1.2) * data_dict[par]
                data_dict[node] = val

        data_arr = np.hstack([data_dict[n] for n in nodes])

        # Our implementation
        pc = PC(alpha=alpha)
        pdag_ours = pc.learn_graph(data_dict, v_structure_rule="conservative")

        # causal-learn (conservative = uc_rule=1)
        cl_result = cl_pc(
            data_arr, alpha=alpha, indep_test="fisherz", stable=True,
            uc_rule=1, show_progress=False, node_names=nodes,
        )

        true_skeleton = _skeleton_from_dag(dag)
        ours_skeleton = _skeleton_from_pdag(pdag_ours)
        cl_skeleton = _skeleton_from_cl_graph(cl_result.G.graph, nodes)

        ours_vs = _vstructs_from_pdag(pdag_ours)
        cl_vs = _vstructs_from_cl_graph(cl_result.G.graph, nodes)

        return {
            "true_skel": true_skeleton,
            "ours_skel": ours_skeleton,
            "cl_skel": cl_skeleton,
            "ours_vs": ours_vs,
            "cl_vs": cl_vs,
        }

    def test_collider_skeleton_matches_causallearn(self) -> None:
        """Both implementations should find the same skeleton for X0->X2<-X1."""
        r = self._run_both([("X0", "X2"), ("X1", "X2")])
        assert r["ours_skel"] == r["cl_skel"]

    def test_collider_vstructure_matches_causallearn(self) -> None:
        """Both should orient X0->X2<-X1 as a v-structure."""
        r = self._run_both([("X0", "X2"), ("X1", "X2")])
        assert r["ours_vs"] == r["cl_vs"]
        assert len(r["ours_vs"]) == 1

    def test_four_node_skeleton_matches_causallearn(self) -> None:
        """X0->X2<-X1, X2->X3: skeleton should match causal-learn."""
        r = self._run_both([("X0", "X2"), ("X1", "X2"), ("X2", "X3")])
        assert r["ours_skel"] == r["cl_skel"]

    def test_four_node_vstructure_matches_causallearn(self) -> None:
        """X0->X2<-X1, X2->X3: v-structure should match causal-learn."""
        r = self._run_both([("X0", "X2"), ("X1", "X2"), ("X2", "X3")])
        assert r["ours_vs"] == r["cl_vs"]

    def test_chain_skeleton_matches_causallearn(self) -> None:
        """Chain X0->X1->X2: skeleton should match."""
        r = self._run_both([("X0", "X1"), ("X1", "X2")])
        assert r["ours_skel"] == r["cl_skel"]

    def test_chain_no_vstructure_matches_causallearn(self) -> None:
        """Chain has no v-structures; both should agree."""
        r = self._run_both([("X0", "X1"), ("X1", "X2")])
        assert r["ours_vs"] == r["cl_vs"] == set()

    def test_fork_skeleton_matches_causallearn(self) -> None:
        """Fork X1->X0, X1->X2: skeleton should match."""
        r = self._run_both([("X1", "X0"), ("X1", "X2")])
        assert r["ours_skel"] == r["cl_skel"]

    def test_five_node_diamond_skeleton(self) -> None:
        """Diamond: X0->X1, X0->X2, X1->X3, X2->X3 (two v-structures at X3)."""
        r = self._run_both([("X0", "X1"), ("X0", "X2"), ("X1", "X3"), ("X2", "X3")])
        assert _shd_skeleton(r["ours_skel"], r["cl_skel"]) <= 1


# ---------------------------------------------------------------------------
# Speed benchmark (not a hard pass/fail, just reports timing)
# ---------------------------------------------------------------------------

class TestSpeedBenchmark:
    """Compare runtime of our PC vs causal-learn across problem sizes."""

    @pytest.mark.parametrize("n_nodes,n_samples", [
        (5, 500),
        (10, 1000),
        (15, 2000),
    ])
    def test_runtime_comparison(self, n_nodes: int, n_samples: int, capsys) -> None:
        """Report timing for both implementations (no hard assertion on speed)."""
        seed = 99
        dag = _random_dag(n_nodes, expected_degree=2.0, seed=seed)
        data_arr, order = _simulate_from_dag(dag, n_samples=n_samples, seed=seed)
        data_dict = {n: data_arr[:, i : i + 1] for i, n in enumerate(order)}
        alpha = 0.05

        # Our implementation
        t0 = time.perf_counter()
        pc = PC(alpha=alpha)
        pc.learn_graph(data_dict, v_structure_rule="conservative")
        t_ours = time.perf_counter() - t0

        # causal-learn
        t0 = time.perf_counter()
        cl_pc(data_arr, alpha=alpha, indep_test="fisherz", stable=True,
              uc_rule=1, show_progress=False)
        t_cl = time.perf_counter() - t0

        with capsys.disabled():
            print(
                f"\n  nodes={n_nodes:2d}, n={n_samples:5d} | "
                f"ours={t_ours:.3f}s  causal-learn={t_cl:.3f}s  "
                f"ratio={t_ours/t_cl:.2f}x"
            )

        # Sanity: both finish in reasonable time
        assert t_ours < 60.0
        assert t_cl < 60.0

    @pytest.mark.parametrize("n_nodes,n_samples", [
        (5, 500),
        (10, 1000),
        (15, 2000),
    ])
    def test_skeleton_accuracy_vs_causallearn(self, n_nodes: int, n_samples: int, capsys) -> None:
        """Compare skeleton F1 of our implementation vs causal-learn."""
        seed = 7
        dag = _random_dag(n_nodes, expected_degree=2.0, seed=seed)
        data_arr, order = _simulate_from_dag(dag, n_samples=n_samples, seed=seed)
        data_dict = {n: data_arr[:, i : i + 1] for i, n in enumerate(order)}
        alpha = 0.05

        true_skel = _skeleton_from_dag(dag)

        pc = PC(alpha=alpha)
        pdag_ours = pc.learn_graph(data_dict, v_structure_rule="conservative")
        ours_skel = _skeleton_from_pdag(pdag_ours)

        cl_result = cl_pc(data_arr, alpha=alpha, indep_test="fisherz", stable=True,
                          uc_rule=1, show_progress=False)
        cl_skel = _skeleton_from_cl_graph(cl_result.G.graph, order)

        m_ours = _skeleton_metrics(ours_skel, true_skel)
        m_cl = _skeleton_metrics(cl_skel, true_skel)
        shd_diff = abs(_shd_skeleton(ours_skel, true_skel) - _shd_skeleton(cl_skel, true_skel))

        with capsys.disabled():
            print(
                f"\n  nodes={n_nodes:2d}, n={n_samples:5d} | "
                f"ours F1={m_ours['f1']:.3f} (P={m_ours['precision']:.2f} R={m_ours['recall']:.2f})  "
                f"cl F1={m_cl['f1']:.3f} (P={m_cl['precision']:.2f} R={m_cl['recall']:.2f})  "
                f"|SHD_diff|={shd_diff}"
            )

        # Our skeleton SHD should be within 2 edges of causal-learn's
        assert shd_diff <= 2, (
            f"Skeleton SHD too far from causal-learn: ours={_shd_skeleton(ours_skel, true_skel)}, "
            f"cl={_shd_skeleton(cl_skel, true_skel)}"
        )
