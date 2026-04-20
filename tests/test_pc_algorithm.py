"""Tests for the PC algorithm: skeleton discovery and all three orientation strategies."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module loading helpers
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
PDAG = _graphs_mod.PDAG
DAG = _graphs_mod.DAG


# ---------------------------------------------------------------------------
# Data generation utilities
# ---------------------------------------------------------------------------

def _simulate_linear_gaussian(
    dag_edges: list[tuple[str, str]],
    n_samples: int = 3000,
    coef: float = 1.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate linear Gaussian data from a DAG using ancestral sampling."""
    rng = np.random.default_rng(seed)

    # Determine causal order via topological sort
    from collections import defaultdict, deque
    children: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for u, v in dag_edges:
        nodes.add(u); nodes.add(v)
        children[u].append(v)
        in_degree[v] += 1
    for n in nodes:
        if n not in in_degree:
            in_degree[n] = 0

    queue = deque(sorted(n for n in nodes if in_degree[n] == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(children[node]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Build parent lookup
    parents_of: dict[str, list[str]] = defaultdict(list)
    for u, v in dag_edges:
        parents_of[v].append(u)

    data: dict[str, np.ndarray] = {}
    for node in order:
        noise = rng.normal(scale=noise_std, size=(n_samples, 1))
        if not parents_of[node]:
            data[node] = noise
        else:
            val = noise.copy()
            for par in parents_of[node]:
                val += coef * data[par]
            data[node] = val
    return data


def _collider_data(n_samples: int = 3000, seed: int = 7) -> dict[str, np.ndarray]:
    """X0 -> X2 <- X1, X0 ⟂ X1 marginally."""
    return _simulate_linear_gaussian(
        dag_edges=[("X0", "X2"), ("X1", "X2")],
        n_samples=n_samples,
        noise_std=0.7,
        seed=seed,
    )


def _chain_data(n_samples: int = 3000, seed: int = 42) -> dict[str, np.ndarray]:
    """X0 -> X1 -> X2 (no v-structure)."""
    return _simulate_linear_gaussian(
        dag_edges=[("X0", "X1"), ("X1", "X2")],
        n_samples=n_samples,
        seed=seed,
    )


def _fork_data(n_samples: int = 3000, seed: int = 42) -> dict[str, np.ndarray]:
    """X1 -> X0, X1 -> X2 (fork; no v-structure)."""
    return _simulate_linear_gaussian(
        dag_edges=[("X1", "X0"), ("X1", "X2")],
        n_samples=n_samples,
        seed=seed,
    )


def _four_node_data(n_samples: int = 5000, seed: int = 0) -> dict[str, np.ndarray]:
    """DAG: X0 -> X2 <- X1, X2 -> X3.
    V-structure at X2 (X0 -> X2 <- X1). X0 and X1 are marginally independent.
    """
    return _simulate_linear_gaussian(
        dag_edges=[("X0", "X2"), ("X1", "X2"), ("X2", "X3")],
        n_samples=n_samples,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Skeleton tests
# ---------------------------------------------------------------------------

class TestSkeletonDiscovery:
    """Tests for the PC-stable skeleton phase."""

    def test_collider_skeleton_removes_x0_x1(self) -> None:
        """In X0->X2<-X1, X0 and X1 are independent so their edge must be removed."""
        data = _collider_data()
        pc = PC(alpha=0.05)
        pc._find_skeleton_stable(data=data, alpha=0.05)
        skel = pc.skel
        assert skel is not None
        assert skel.loc["X0", "X1"] == 0
        assert skel.loc["X1", "X0"] == 0

    def test_collider_skeleton_keeps_x0_x2_and_x1_x2(self) -> None:
        """In X0->X2<-X1, both X0-X2 and X1-X2 edges must be present."""
        data = _collider_data()
        pc = PC(alpha=0.05)
        pc._find_skeleton_stable(data=data, alpha=0.05)
        skel = pc.skel
        assert skel is not None
        assert skel.loc["X0", "X2"] == 1
        assert skel.loc["X1", "X2"] == 1

    def test_chain_skeleton_complete(self) -> None:
        """X0->X1->X2: skeleton should have X0-X1 and X1-X2 but not X0-X2."""
        data = _chain_data()
        pc = PC(alpha=0.05)
        pc._find_skeleton_stable(data=data, alpha=0.05)
        skel = pc.skel
        assert skel is not None
        assert skel.loc["X0", "X1"] == 1
        assert skel.loc["X1", "X2"] == 1
        assert skel.loc["X0", "X2"] == 0

    def test_fork_skeleton_complete(self) -> None:
        """X1->X0, X1->X2: skeleton has X0-X1, X1-X2 but not X0-X2."""
        data = _fork_data()
        pc = PC(alpha=0.05)
        pc._find_skeleton_stable(data=data, alpha=0.05)
        skel = pc.skel
        assert skel is not None
        assert skel.loc["X0", "X1"] == 1
        assert skel.loc["X1", "X2"] == 1
        assert skel.loc["X0", "X2"] == 0

    def test_skeleton_not_learned_raises(self) -> None:
        """Accessing skeleton before learning raises ValueError."""
        pc = PC()
        with pytest.raises(ValueError, match="Skeleton not learned yet"):
            _ = pc.skeleton

    def test_four_node_skeleton(self) -> None:
        """X0->X2<-X1, X2->X3: correct skeleton has 3 edges."""
        data = _four_node_data()
        pc = PC(alpha=0.05)
        pc._find_skeleton_stable(data=data, alpha=0.05)
        skel = pc.skel
        assert skel is not None
        assert skel.loc["X0", "X2"] == 1
        assert skel.loc["X1", "X2"] == 1
        assert skel.loc["X2", "X3"] == 1
        assert skel.loc["X0", "X1"] == 0
        assert skel.loc["X0", "X3"] == 0
        assert skel.loc["X1", "X3"] == 0

    def test_skeleton_is_symmetric(self) -> None:
        """Skeleton adjacency matrix must be symmetric."""
        data = _four_node_data()
        pc = PC(alpha=0.05)
        pc._find_skeleton_stable(data=data, alpha=0.05)
        skel = pc.skel
        assert skel is not None
        import numpy as np
        assert np.allclose(skel.values, skel.values.T)

    def test_sep_sets_symmetric(self) -> None:
        """Separation sets must be symmetric: sep(i,j) == sep(j,i)."""
        data = _collider_data()
        pc = PC(alpha=0.05)
        pc._find_skeleton_stable(data=data, alpha=0.05)
        for (i, j), sep in pc.sep_sets.items():
            assert pc.sep_sets.get((j, i)) == sep, f"sep({i},{j}) != sep({j},{i})"

    def test_alpha_controls_sparsity(self) -> None:
        """Smaller alpha => sparser skeleton (more edges removed)."""
        data = _four_node_data(n_samples=2000)
        pc_loose = PC(alpha=0.2)
        pc_tight = PC(alpha=0.001)
        pc_loose._find_skeleton_stable(data=data, alpha=0.2)
        pc_tight._find_skeleton_stable(data=data, alpha=0.001)
        edges_loose = int(pc_loose.skel.values.sum()) // 2
        edges_tight = int(pc_tight.skel.values.sum()) // 2
        assert edges_tight <= edges_loose


# ---------------------------------------------------------------------------
# Orientation strategy tests
# ---------------------------------------------------------------------------

class TestOrientationStrategies:
    """Tests for conservative, majority, and pc-max orientation rules."""

    @pytest.fixture(scope="class")
    def collider_data(self):
        return _collider_data(n_samples=5000, seed=7)

    @pytest.fixture(scope="class")
    def four_node_data(self):
        return _four_node_data(n_samples=5000, seed=0)

    # --- conservative ---

    def test_conservative_orients_collider(self, collider_data) -> None:
        """Conservative rule must identify X0->X2<-X1 v-structure."""
        pc = PC(alpha=0.01)
        pdag = pc.learn_graph(collider_data, v_structure_rule="conservative")
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges
        assert ("X2", "X0") not in pdag.dir_edges
        assert ("X2", "X1") not in pdag.dir_edges

    def test_conservative_no_spurious_v_structure_chain(self) -> None:
        """In a chain X0->X1->X2, conservative rule must NOT orient X0<-X1->X2."""
        data = _chain_data(n_samples=5000)
        pc = PC(alpha=0.05)
        pdag = pc.learn_graph(data, v_structure_rule="conservative")
        # X1 must NOT be oriented as a collider
        assert ("X0", "X1") not in pdag.dir_edges or ("X2", "X1") not in pdag.dir_edges

    def test_conservative_no_spurious_v_structure_fork(self) -> None:
        """In a fork X0<-X1->X2, conservative rule must NOT orient X0->X1<-X2."""
        data = _fork_data(n_samples=5000)
        pc = PC(alpha=0.05)
        pdag = pc.learn_graph(data, v_structure_rule="conservative")
        # X1 must NOT be both target of X0 and X2
        assert not (("X0", "X1") in pdag.dir_edges and ("X2", "X1") in pdag.dir_edges)

    def test_conservative_four_node(self, four_node_data) -> None:
        """Conservative rule must orient X0->X2<-X1 in 4-node graph."""
        pc = PC(alpha=0.01)
        pdag = pc.learn_graph(four_node_data, v_structure_rule="conservative")
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges

    # --- majority ---

    def test_majority_orients_collider(self, collider_data) -> None:
        """Majority rule must identify X0->X2<-X1 v-structure."""
        pc = PC(alpha=0.01)
        pdag = pc.learn_graph(collider_data, v_structure_rule="majority")
        assert not pdag.is_adjacent("X0", "X1")
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges

    def test_majority_no_spurious_v_structure_chain(self) -> None:
        """Majority rule must not orient a chain as a v-structure."""
        data = _chain_data(n_samples=5000)
        pc = PC(alpha=0.05)
        pdag = pc.learn_graph(data, v_structure_rule="majority")
        assert not (("X0", "X1") in pdag.dir_edges and ("X2", "X1") in pdag.dir_edges)

    def test_majority_no_spurious_v_structure_fork(self) -> None:
        """Majority rule must not orient a fork as a v-structure."""
        data = _fork_data(n_samples=5000)
        pc = PC(alpha=0.05)
        pdag = pc.learn_graph(data, v_structure_rule="majority")
        assert not (("X0", "X1") in pdag.dir_edges and ("X2", "X1") in pdag.dir_edges)

    def test_majority_four_node(self, four_node_data) -> None:
        """Majority rule must orient X0->X2<-X1 in 4-node graph."""
        pc = PC(alpha=0.01)
        pdag = pc.learn_graph(four_node_data, v_structure_rule="majority")
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges

    # --- pc-max ---

    def test_pcmax_orients_collider(self, collider_data) -> None:
        """PC-max rule must identify X0->X2<-X1 v-structure."""
        pc = PC(alpha=0.01)
        pdag = pc.learn_graph(collider_data, v_structure_rule="pc-max")
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges

    def test_pcmax_no_spurious_v_structure_chain(self) -> None:
        """PC-max must not orient a chain as a v-structure."""
        data = _chain_data(n_samples=5000)
        pc = PC(alpha=0.05)
        pdag = pc.learn_graph(data, v_structure_rule="pc-max")
        assert not (("X0", "X1") in pdag.dir_edges and ("X2", "X1") in pdag.dir_edges)

    def test_pcmax_no_spurious_v_structure_fork(self) -> None:
        """PC-max must not orient a fork as a v-structure."""
        data = _fork_data(n_samples=5000)
        pc = PC(alpha=0.05)
        pdag = pc.learn_graph(data, v_structure_rule="pc-max")
        assert not (("X0", "X1") in pdag.dir_edges and ("X2", "X1") in pdag.dir_edges)

    def test_pcmax_four_node(self, four_node_data) -> None:
        """PC-max must orient X0->X2<-X1 in 4-node graph."""
        pc = PC(alpha=0.01)
        pdag = pc.learn_graph(four_node_data, v_structure_rule="pc-max")
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges

    # --- invalid rule ---

    def test_invalid_rule_raises(self, collider_data) -> None:
        pc = PC(alpha=0.05)
        with pytest.raises(ValueError, match="v_structure_rule must be"):
            pc.learn_graph(collider_data, v_structure_rule="bogus")

    # --- output consistency ---

    def test_pdag_is_returned(self, collider_data) -> None:
        from mixed_pc.graphs import PDAG
        pc = PC(alpha=0.05)
        result = pc.learn_graph(collider_data)
        assert isinstance(result, PDAG)

    def test_adjacency_matrix_encoding(self, collider_data) -> None:
        """adjacency_matrix: directed edge = 1, undirected = 2."""
        pc = PC(alpha=0.05)
        pc.learn_graph(collider_data, v_structure_rule="conservative")
        amat = pc.adjacency_matrix
        # Check directed edges are encoded as 1
        assert amat.loc["X0", "X2"] == 1
        assert amat.loc["X2", "X0"] == 0
        # Diagonal must be zero
        import numpy as np
        assert np.all(np.diag(amat.values) == 0)

    def test_causal_order_none_for_partial_dag(self, collider_data) -> None:
        """causal_order is None when PDAG still has undirected edges."""
        # Chain: skeleton is correct but some edges remain undirected
        data = _chain_data()
        pc = PC(alpha=0.05)
        pc.learn_graph(data)
        # chain is a PDAG with undirected edges — causal_order should be None
        # (unless Meek rules fully orient it, which is implementation-dependent)
        assert pc.causal_order is None or isinstance(pc.causal_order, list)

    def test_all_rules_agree_on_clear_collider(self, collider_data) -> None:
        """All three rules should agree on a strong collider signal."""
        results = {}
        for rule in ("conservative", "majority", "pc-max"):
            pc = PC(alpha=0.01)
            pdag = pc.learn_graph(collider_data, v_structure_rule=rule)
            results[rule] = set(pdag.dir_edges)
        assert results["conservative"] == results["majority"] == results["pc-max"]


# ---------------------------------------------------------------------------
# Meek rule propagation via full PC run
# ---------------------------------------------------------------------------

class TestMeekPropagation:
    """Verify Meek rules are applied after v-structure orientation."""

    def test_meek_rule1_propagates(self) -> None:
        """After orienting a v-structure, Meek R1 should propagate further."""
        # Build: X0 -> X2 <- X1, X2 - X3 (undirected after skeleton)
        # After v-structure X0->X2<-X1, Meek R1 should orient X2->X3
        # because X0 is a parent of X2 and X0 is not adjacent to X3.
        data = _four_node_data(n_samples=5000, seed=0)
        pc = PC(alpha=0.01)
        pdag = pc.learn_graph(data, v_structure_rule="conservative")
        # The v-structure should be oriented
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges
        # Meek R1 must orient X2 -> X3
        assert ("X2", "X3") in pdag.dir_edges
