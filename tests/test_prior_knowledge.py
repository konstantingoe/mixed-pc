"""Tests for prior-knowledge support in PC.

Covers:
- PriorKnowledge dataclass: normalization, predicates, validation errors.
- Skeleton phase: forbidden edges removed, required edges protected, layering
  restricts conditioning sets.
- Orientation phase: layering / required_directions pre-orient between-layer edges,
  v-structures respect forbidden directions, Meek output reconciled with prior.
- End-to-end: same DGP, different prior regimes — accuracy and CI-test count.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module loading (mirror the pattern used by sibling test files)
# ---------------------------------------------------------------------------

def _load_module(name: str, filename: str):
    package_dir = Path(__file__).resolve().parents[1] / "mixpc"
    if "mixpc" not in sys.modules:
        pkg = types.ModuleType("mixpc")
        pkg.__path__ = [str(package_dir)]
        sys.modules["mixpc"] = pkg
    spec = importlib.util.spec_from_file_location(f"mixpc.{name}", package_dir / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"mixpc.{name}"] = module
    spec.loader.exec_module(module)
    return module


_corr_mod = _load_module("correlations", "correlations.py")
_itest_mod = _load_module("independence_tests", "independence_tests.py")
_prior_mod = _load_module("prior_knowledge", "prior_knowledge.py")
_pc_mod = _load_module("pc_algorithm", "pc_algorithm.py")

PriorKnowledge = _prior_mod.PriorKnowledge
MixedFisherZ = _itest_mod.MixedFisherZ
PC = _pc_mod.PC


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# PriorKnowledge: normalization and predicates
# ---------------------------------------------------------------------------

class TestPriorKnowledgePredicates:
    def test_undirected_edges_are_symmetric(self) -> None:
        pk = PriorKnowledge(forbidden_edges=[("A", "B")])
        assert pk.is_forbidden_edge("A", "B")
        assert pk.is_forbidden_edge("B", "A")

    def test_required_directions_imply_required_edge(self) -> None:
        pk = PriorKnowledge(required_directions=[("A", "B")])
        assert pk.is_required_edge("A", "B")
        assert pk.is_required_edge("B", "A")

    def test_layering_implies_forbidden_reverse_direction(self) -> None:
        pk = PriorKnowledge(layering=[["A"], ["B", "C"], ["D"]])
        assert pk.is_forbidden_direction("B", "A")
        assert not pk.is_forbidden_direction("A", "B")
        # Same layer: neither direction is forbidden by layering.
        assert not pk.is_forbidden_direction("B", "C")
        assert not pk.is_forbidden_direction("C", "B")

    def test_required_direction_for_resolves_layering(self) -> None:
        pk = PriorKnowledge(layering=[["A"], ["B"]])
        assert pk.required_direction_for("A", "B") == ("A", "B")
        assert pk.required_direction_for("B", "A") == ("A", "B")

    def test_required_direction_for_resolves_forbidden_alternative(self) -> None:
        pk = PriorKnowledge(forbidden_directions=[("A", "B")])
        assert pk.required_direction_for("A", "B") == ("B", "A")

    def test_required_direction_for_returns_none_when_unconstrained(self) -> None:
        pk = PriorKnowledge()
        assert pk.required_direction_for("A", "B") is None

    def test_filter_separating_set_drops_future_layers(self) -> None:
        pk = PriorKnowledge(layering=[["A"], ["B"], ["C", "D"]])
        # Pair (A, B) — max layer 1 — drops C and D (layer 2).
        assert sorted(pk.filter_separating_set("A", "B", ["C", "D"])) == []
        # Pair (B, C) — max layer 2 — keeps everything.
        assert sorted(pk.filter_separating_set("B", "C", ["A", "D"])) == ["A", "D"]


class TestPriorKnowledgeValidation:
    def test_unknown_node_raises(self) -> None:
        pk = PriorKnowledge(required_edges=[("A", "Z")])
        with pytest.raises(ValueError, match="unknown nodes"):
            pk.validate(nodes={"A", "B"})

    def test_self_loop_raises(self) -> None:
        pk = PriorKnowledge(required_edges=[("A", "A")])
        with pytest.raises(ValueError, match="Self-loop"):
            pk.validate(nodes={"A"})

    def test_required_and_forbidden_overlap_raises(self) -> None:
        pk = PriorKnowledge(required_edges=[("A", "B")], forbidden_edges=[("B", "A")])
        with pytest.raises(ValueError, match="both required_edges and forbidden_edges"):
            pk.validate(nodes={"A", "B"})

    def test_required_direction_against_layering_raises(self) -> None:
        pk = PriorKnowledge(
            required_directions=[("B", "A")],
            layering=[["A"], ["B"]],
        )
        with pytest.raises(ValueError, match="contradicts layering"):
            pk.validate(nodes={"A", "B"})

    def test_duplicate_node_in_layers_raises(self) -> None:
        pk = PriorKnowledge(layering=[["A", "B"], ["B"]])
        with pytest.raises(ValueError, match="multiple layers"):
            pk.validate(nodes={"A", "B"})

    def test_opposite_required_directions_raise(self) -> None:
        pk = PriorKnowledge(required_directions=[("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="Conflicting required_directions"):
            pk.validate(nodes={"A", "B"})


# ---------------------------------------------------------------------------
# Skeleton phase: forbidden / required / layering effects on the skeleton itself
# ---------------------------------------------------------------------------

class TestPriorKnowledgeInSkeleton:
    """Use a chain X0 -> X1 -> X2 -> X3 with continuous Gaussian noise."""

    @pytest.fixture(scope="class")
    def chain_data(self) -> dict[str, np.ndarray]:
        rng = _rng(11)
        n = 1500
        x0 = rng.normal(size=(n, 1))
        x1 = x0 + 0.3 * rng.normal(size=(n, 1))
        x2 = x1 + 0.3 * rng.normal(size=(n, 1))
        x3 = x2 + 0.3 * rng.normal(size=(n, 1))
        return {"X0": x0, "X1": x1, "X2": x2, "X3": x3}

    def test_forbidden_edge_is_absent(self, chain_data) -> None:
        pk = PriorKnowledge(forbidden_edges=[("X0", "X1")])
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pc.learn_graph(chain_data, prior_knowledge=pk)
        assert pc.skel.loc["X0", "X1"] == 0
        assert pc.skel.loc["X1", "X0"] == 0

    def test_required_edge_survives_marginal_independence(self, chain_data) -> None:
        # In the chain X0 -> X1 -> X2, X0 and X2 are marginally dependent but become
        # independent given X1. Without prior knowledge PC removes the X0-X2 edge;
        # marking it required must keep it.
        pk = PriorKnowledge(required_edges=[("X0", "X2")])
        pc_with = PC(alpha=0.05, test=MixedFisherZ)
        pc_with.learn_graph(chain_data, prior_knowledge=pk)
        assert pc_with.skel.loc["X0", "X2"] == 1

        pc_without = PC(alpha=0.05, test=MixedFisherZ)
        pc_without.learn_graph(chain_data)
        assert pc_without.skel.loc["X0", "X2"] == 0

    def test_layering_skips_future_separators(self) -> None:
        """In a chain, layering forbids X2 from separating (X0, X3): with full layering
        the X0-X3 edge is removed only via the within-past conditioning set {X1}.
        """
        rng = _rng(22)
        n = 2000
        x0 = rng.normal(size=(n, 1))
        x1 = x0 + 0.3 * rng.normal(size=(n, 1))
        x2 = x1 + 0.3 * rng.normal(size=(n, 1))
        x3 = x2 + 0.3 * rng.normal(size=(n, 1))
        data = {"X0": x0, "X1": x1, "X2": x2, "X3": x3}

        pk = PriorKnowledge(layering=[["X0"], ["X1"], ["X2"], ["X3"]])
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pc.learn_graph(data, prior_knowledge=pk)
        # All forward chain edges remain; cross edges are removed.
        assert pc.skel.loc["X0", "X1"] == 1
        assert pc.skel.loc["X1", "X2"] == 1
        assert pc.skel.loc["X2", "X3"] == 1
        assert pc.skel.loc["X0", "X2"] == 0
        assert pc.skel.loc["X1", "X3"] == 0
        assert pc.skel.loc["X0", "X3"] == 0


# ---------------------------------------------------------------------------
# Orientation: layering and required_directions pre-orient edges
# ---------------------------------------------------------------------------

class TestPriorKnowledgeOrientation:
    @pytest.fixture(scope="class")
    def collider_data(self) -> dict[str, np.ndarray]:
        rng = _rng(33)
        n = 2000
        x0 = rng.normal(size=(n, 1))
        x1 = rng.normal(size=(n, 1))
        x2 = x0 + x1 + 0.3 * rng.normal(size=(n, 1))
        return {"X0": x0, "X1": x1, "X2": x2}

    def test_layering_forces_orientation(self, collider_data) -> None:
        pk = PriorKnowledge(layering=[["X0", "X1"], ["X2"]])
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pdag = pc.learn_graph(collider_data, prior_knowledge=pk)
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges
        # Reverse direction never appears.
        assert ("X2", "X0") not in pdag.dir_edges
        assert ("X2", "X1") not in pdag.dir_edges

    def test_required_direction_pins_orientation(self, collider_data) -> None:
        # Manually pin X0 -> X2 (the natural direction); PC must respect it.
        pk = PriorKnowledge(required_directions=[("X0", "X2")])
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pdag = pc.learn_graph(collider_data, prior_knowledge=pk)
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X2", "X0") not in pdag.dir_edges

    def test_forbidden_direction_blocks_v_structure(self, collider_data) -> None:
        # Forbid orienting X0 -> X2: the v-structure X0 -> X2 <- X1 must not be set.
        pk = PriorKnowledge(forbidden_directions=[("X0", "X2"), ("X1", "X2")])
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pdag = pc.learn_graph(collider_data, prior_knowledge=pk)
        # Neither forbidden orientation should appear.
        assert ("X0", "X2") not in pdag.dir_edges
        assert ("X1", "X2") not in pdag.dir_edges


# ---------------------------------------------------------------------------
# End-to-end: same DGP, multiple prior regimes — accuracy + CI-test count
# ---------------------------------------------------------------------------

def _ground_truth_undirected() -> set[frozenset[str]]:
    # Layered DGP used by the benchmark fixture below.
    edges = {("L1", "L2a"), ("L1", "L2b"), ("L2a", "L3"), ("L2b", "L3"), ("L3", "L4")}
    return {frozenset(e) for e in edges}


def _skeleton_edges(skel: pd.DataFrame) -> set[frozenset[str]]:
    edges: set[frozenset[str]] = set()
    nodes = list(skel.columns)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if skel.iloc[i, j] == 1:
                edges.add(frozenset((nodes[i], nodes[j])))
    return edges


class TestPriorKnowledgeBenchmark:
    """Layered DGP: L1 -> {L2a, L2b} -> L3 -> L4. L2a and L2b share layer 2."""

    @pytest.fixture(scope="class")
    def layered_data(self) -> dict[str, np.ndarray]:
        rng = _rng(44)
        n = 2000
        l1 = rng.normal(size=(n, 1))
        l2a = l1 + 0.3 * rng.normal(size=(n, 1))
        l2b = l1 + 0.3 * rng.normal(size=(n, 1))
        l3 = l2a + l2b + 0.3 * rng.normal(size=(n, 1))
        l4 = l3 + 0.3 * rng.normal(size=(n, 1))
        return {"L1": l1, "L2a": l2a, "L2b": l2b, "L3": l3, "L4": l4}

    def test_layering_reduces_ci_test_count(self, layered_data) -> None:
        baseline = PC(alpha=0.05, test=MixedFisherZ)
        baseline.learn_graph(layered_data)

        with_layering = PC(alpha=0.05, test=MixedFisherZ)
        pk = PriorKnowledge(layering=[["L1"], ["L2a", "L2b"], ["L3"], ["L4"]])
        with_layering.learn_graph(layered_data, prior_knowledge=pk)

        # Layering can only forbid sep-sets — never adds tests — so the count must drop or tie.
        assert with_layering.ci_test_count <= baseline.ci_test_count

    def test_layering_preserves_skeleton(self, layered_data) -> None:
        pk = PriorKnowledge(layering=[["L1"], ["L2a", "L2b"], ["L3"], ["L4"]])
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pc.learn_graph(layered_data, prior_knowledge=pk)
        assert _skeleton_edges(pc.skel) == _ground_truth_undirected()

    def test_full_prior_recovers_dag(self, layered_data) -> None:
        """With layering, every edge gets a forced orientation — the result is a DAG,
        not a CPDAG, and it matches the true generating structure."""
        pk = PriorKnowledge(layering=[["L1"], ["L2a", "L2b"], ["L3"], ["L4"]])
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pdag = pc.learn_graph(layered_data, prior_knowledge=pk)
        expected_dir = {
            ("L1", "L2a"), ("L1", "L2b"),
            ("L2a", "L3"), ("L2b", "L3"),
            ("L3", "L4"),
        }
        assert set(pdag.dir_edges) == expected_dir
        assert pdag.undir_edges == []
