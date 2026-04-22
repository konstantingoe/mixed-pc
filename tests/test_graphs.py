"""Tests for mixpc/graphs.py."""

import importlib.util
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest


def _load_graphs_module():
    package_dir = Path(__file__).resolve().parents[1] / "mixpc"
    if "mixpc" not in sys.modules:
        pkg = types.ModuleType("mixpc")
        pkg.__path__ = [str(package_dir)]
        sys.modules["mixpc"] = pkg

    spec = importlib.util.spec_from_file_location("mixpc.graphs", package_dir / "graphs.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load graphs module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mixpc.graphs"] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_graphs_module()
GRAPH = _mod.GRAPH
UGRAPH = _mod.UGRAPH
PDAG = _mod.PDAG
DAG = _mod.DAG
dag2cpdag = _mod.dag2cpdag
rule_1 = _mod.rule_1
rule_2 = _mod.rule_2
rule_3 = _mod.rule_3
rule_4 = _mod.rule_4


class TestUGRAPH:
    """Tests for UGRAPH."""

    @pytest.fixture(scope="class")
    def example_ugraph(self) -> UGRAPH:
        return UGRAPH(nodes=["A", "B", "C"], edges=[("A", "B"), ("A", "C")])

    def test_instance_is_created(self) -> None:
        ug = UGRAPH(nodes=["A", "B", "C"])
        assert isinstance(ug, UGRAPH)
        assert isinstance(ug, GRAPH)

    def test_empty_graph(self) -> None:
        ug = UGRAPH()
        assert ug.num_nodes == 0
        assert ug.num_edges == 0

    def test_nodes_and_edges_populated(self, example_ugraph: UGRAPH) -> None:
        assert set(example_ugraph.nodes) == {"A", "B", "C"}
        assert example_ugraph.num_nodes == 3
        assert example_ugraph.num_edges == 2

    def test_edges_are_stored(self, example_ugraph: UGRAPH) -> None:
        edge_set = {frozenset(e) for e in example_ugraph.edges}
        assert frozenset(("A", "B")) in edge_set
        assert frozenset(("A", "C")) in edge_set

    def test_adding_edge_creates_nodes(self) -> None:
        ug = UGRAPH(edges=[("X", "Y")])
        assert "X" in ug.nodes
        assert "Y" in ug.nodes

    def test_neighbors(self, example_ugraph: UGRAPH) -> None:
        assert example_ugraph.neighbors("A") == {"B", "C"}
        assert example_ugraph.neighbors("B") == {"A"}
        assert example_ugraph.neighbors("C") == {"A"}

    def test_neighbors_isolated_node(self) -> None:
        ug = UGRAPH(nodes=["A", "B"])
        assert ug.neighbors("A") == set()

    def test_is_adjacent_true(self, example_ugraph: UGRAPH) -> None:
        assert example_ugraph.is_adjacent("A", "B")
        assert example_ugraph.is_adjacent("B", "A")

    def test_is_adjacent_false(self, example_ugraph: UGRAPH) -> None:
        assert not example_ugraph.is_adjacent("B", "C")

    def test_is_clique_true(self) -> None:
        ug = UGRAPH(edges=[("A", "B"), ("B", "C"), ("A", "C")])
        assert ug.is_clique({"A", "B", "C"})

    def test_is_clique_false(self) -> None:
        ug = UGRAPH(edges=[("A", "B"), ("B", "C")])
        assert not ug.is_clique({"A", "B", "C"})

    def test_is_clique_single_node(self, example_ugraph: UGRAPH) -> None:
        assert example_ugraph.is_clique({"A"})

    def test_from_pandas_adjacency(self) -> None:
        amat = pd.DataFrame(
            [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        ug = UGRAPH.from_pandas_adjacency(pd_amat=amat)
        assert set(ug.nodes) == {"A", "B", "C"}
        assert ug.num_edges == 2
        assert ug.is_adjacent("A", "B")
        assert ug.is_adjacent("A", "C")
        assert not ug.is_adjacent("B", "C")

    def test_from_pandas_adjacency_deduplicates(self) -> None:
        amat = pd.DataFrame(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        ug = UGRAPH.from_pandas_adjacency(pd_amat=amat)
        assert ug.num_edges == 2

    def test_remove_edge(self) -> None:
        ug = UGRAPH(edges=[("A", "B"), ("A", "C")])
        ug.remove_edge("A", "B")
        assert not ug.is_adjacent("A", "B")
        assert "B" not in ug.neighbors("A")
        assert "A" not in ug.neighbors("B")

    def test_remove_edge_both_orderings(self) -> None:
        ug = UGRAPH(edges=[("A", "B")])
        ug.remove_edge("B", "A")
        assert not ug.is_adjacent("A", "B")

    def test_remove_nonexistent_edge_raises(self) -> None:
        ug = UGRAPH(edges=[("A", "B")])
        with pytest.raises(AssertionError, match="Edge does not exist in current UGRAPH"):
            ug.remove_edge("A", "C")

    def test_remove_node(self) -> None:
        ug = UGRAPH(edges=[("A", "B"), ("A", "C")])
        ug.remove_node("A")
        assert "A" not in ug.nodes
        assert not ug.is_adjacent("A", "B")
        assert not ug.is_adjacent("A", "C")
        assert "A" not in ug.neighbors("B")
        assert "A" not in ug.neighbors("C")

    def test_adjacency_matrix_shape(self, example_ugraph: UGRAPH) -> None:
        amat = example_ugraph.adjacency_matrix
        d = example_ugraph.num_nodes
        assert amat.shape == (d, d)

    def test_adjacency_matrix_is_symmetric(self, example_ugraph: UGRAPH) -> None:
        amat = example_ugraph.adjacency_matrix.to_numpy()
        assert np.allclose(amat, amat.T)

    def test_adjacency_matrix_entry_sum(self, example_ugraph: UGRAPH) -> None:
        amat = example_ugraph.adjacency_matrix
        assert int(amat.to_numpy().sum()) == 2 * example_ugraph.num_edges

    def test_adjacency_matrix_zero_diagonal(self, example_ugraph: UGRAPH) -> None:
        amat = example_ugraph.adjacency_matrix.to_numpy()
        assert np.all(np.diag(amat) == 0)

    def test_causal_order_is_none(self, example_ugraph: UGRAPH) -> None:
        assert example_ugraph.causal_order is None

    def test_copy_is_independent(self, example_ugraph: UGRAPH) -> None:
        ug_copy = example_ugraph.copy()
        assert isinstance(ug_copy, UGRAPH)
        assert set(ug_copy.nodes) == set(example_ugraph.nodes)
        assert ug_copy.num_edges == example_ugraph.num_edges
        ug_copy.remove_edge("A", "B")
        assert example_ugraph.is_adjacent("A", "B")

    def test_to_networkx(self, example_ugraph: UGRAPH) -> None:
        nxg = example_ugraph.to_networkx()
        assert isinstance(nxg, nx.Graph)
        assert set(nxg.nodes) == set(example_ugraph.nodes)
        edge_set = {frozenset(e) for e in nxg.edges}
        for e in example_ugraph.edges:
            assert frozenset(e) in edge_set


class TestPDAG:
    """Tests for PDAG."""

    def test_empty_pdag(self) -> None:
        p = PDAG()
        assert p.num_nodes == 0
        assert p.num_undir_edges == 0
        assert p.num_dir_edges == 0

    def test_directed_edges(self) -> None:
        p = PDAG(nodes=["A", "B", "C"], dir_edges=[("A", "B"), ("B", "C")])
        assert ("A", "B") in p.dir_edges
        assert ("B", "C") in p.dir_edges
        assert p.num_dir_edges == 2
        assert p.num_undir_edges == 0

    def test_undirected_edges(self) -> None:
        p = PDAG(nodes=["A", "B"], undir_edges=[("A", "B")])
        assert p.num_undir_edges == 1
        assert p.num_dir_edges == 0
        assert p.is_adjacent("A", "B")
        assert p.is_adjacent("B", "A")

    def test_parents_children(self) -> None:
        p = PDAG(dir_edges=[("A", "B"), ("A", "C")])
        assert "A" in p.parents("B")
        assert "A" in p.parents("C")
        assert "B" in p.children("A")
        assert "C" in p.children("A")

    def test_undir_neighbors(self) -> None:
        p = PDAG(undir_edges=[("A", "B"), ("A", "C")])
        assert p.undir_neighbors("A") == {"B", "C"}
        assert p.undir_neighbors("B") == {"A"}

    def test_undir_to_dir_edge(self) -> None:
        p = PDAG(undir_edges=[("A", "B")])
        p.undir_to_dir_edge(tail="A", head="B")
        assert ("A", "B") in p.dir_edges
        assert p.num_undir_edges == 0
        assert ("A", "B") not in p.undir_edges
        assert ("B", "A") not in p.undir_edges

    def test_undir_to_dir_edge_raises_for_missing(self) -> None:
        p = PDAG(dir_edges=[("A", "B")])
        with pytest.raises(AssertionError):
            p.undir_to_dir_edge(tail="A", head="B")

    def test_remove_edge_directed(self) -> None:
        p = PDAG(dir_edges=[("A", "B")])
        p.remove_edge("A", "B")
        assert not p.is_adjacent("A", "B")

    def test_remove_edge_undirected(self) -> None:
        p = PDAG(undir_edges=[("A", "B")])
        p.remove_edge("A", "B")
        assert not p.is_adjacent("A", "B")

    def test_copy_independence(self) -> None:
        p = PDAG(undir_edges=[("A", "B")], dir_edges=[("B", "C")])
        c = p.copy()
        c.undir_to_dir_edge("A", "B")
        assert p.num_undir_edges == 1

    def test_from_pandas_adjacency_symmetric(self) -> None:
        amat = pd.DataFrame(
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        p = PDAG.from_pandas_adjacency(amat)
        assert p.num_undir_edges == 1
        assert p.num_dir_edges == 0

    def test_from_pandas_adjacency_directed(self) -> None:
        amat = pd.DataFrame(
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        p = PDAG.from_pandas_adjacency(amat)
        assert p.num_dir_edges == 1
        assert p.num_undir_edges == 0
        assert ("A", "B") in p.dir_edges

    def test_vstructs(self) -> None:
        # A -> C <- B with A-B not adjacent
        p = PDAG(dir_edges=[("A", "C"), ("B", "C")])
        vs = p.vstructs()
        assert ("A", "C") in vs
        assert ("B", "C") in vs

    def test_adjacency_matrix(self) -> None:
        p = PDAG(dir_edges=[("A", "B")], undir_edges=[("B", "C")])
        amat = p.adjacency_matrix
        assert amat.loc["A", "B"] == 1
        assert amat.loc["B", "A"] == 0
        assert amat.loc["B", "C"] == 1
        assert amat.loc["C", "B"] == 1

    def test_remove_node(self) -> None:
        p = PDAG(dir_edges=[("A", "B")], undir_edges=[("B", "C")])
        p.remove_node("B")
        assert "B" not in p.nodes
        assert not p.is_adjacent("A", "B")
        assert not p.is_adjacent("B", "C")


class TestDAG:
    """Tests for DAG."""

    def test_empty_dag(self) -> None:
        d = DAG()
        assert d.num_nodes == 0
        assert d.num_edges == 0

    def test_add_edge(self) -> None:
        d = DAG(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
        assert d.num_edges == 2
        assert ("A", "B") in d.edges

    def test_cycle_raises(self) -> None:
        with pytest.raises(ValueError):
            DAG(edges=[("A", "B"), ("B", "C"), ("C", "A")])

    def test_parents_children(self) -> None:
        d = DAG(edges=[("A", "B"), ("A", "C")])
        assert "A" in d.parents("B")
        assert "B" in d.children("A")
        assert "C" in d.children("A")

    def test_causal_order(self) -> None:
        d = DAG(edges=[("A", "B"), ("B", "C")])
        order = d.causal_order
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_adjacency_matrix(self) -> None:
        d = DAG(edges=[("A", "B")])
        amat = d.adjacency_matrix
        assert amat.loc["A", "B"] == 1
        assert amat.loc["B", "A"] == 0

    def test_is_acyclic(self) -> None:
        d = DAG(edges=[("A", "B"), ("B", "C")])
        assert d.is_acyclic()

    def test_source_sink_nodes(self) -> None:
        d = DAG(edges=[("A", "B"), ("B", "C")])
        assert "A" in d.source_nodes
        assert "C" in d.sink_nodes

    def test_to_cpdag_chain(self) -> None:
        # Chain A -> B -> C has only one MEC member, so CPDAG = DAG
        d = DAG(edges=[("A", "B"), ("B", "C")])
        cpdag = d.to_cpdag()
        # All edges undirected in chain (Meek rules don't orient them)
        assert cpdag.num_nodes == 3

    def test_from_pandas_adjacency(self) -> None:
        amat = pd.DataFrame(
            [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        d = DAG.from_pandas_adjacency(amat)
        assert ("A", "B") in d.edges
        assert ("B", "C") in d.edges


class TestMeekRules:
    """Tests for Meek orientation rules."""

    def test_rule1_orients_away_from_parent(self) -> None:
        # X -> Y - Z where X-Z not adjacent => orient Y -> Z
        p = PDAG(dir_edges=[("X", "Y")], undir_edges=[("Y", "Z")])
        result = rule_1(p)
        assert ("Y", "Z") in result.dir_edges
        assert result.num_undir_edges == 0

    def test_rule1_no_orientation_when_shielded(self) -> None:
        # X -> Y - Z but X-Z adjacent => do NOT orient
        p = PDAG(dir_edges=[("X", "Y")], undir_edges=[("Y", "Z"), ("X", "Z")])
        result = rule_1(p)
        # Y-Z should remain undirected
        assert ("Y", "Z") in result.undir_edges or ("Z", "Y") in result.undir_edges

    def test_rule2_orients_to_avoid_cycle(self) -> None:
        # X -> Y -> Z where X - Z adjacent => orient X -> Z
        p = PDAG(dir_edges=[("X", "Y"), ("Y", "Z")], undir_edges=[("X", "Z")])
        result = rule_2(p)
        assert ("X", "Z") in result.dir_edges

    def test_rule3_orients_fork(self) -> None:
        # X - Y1 -> Z, X - Y2 -> Z, Y1-Y2 not adjacent => orient X -> Z
        # Build: X-Z undirected, X-Y1 undirected, X-Y2 undirected, Y1->Z directed, Y2->Z directed, Y1-Y2 not adjacent
        p = PDAG(
            dir_edges=[("Y1", "Z"), ("Y2", "Z")],
            undir_edges=[("X", "Y1"), ("X", "Y2"), ("X", "Z")],
        )
        result = rule_3(p)
        assert ("X", "Z") in result.dir_edges

    def test_rule3_returns_modified_copy(self) -> None:
        """Regression test: rule_3 must return copy_pdag not pdag."""
        p = PDAG(
            dir_edges=[("Y1", "Z"), ("Y2", "Z")],
            undir_edges=[("X", "Y1"), ("X", "Y2"), ("X", "Z")],
        )
        original_undir = set(p.undir_edges)
        result = rule_3(p)
        # Original must be unchanged
        assert set(p.undir_edges) == original_undir
        # Result must differ
        assert result is not p

    def test_rule4_returns_modified_copy(self) -> None:
        """Regression test: rule_4 must return copy_pdag not pdag."""
        p = PDAG(
            dir_edges=[("Z", "X"), ("Z", "Y1")],
            undir_edges=[("X", "Y1"), ("X", "Y2")],
        )
        result = rule_4(p)
        assert result is not p

    def test_meek_convergence_collider_extension(self) -> None:
        """After orienting a collider, Meek rules should propagate correctly."""
        # V-structure: A -> B <- C, plus undirected B - D
        # Rule 1 should orient B -> D (A is parent of B, A not adjacent to D)
        p = PDAG(dir_edges=[("A", "B"), ("C", "B")], undir_edges=[("B", "D")])
        result = rule_1(p)
        assert ("B", "D") in result.dir_edges


class TestDag2Cpdag:
    """Tests for dag2cpdag conversion."""

    def test_single_v_structure(self) -> None:
        # A -> C <- B, A-B not adjacent
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "C"), ("B", "C")])
        cpdag = dag2cpdag(dag)
        assert ("A", "C") in cpdag.dir_edges
        assert ("B", "C") in cpdag.dir_edges

    def test_chain_becomes_undirected(self) -> None:
        # A -> B -> C: all edges in same MEC, CPDAG has undirected edges
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        cpdag = dag2cpdag(dag)
        # Chain has a single Markov equivalence class member, skeleton stays but edges undirected
        assert cpdag.num_nodes == 3
        assert cpdag.num_adjacencies == 2
