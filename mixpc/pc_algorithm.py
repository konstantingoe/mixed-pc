"""PC Algorithm implementation for learning causal graphical models.

This module contains:
- PC: Modern efficient implementation with PC stable variant (Colombo & Maathuis 2014)
  and three v-structure determination rules from Ramsey et al. (2016)
"""

from contextlib import suppress
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from .graphs import DAG, GRAPH, PDAG, rule_1, rule_2, rule_3, rule_4
from .independence_tests import CItest, MixedFisherZ
from .prior_knowledge import PriorKnowledge


class LearnAlgo(metaclass=ABCMeta):
    """Abstract base class for all learning algorithms."""

    @property
    @abstractmethod
    def causal_order(self) -> list[str] | None:
        """Return causal order if applicable."""
        pass

    @property
    @abstractmethod
    def adjacency_matrix(self) -> pd.DataFrame:
        """Return adjacency matrix."""
        pass

    @abstractmethod
    def learn_graph(self, data_dict: dict[str, np.ndarray], *args: Any, **kwargs: Any) -> GRAPH:
        """Learn the graph from the data.

        Args:
            data_dict (dict[str, np.ndarray]): Dictionary mapping variable names to data arrays.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            GRAPH: Learned graph object.
        """


class PC(LearnAlgo):
    """PC algorithm with stable variant (Colombo & Maathuis 2014).

    Implements three v-structure determination rules from Ramsey et al. (2016):
    - Conservative: Orient as v-structure only if unanimous across separating sets
    - Majority: Orient if majority of separating sets do not contain the middle node
    - PC-Max: Orient based on highest p-value for independence
    """

    def __init__(self, alpha: float = 0.05, test: type[CItest] = MixedFisherZ) -> None:
        """Initialize PC algorithm.

        Args:
            alpha (float, optional): Significance threshold for independence tests.
                Smaller values result in sparser graphs. Defaults to 0.05.
            test (type[CItest], optional): Conditional independence test class.
                Defaults to MixedFisherZ.
        """
        self.alpha: float = alpha
        self.pdag: PDAG = PDAG()
        self.skel: pd.DataFrame | None = None
        self.sep_sets: dict[tuple[str, str], set[str]] = {}
        self.ci_test = test()
        self.prior: PriorKnowledge | None = None
        self.ci_test_count: int = 0

    def learn_graph(
        self,
        data_dict: dict[str, np.ndarray],
        v_structure_rule: Literal["conservative", "majority", "pc-max"] = "conservative",
        prior_knowledge: PriorKnowledge | None = None,
    ) -> PDAG:
        """Learn causal graph using PC stable algorithm.

        Args:
            data_dict (dict[str, np.ndarray]): Dictionary mapping variable names to data arrays.
            v_structure_rule (Literal["conservative", "majority", "pc-max"], optional):
                v-structure determination rule from Ramsey et al. (2016).
                Defaults to "conservative".
            prior_knowledge (PriorKnowledge, optional): Edge/direction/layering hints
                consulted across all three phases. Defaults to None.

        Returns:
            PDAG: Partially directed acyclic graph.

        Raises:
            ValueError: If v_structure_rule is not recognized.
        """
        if v_structure_rule not in {"conservative", "majority", "pc-max"}:
            raise ValueError(
                f"v_structure_rule must be 'conservative', 'majority', or 'pc-max', got {v_structure_rule}"
            )

        if prior_knowledge is not None:
            prior_knowledge.validate(set(data_dict.keys()))
        self.prior = prior_knowledge
        self.ci_test_count = 0

        # Phase 1: Skeleton learning (PC stable, optionally constrained by prior knowledge)
        self._find_skeleton_stable(data=data_dict, alpha=self.alpha)

        # Pre-orient edges with a uniquely allowed direction (layering / required_directions /
        # forbidden_directions that pin the alternative). Done before v-structure phase so that
        # downstream rules see the constraints as already-decided orientations.
        if self.prior is not None:
            self._apply_prior_orientations()

        # Phase 2: V-structure orientation using chosen rule
        self._orient_v_structures(data=data_dict, alpha=self.alpha, rule=v_structure_rule)

        # Phase 3: Meek rule application for remaining undirected edges
        self._apply_meek_rules()

        # Final pass: if Meek introduced an orientation that prior knowledge forbids, flip it
        # when the reverse is allowed. Both-directions-forbidden edges are left as-is.
        if self.prior is not None:
            self._reconcile_with_prior()

        return self.pdag

    def _find_skeleton_stable(
        self,
        data: dict[str, np.ndarray],
        alpha: float = 0.05,
    ) -> None:
        """Find skeleton using PC stable algorithm (Colombo & Maathuis 2014).

        The stable version ensures deterministic skeleton discovery independent
        of test order.

        Args:
            data (dict[str, np.ndarray]): Dictionary mapping variable names to data arrays.
            alpha (float): Significance threshold for independence tests.
        """
        node_names = sorted(list(data.keys()))
        n_features = len(node_names)

        # Initialize complete undirected graph
        skeleton = pd.DataFrame(
            np.ones((n_features, n_features)) - np.eye(n_features),
            columns=node_names,
            index=node_names,
        )

        # Drop forbidden edges before the first CI test — they are never even considered.
        if self.prior is not None:
            for a, b in self.prior.forbidden_edges:
                skeleton.loc[a, b] = skeleton.loc[b, a] = 0

        self.sep_sets = {}
        d = 0

        # Iterate over conditioning set sizes until no pair can be tested further.
        while True:
            adj_pairs = self._get_adjacent_pairs(skeleton)
            if not adj_pairs:
                break

            # Snapshot adjacencies once per level (PC-stable: neighbors must not change mid-level)
            skeleton_snapshot = skeleton.copy()

            any_test_possible = False
            for i, j in adj_pairs:
                if self._try_separate_pair(
                    i, j, d=d, data=data, alpha=alpha,
                    skeleton=skeleton, skeleton_snapshot=skeleton_snapshot,
                ):
                    any_test_possible = True

            if not any_test_possible:
                break
            d += 1

        self.skel = skeleton
        self.pdag = PDAG.from_pandas_adjacency(skeleton)

    def _try_separate_pair(
        self,
        i: str,
        j: str,
        *,
        d: int,
        data: dict[str, np.ndarray],
        alpha: float,
        skeleton: pd.DataFrame,
        skeleton_snapshot: pd.DataFrame,
    ) -> bool:
        """Test the (i, j) pair at conditioning-set size ``d``; mutates ``skeleton`` if separated.

        Returns ``True`` when a CI test was actually attempted at this level (so the caller
        knows to keep iterating with a larger ``d``).
        """
        if skeleton.loc[i, j] == 0:
            return False
        if self.prior is not None and self.prior.is_required_edge(i, j):
            return False

        candidate_neighbors = self._candidate_separators(i, j, skeleton_snapshot)
        if self.prior is not None:
            candidate_neighbors = set(
                self.prior.filter_separating_set(i, j, list(candidate_neighbors))
            )
        if len(candidate_neighbors) < d:
            return False

        for sep_set_subset in combinations(sorted(candidate_neighbors), d):
            sep_set_list = list(sep_set_subset)
            if not sep_set_list:
                _, p_value = self.ci_test.test(x_data=data[i], y_data=data[j])
            else:
                z_data = np.concatenate([data[node] for node in sep_set_list], axis=1)
                _, p_value = self.ci_test.test(x_data=data[i], y_data=data[j], z_data=z_data)
            self.ci_test_count += 1

            if p_value >= alpha:
                skeleton.loc[i, j] = skeleton.loc[j, i] = 0
                self.sep_sets[(i, j)] = set(sep_set_list)
                self.sep_sets[(j, i)] = set(sep_set_list)
                return True
        return True

    def _candidate_separators(
        self, i: str, j: str, skeleton_snapshot: pd.DataFrame
    ) -> set[str]:
        """Union of neighbors of ``i`` and ``j`` (excluding each other) in the snapshot."""
        cols = skeleton_snapshot.columns
        row_i = skeleton_snapshot.loc[i].to_numpy()
        row_j = skeleton_snapshot.loc[j].to_numpy()
        neighbors_i = {str(cols[idx]) for idx, val in enumerate(row_i) if val == 1 and cols[idx] != j}
        neighbors_j = {str(cols[idx]) for idx, val in enumerate(row_j) if val == 1 and cols[idx] != i}
        return neighbors_i | neighbors_j

    def _get_adjacent_pairs(self, skeleton: pd.DataFrame) -> list[tuple[str, str]]:
        """Get all adjacent pairs in the skeleton.

        Args:
            skeleton (pd.DataFrame): Adjacency matrix of skeleton.

        Returns:
            list[tuple[str, str]]: List of adjacent node pairs (sorted).
        """
        pairs = []
        nodes = sorted(skeleton.columns.tolist())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if skeleton.iloc[i, j] == 1:
                    pairs.append((nodes[i], nodes[j]))
        return pairs

    def _find_unshielded_triples(self, pdag: PDAG) -> list[tuple[str, str, str]]:
        """Find all unshielded triples in the graph.

        An unshielded triple is (i, j, k) where i-j-k is a path and i,k are not adjacent.

        Args:
            pdag (PDAG): Current partially directed graph.

        Returns:
            list[tuple[str, str, str]]: List of unshielded triples (i, j, k).
        """
        triples = []
        for j in pdag.nodes:
            neighbors_j = pdag.undir_neighbors(j)
            if len(neighbors_j) < 2:
                continue

            # Find all pairs of neighbors
            for i, k in combinations(sorted(neighbors_j), 2):
                # Check if i and k are not adjacent (unshielded)
                if not pdag.is_adjacent(i, k):
                    triples.append((i, j, k))
        return triples

    def _orient_v_structures(
        self,
        data: dict[str, np.ndarray],
        alpha: float = 0.05,
        rule: Literal["conservative", "majority", "pc-max"] = "conservative",
    ) -> None:
        """Determine and orient v-structures using specified rule.

        Implements three rules from Ramsey et al. (2016) for v-structure discovery.

        Args:
            data (dict[str, np.ndarray]): Dictionary mapping variable names to data arrays.
            alpha (float): Significance threshold.
            rule (Literal["conservative", "majority", "pc-max"]): v-structure rule.
        """
        pdag = self.pdag.copy()
        unshielded_triples = self._find_unshielded_triples(pdag)

        for i, j, k in unshielded_triples:
            # Get all potential separating sets for (i, k)
            potential_sep_sets = self._get_potential_separating_sets(i, k, pdag, data)

            # Determine if this is a v-structure based on chosen rule.
            if rule == "conservative":
                is_v_structure = self._conservative_v_structure_rule(i, j, k, potential_sep_sets, alpha)
            elif rule == "majority":
                is_v_structure = self._majority_v_structure_rule(i, j, k, potential_sep_sets, alpha)
            elif rule == "pc-max":
                is_v_structure = self._pc_max_v_structure_rule(i, j, k, potential_sep_sets, alpha)
            else:
                raise ValueError(f"Unknown rule: {rule}")

            if is_v_structure:
                # Skip when the proposed v-structure conflicts with prior knowledge: orienting
                # i -> j (or k -> j) is forbidden — for example, j is in an earlier layer.
                if self.prior is not None and (
                    self.prior.is_forbidden_direction(i, j) or self.prior.is_forbidden_direction(k, j)
                ):
                    continue

                # Orient as v-structure: i -> j <- k (only if edges are still undirected)
                try:
                    if pdag.is_adjacent(i, j) and ((i, j) in pdag.undir_edges or (j, i) in pdag.undir_edges):
                        pdag.undir_to_dir_edge(tail=i, head=j)
                    if pdag.is_adjacent(k, j) and ((k, j) in pdag.undir_edges or (j, k) in pdag.undir_edges):
                        pdag.undir_to_dir_edge(tail=k, head=j)
                except AssertionError:
                    # Edge may have been already oriented - skip
                    pass

        self.pdag = pdag

    def _get_potential_separating_sets(
        self, i: str, k: str, pdag: PDAG, data: dict[str, np.ndarray]
    ) -> list[tuple[set[str], float]]:
        """Get all potential separating sets for a pair of nodes with p-values.

        Args:
            i (str): First node.
            k (str): Second node.
            pdag (PDAG): Current graph.
            data (dict[str, np.ndarray]): Dictionary mapping variable names to data arrays.

        Returns:
            list[tuple[set[str], float]]: List of (separating set, p-value) tuples.
        """
        potential_sets = []

        # Get neighbors of both nodes
        neighbors_i = pdag.neighbors(i)
        neighbors_k = pdag.neighbors(k)

        # Get all combinations of neighbors (excluding i and k)
        all_neighbors = neighbors_i.union(neighbors_k)
        all_neighbors.discard(i)
        all_neighbors.discard(k)

        # Layering: drop "future" nodes from the conditioning pool to keep v-structure
        # decisions consistent with the skeleton phase.
        if self.prior is not None:
            all_neighbors = set(self.prior.filter_separating_set(i, k, list(all_neighbors)))

        # Test all possible subsets
        for r in range(len(all_neighbors) + 1):
            for sep_set in combinations(sorted(all_neighbors), r):
                sep_set_list = list(sep_set)

                # Perform independence test
                if not sep_set_list:
                    _, p_value = self.ci_test.test(x_data=data[i], y_data=data[k])
                else:
                    z_data = np.concatenate(
                        [data[node] for node in sep_set_list],
                        axis=1,
                    )
                    _, p_value = self.ci_test.test(
                        x_data=data[i],
                        y_data=data[k],
                        z_data=z_data,
                    )
                self.ci_test_count += 1

                potential_sets.append((set(sep_set), p_value))

        return potential_sets

    def _conservative_v_structure_rule(
        self,
        i: str,
        j: str,
        k: str,
        potential_sep_sets: list[tuple[set[str], float]],
        alpha: float,
    ) -> bool:
        """Conservative rule: Orient as v-structure only if unanimous.

        A v-structure is oriented only if all separating sets that make (i,k)
        independent do NOT contain j.

        Args:
            i (str): First parent.
            j (str): Middle node.
            k (str): Second parent.
            potential_sep_sets (list[tuple[set[str], float]]): Potential separating sets.
            alpha (float): Significance threshold.

        Returns:
            bool: True if v-structure should be oriented.
        """
        # Find all separating sets that render i and k independent
        independent_sep_sets = [sep_set for sep_set, p_value in potential_sep_sets if p_value >= alpha]

        if not independent_sep_sets:
            # If no separating set makes i,k independent, it's a v-structure
            return True

        # Orient only when every separating set excludes the middle node.
        return all(j not in sep_set for sep_set in independent_sep_sets)

    def _majority_v_structure_rule(
        self,
        i: str,
        j: str,
        k: str,
        potential_sep_sets: list[tuple[set[str], float]],
        alpha: float,
    ) -> bool:
        """Majority rule: Orient if majority of separating sets do not contain j.

        Args:
            i (str): First parent.
            j (str): Middle node.
            k (str): Second parent.
            potential_sep_sets (list[tuple[set[str], float]]): Potential separating sets.
            alpha (float): Significance threshold.

        Returns:
            bool: True if v-structure should be oriented.
        """
        # Find all separating sets that render i and k independent
        independent_sep_sets = [sep_set for sep_set, p_value in potential_sep_sets if p_value >= alpha]

        if not independent_sep_sets:
            # If no separating set makes i,k independent, it's a v-structure
            return True

        # Count how many separating sets do NOT contain j
        count_without_j = sum(1 for sep_set in independent_sep_sets if j not in sep_set)

        # Orient as v-structure if majority do not contain j
        return count_without_j > len(independent_sep_sets) / 2

    def _pc_max_v_structure_rule(
        self,
        i: str,
        j: str,
        k: str,
        potential_sep_sets: list[tuple[set[str], float]],
        alpha: float,
    ) -> bool:
        """PC-Max rule: Orient based on highest p-value for independence.

        Compares the highest p-value obtained when conditioning on sets excluding j
        with the highest p-value when including sets with j. Orients as v-structure
        if independence is more likely with j excluded.

        Args:
            i (str): First parent.
            j (str): Middle node.
            k (str): Second parent.
            potential_sep_sets (list[tuple[set[str], float]]): Potential separating sets.
            alpha (float): Significance threshold (not directly used in this rule).

        Returns:
            bool: True if v-structure should be oriented.
        """
        # Separate p-values by whether j is in the separating set
        p_values_without_j = [p for sep_set, p in potential_sep_sets if j not in sep_set]
        p_values_with_j = [p for sep_set, p in potential_sep_sets if j in sep_set]

        # Get maximum p-values from each group
        max_p_without_j = max(p_values_without_j) if p_values_without_j else 0
        max_p_with_j = max(p_values_with_j) if p_values_with_j else 0

        # Orient as v-structure if independence is more likely without j
        return max_p_without_j > max_p_with_j

    def _apply_prior_orientations(self) -> None:
        """Orient every undirected edge whose direction is uniquely fixed by prior knowledge.

        Runs after skeleton discovery and before v-structure orientation. Layering between
        layers, explicit ``required_directions``, and ``forbidden_directions`` that pin the
        alternative all flow through ``PriorKnowledge.required_direction_for``.
        """
        assert self.prior is not None
        pdag = self.pdag.copy()
        for i, j in list(pdag.undir_edges):
            forced = self.prior.required_direction_for(i, j)
            if forced is None:
                continue
            tail, head = forced
            # Edge may have been oriented by an earlier iteration of this loop — ignore.
            with suppress(AssertionError):
                pdag.undir_to_dir_edge(tail=tail, head=head)
        self.pdag = pdag

    def _reconcile_with_prior(self) -> None:
        """Flip directed edges Meek introduced in a forbidden direction when the reverse is allowed.

        If both directions are forbidden, the edge is left untouched: the user contradicted
        themselves about an edge that PC nonetheless found in the skeleton, and silently
        rewriting it is worse than surfacing the inconsistency.
        """
        assert self.prior is not None
        pdag = self.pdag.copy()
        flipped = False
        for tail, head in list(pdag.dir_edges):
            if not self.prior.is_forbidden_direction(tail, head):
                continue
            if self.prior.is_forbidden_direction(head, tail):
                continue
            # Use the public remove + private add to swap orientation in place.
            pdag.remove_edge(tail, head)
            pdag._add_dir_edge(head, tail)
            flipped = True
        if flipped:
            self.pdag = pdag

    def _apply_meek_rules(self) -> None:
        """Apply Meek rules to orient remaining undirected edges.

        Implements rules R1-R4 from Meek (1995) to maximize the number of
        directed edges consistent with acyclicity.
        """
        pdag = self.pdag.copy()

        # Apply rules until convergence to maximize orientations.
        while True:
            before = (set(pdag.undir_edges), set(pdag.dir_edges))
            pdag = rule_1(pdag=pdag)
            pdag = rule_2(pdag=pdag)
            pdag = rule_3(pdag=pdag)
            pdag = rule_4(pdag=pdag)
            after = (set(pdag.undir_edges), set(pdag.dir_edges))
            if after == before:
                break

        self.pdag = pdag

    @property
    def skeleton(self) -> pd.DataFrame:
        """Return the underlying skeleton as adjacency matrix.

        Returns:
            pd.DataFrame: Adjacency matrix of the skeleton.

        Raises:
            ValueError: If skeleton has not been learned yet.
        """
        if self.skel is None:
            raise ValueError("Skeleton not learned yet.")
        return self.skel

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Return the learned PDAG as adjacency matrix.

        Returns:
            pd.DataFrame: Adjacency matrix of the PDAG.
                - A[i,j]=1, A[j,i]=0: directed edge i→j
                - A[i,j]=1, A[j,i]=1: undirected edge i—j
                - A[i,j]=0, A[j,i]=0: no edge
        """
        return self.pdag.adjacency_matrix

    @property
    def causal_order(self) -> list[str] | None:
        """Return causal order if PDAG is fully directed (DAG).

        Returns:
            list[str] | None: Causal order if PDAG is a DAG, None otherwise.
        """
        if self.pdag.num_undir_edges == 0:
            dag = DAG(nodes=self.pdag.nodes, edges=self.pdag.dir_edges)
            return dag.causal_order
        return None
