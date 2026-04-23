"""Prior knowledge for the PC algorithm.

Bundles user-provided constraints — required/forbidden edges, required/forbidden
directions, and a partial temporal layering — into a single normalized object
that the PC algorithm consults during skeleton, v-structure, and Meek phases.

Layering is a partial temporal order: each layer is a set of nodes whose mutual
order is unknown, but every node in layer ``k`` precedes every node in layer
``k+1``. This matches use cases such as production-line stations or time-stamped
batches where intra-stage ordering is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


Edge = tuple[str, str]


@dataclass
class PriorKnowledge:
    """User-supplied constraints consumed by :class:`mixpc.pc_algorithm.PC`.

    All edge tuples are ``(tail, head)``. For undirected hints (``required_edges``,
    ``forbidden_edges``) the ordering does not matter — both ``(a, b)`` and
    ``(b, a)`` are treated identically. For directed hints
    (``required_directions``, ``forbidden_directions``) the tuple is read as
    ``tail -> head``.

    Args:
        required_edges: Edges that must appear in the skeleton (undirected
            sense). Skipped during CI testing so they are never removed.
        forbidden_edges: Edges that must not appear in the skeleton. Removed
            from the initial complete graph; never tested.
        required_directions: Edges pinned to a specific orientation
            ``tail -> head``. Implies the edge is in the skeleton.
        forbidden_directions: Orientations ``tail -> head`` that must never
            appear. The undirected edge may still exist.
        layering: Partial temporal order. ``layering[k]`` is the set of nodes
            in stage ``k``; every node in stage ``k`` precedes every node in
            stage ``k+1``. Order within a stage is unknown.
    """

    required_edges: list[Edge] = field(default_factory=list)
    forbidden_edges: list[Edge] = field(default_factory=list)
    required_directions: list[Edge] = field(default_factory=list)
    forbidden_directions: list[Edge] = field(default_factory=list)
    layering: list[list[str]] | None = None

    def __post_init__(self) -> None:
        """Init dataclass."""
        self._required_edges_set: set[frozenset[str]] = {frozenset(e) for e in self.required_edges}
        self._forbidden_edges_set: set[frozenset[str]] = {frozenset(e) for e in self.forbidden_edges}
        self._required_dir_set: set[Edge] = {(e[0], e[1]) for e in self.required_directions}
        self._forbidden_dir_set: set[Edge] = {(e[0], e[1]) for e in self.forbidden_directions}
        self._layer_of: dict[str, int] = {}
        if self.layering is not None:
            for idx, stage in enumerate(self.layering):
                for node in stage:
                    self._layer_of[node] = idx

    def validate(self, nodes: set[str]) -> None:
        """Check internal consistency and that every named node exists in ``nodes``."""
        self._validate_node_membership(nodes)
        self._validate_edge_conflicts()
        if self.layering is not None:
            self._validate_layering()

    def _validate_node_membership(self, nodes: set[str]) -> None:
        all_named = (
            {n for e in self.required_edges for n in e}
            | {n for e in self.forbidden_edges for n in e}
            | {n for e in self.required_directions for n in e}
            | {n for e in self.forbidden_directions for n in e}
            | set(self._layer_of.keys())
        )
        unknown = all_named - nodes
        if unknown:
            raise ValueError(f"Prior knowledge references unknown nodes: {sorted(unknown)}")
        all_edges = (
            self.required_edges + self.forbidden_edges + self.required_directions + self.forbidden_directions
        )
        for e in all_edges:
            if e[0] == e[1]:
                raise ValueError(f"Self-loop in prior knowledge: {e}")

    def _validate_edge_conflicts(self) -> None:
        overlap = self._required_edges_set & self._forbidden_edges_set
        if overlap:
            raise ValueError(f"Edges appear in both required_edges and forbidden_edges: {[tuple(e) for e in overlap]}")
        for tail, head in self._required_dir_set:
            if frozenset((tail, head)) in self._forbidden_edges_set:
                raise ValueError(f"required_direction {(tail, head)} contradicts a forbidden_edge.")
            if (tail, head) in self._forbidden_dir_set:
                raise ValueError(f"required_direction {(tail, head)} contradicts a forbidden_direction.")
            if (head, tail) in self._required_dir_set:
                raise ValueError(
                    f"Conflicting required_directions for the same edge: {(tail, head)} and {(head, tail)}."
                )

    def _validate_layering(self) -> None:
        assert self.layering is not None
        seen: set[str] = set()
        for stage in self.layering:
            dup = seen & set(stage)
            if dup:
                raise ValueError(f"Node(s) appear in multiple layers: {sorted(dup)}")
            seen |= set(stage)
        for tail, head in self._required_dir_set:
            if (
                tail in self._layer_of
                and head in self._layer_of
                and self._layer_of[tail] > self._layer_of[head]
            ):
                raise ValueError(
                    f"required_direction {tail} -> {head} contradicts layering "
                    f"(layer {self._layer_of[tail]} > layer {self._layer_of[head]})."
                )

    # ----- predicates used by PC ------------------------------------------------

    def is_forbidden_edge(self, i: str, j: str) -> bool:
        """Whether the undirected edge ``{i, j}`` is blacklisted."""
        return frozenset((i, j)) in self._forbidden_edges_set

    def is_required_edge(self, i: str, j: str) -> bool:
        """Whether the undirected edge ``{i, j}`` must appear (directly or via a required direction)."""
        if frozenset((i, j)) in self._required_edges_set:
            return True
        return (i, j) in self._required_dir_set or (j, i) in self._required_dir_set

    def is_forbidden_direction(self, tail: str, head: str) -> bool:
        """Whether the orientation ``tail -> head`` is forbidden by any hint or by layering."""
        if (tail, head) in self._forbidden_dir_set:
            return True
        if (head, tail) in self._required_dir_set:
            return True
        return (
            self.layering is not None
            and tail in self._layer_of
            and head in self._layer_of
            and self._layer_of[tail] > self._layer_of[head]
        )

    def required_direction_for(self, i: str, j: str) -> Edge | None:
        """Return the uniquely allowed orientation of edge {i, j}, if any.

        Resolution order: explicit required_direction → layering → forbidden_direction
        leaving exactly one valid side. Returns ``None`` when both orientations are
        permitted or when both are forbidden (caller decides what to do).
        """
        if (i, j) in self._required_dir_set:
            return (i, j)
        if (j, i) in self._required_dir_set:
            return (j, i)
        if self.layering is not None and i in self._layer_of and j in self._layer_of:
            li, lj = self._layer_of[i], self._layer_of[j]
            if li != lj:
                return (i, j) if li < lj else (j, i)
        ij_forbidden = (i, j) in self._forbidden_dir_set
        ji_forbidden = (j, i) in self._forbidden_dir_set
        if ij_forbidden ^ ji_forbidden:
            return (j, i) if ij_forbidden else (i, j)
        return None

    def filter_separating_set(self, i: str, j: str, candidates: list[str]) -> list[str]:
        """Drop conditioning-set candidates that lie in a layer strictly later than max(layer(i), layer(j))."""
        if self.layering is None:
            return list(candidates)
        layer_of = self._layer_of
        if i not in layer_of or j not in layer_of:
            return list(candidates)
        max_layer = max(layer_of[i], layer_of[j])
        return [c for c in candidates if c not in layer_of or layer_of[c] <= max_layer]
