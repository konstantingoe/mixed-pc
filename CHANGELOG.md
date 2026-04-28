# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-28

### Changed
- `PDAG.dir_edges` and `PDAG.undir_edges` now return sorted lists, so iteration
  order is stable across runs regardless of the order in which edges were
  added. Internal membership checks bypass the sort by going through the
  underlying sets, so hot paths (`is_adjacent`, Meek rules) stay O(1) per probe.
- Release workflow split: prerelease tags (`rc`/`alpha`/`beta`/`dev`) publish
  to TestPyPI; stable tags publish to PyPI directly with no cross-dependency.

### Fixed
- README and docs `MixedFisherZ` example: replaced the deterministic
  discretization (`x3 = searchsorted(quantiles_of_x2, x2)`) with a thresholded
  latent variable `z3 = x2 + noise`, matching the generative model the
  polyserial CI test is built on. The published seed now reproducibly
  recovers `X0 → X2 ← X1, X2 → X3`.

## [0.1.0] - 2026-04-23

Initial release.

### Added
- PC-stable skeleton discovery (Colombo & Maathuis 2014).
- V-structure orientation strategies: conservative, majority, and PC-Max (Ramsey et al. 2016).
- Meek rules R1–R4 for maximal orientation of the remaining undirected edges.
- `MixedFisherZ` conditional independence test dispatching to:
  - Nonparanormal Spearman sin-transform for continuous pairs (Liu et al. 2009).
  - Polychoric MLE (Brent or Newton-Fisher scoring) for ordinal pairs (Olsson 1979).
  - Ad-hoc polyserial correlation for mixed pairs (Olsson–Drasgow–Dorans 1982).
- `PriorKnowledge` for user-supplied constraints: required/forbidden edges,
  required/forbidden directions, and partial temporal layering. Integrated into
  skeleton discovery, v-structure orientation, and Meek propagation.
- PEP 561 `py.typed` marker.

[Unreleased]: https://github.com/konstantingoe/mixed-pc/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/konstantingoe/mixed-pc/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/konstantingoe/mixed-pc/releases/tag/v0.1.0
