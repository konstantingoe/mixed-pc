# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/konstantingoe/mixed-pc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/konstantingoe/mixed-pc/releases/tag/v0.1.0
