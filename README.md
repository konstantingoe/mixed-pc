# my_package: A Python project template

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://github.com/your-org/your-repo)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](https://github.com/your-org/your-repo/actions)

> **This is a Python project template.** See the checklist below before developing.

---

## After cloning — checklist

Complete these steps before writing any code.

- [ ] **Rename the package directory** — rename `graph_models/` to your package name (e.g. `my_package/`)
- [ ] **Update `pyproject.toml`** — change `name`, `description`, `authors`, `maintainers`, `keywords`, `classifiers`, and the `[project.urls]` section
- [ ] **Update `[tool.hatch.version]`** — change `path` to point to your new package's `__init__.py`
- [ ] **Update `[tool.hatch.build.targets.wheel]`** — change `packages` to your new package name
- [ ] **Update `[tool.hatch.build.targets.sdist]`** — remove the `venv_graph_models/` entry and replace with your venv name if needed
- [ ] **Reset the version** — set `__version__` in `your_package/__init__.py` to `"0.1.0"` (or your starting version)
- [ ] **Update `mkdocs.yml`** — set `site_name`, `repo_url`, and `site_url`
- [ ] **Update `docs/index.md`** — replace the placeholder content with your package's overview and quick-start guide
- [ ] **Update `docs/reference.md`** — point the `:::` autodoc directives at your modules
- [ ] **Update this README** — replace all placeholder text, links, and the badges at the top
- [ ] **Update `.pre-commit-config.yaml`** — review hook versions and add/remove hooks as needed
- [ ] **Set up the remote** — `git remote set-url origin https://github.com/your-org/your-repo.git`
- [ ] **Create and push the initial commit** — `git add -A && git commit -m "chore: init from template" && git push -u origin main`

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Development](#development)
- [License](#license)

## Overview

<!-- TODO: replace with a description of your project -->

This repository ships a ready-to-use Python project skeleton with:

- **Hatchling** build backend and `pyproject.toml`-only configuration
- **uv** for fast, reproducible dependency management
- **ruff** for linting and formatting (Google-style docstrings enforced)
- **mypy** for static type checking (strict mode)
- **pytest + pytest-cov** for testing with coverage
- **pre-commit** hooks (ruff, mypy, bandit security scan, uv-based pip-compile on push)
- **MkDocs + Material** for documentation with versioning via `mike`

The example package (`graph_models`) provides an undirected-graph data structure
(`UGRAPH`) to illustrate the intended layout.  Replace it with your own code.

## Installation

### From source

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
pip install -e .
```

### Development installation

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
make sync-venv
```

`make sync-venv` creates a virtual environment, installs all pinned dev
dependencies via `uv pip sync`, and installs the package in editable mode.

## Quick Start

```python
from graph_models import UGRAPH  # replace with your package

g = UGRAPH(nodes=["A", "B", "C"], edges=[("A", "B"), ("A", "C")])
print(g.num_nodes)   # 3
print(g.num_edges)   # 2
print(g.adjacency_matrix)
```

## Project Structure

```
.
├── graph_models/          # → rename to your package
│   ├── __init__.py
│   └── graphs.py
├── tests/
│   └── test_graphs.py
├── docs/
│   ├── index.md
│   └── reference.md
├── pyproject.toml         # build, deps, tool config
├── Makefile               # sync-venv, requirements, test, precommit
├── requirements.txt       # pinned runtime deps (auto-generated)
├── requirements_dev.txt   # pinned dev deps (auto-generated)
├── mkdocs.yml
└── .pre-commit-config.yaml
```

## Testing

```bash
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov=graph_models --cov-report=html
```

Or via Make:

```bash
make test
```

## Development

### Managing dependencies

Add a dependency to `pyproject.toml`, then regenerate lock files:

```bash
make requirements        # re-pin without upgrading existing deps
make update-requirements # re-pin and upgrade all deps
make sync-venv           # apply the lock files to the venv
```

### Pre-commit hooks

```bash
pre-commit install          # enable hooks (run automatically on git commit/push)
pre-commit run --all-files  # run all hooks manually
```

### Building the documentation locally

```bash
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build   # static site in site/
```

### Deploying docs to GitHub Pages

Versioned docs are published via [mike](https://github.com/jimporter/mike) to the `gh-pages` branch:

```bash
mike deploy --push --update-aliases 0.1.0 latest   # publish a version
mike set-default --push latest                      # set the default (once)
```

> Enable GitHub Pages in your repo settings (Settings → Pages → Source: `gh-pages` branch) before deploying.

### Linting and type checking

```bash
ruff check graph_models/     # linting
mypy graph_models/           # static type checking
```

## License

MIT License — see [LICENSE](LICENSE) for details.
