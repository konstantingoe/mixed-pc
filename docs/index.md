# graph_models

<!-- TODO: replace this page with an overview and quick-start for your package -->

This site contains the API documentation for the `graph_models` package.

## Overview

`graph_models` is the example package that ships with this Python project template.
It provides a simple undirected-graph data structure (`UGRAPH`) to illustrate the
intended project layout.  **Replace it with your own code.**

## Quick Start

```python
from graph_models import UGRAPH

g = UGRAPH(nodes=["A", "B", "C"], edges=[("A", "B"), ("A", "C")])
print(g.num_nodes)         # 3
print(g.num_edges)         # 2
print(g.adjacency_matrix)
```

## Installation

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
make sync-venv
```

See the [API Reference](reference.md) for detailed documentation.

---

## Developer Guide

### Pre-commit hooks

The project uses [pre-commit](https://pre-commit.com) to enforce code quality before every commit and push.
Hooks are defined in `.pre-commit-config.yaml` and include:

| Hook | Stage | Purpose |
|------|-------|---------|
| `ruff` | commit | Linting and auto-formatting |
| `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` | commit | General hygiene |
| `mypy` | commit | Static type checking (strict mode) |
| `bandit` | commit | Security vulnerability scanning |
| `pip-compile` (×2) | push | Re-pin `requirements.txt` and `requirements_dev.txt` from `pyproject.toml` |

Install and run all hooks against every file in one step:

```bash
make precommit
```

This runs `pre-commit install` (registers the git hooks) and then `pre-commit run --all-files`.
After that, hooks run automatically on `git commit` and `git push`.

To update hook versions to their latest releases:

```bash
pre-commit autoupdate
```

---

### Documentation (MkDocs)

Docs are built with [MkDocs](https://www.mkdocs.org) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).
API reference pages are auto-generated from source docstrings via the `mkdocstrings` plugin.

**Preview locally:**

```bash
source venv_<your-repo>/bin/activate
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

**Build a static site:**

```bash
mkdocs build          # output goes to site/
```

---

### Deploy docs to GitHub Pages

Versioned docs are managed with [mike](https://github.com/jimporter/mike), which pushes each release to the `gh-pages` branch of your repository.

**Deploy a new version:**

```bash
mike deploy --push --update-aliases <version> latest
# e.g.  mike deploy --push --update-aliases 0.1.0 latest
```

**Set the default version** (only needed once):

```bash
mike set-default --push latest
```

**View available deployed versions:**

```bash
mike list
```

> **Prerequisites:** your GitHub repository must have GitHub Pages enabled (Settings → Pages → Source: `gh-pages` branch).
> The `repo_url` and `site_url` fields in `mkdocs.yml` must point at your actual repository.
