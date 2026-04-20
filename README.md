# mixpc: Mixed PC Algorithm for Causal Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://konstantingoe.github.io/mixed-pc/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](https://github.com/konstantingoe/mixed-pc/actions)

**mixpc** is a Python library for learning causal graphical models from data with mixed variable types (continuous and ordinal). It implements the PC algorithm with the PC-stable skeleton discovery variant (Colombo & Maathuis 2014) and three v-structure orientation strategies from Ramsey et al. (2016).

---

## Features

- **PC-stable skeleton discovery** — deterministic edge removal order, independent of variable ordering.
- **Three orientation strategies** — conservative, majority, and PC-Max v-structure rules.
- **Mixed data support** — automatic dispatch to the right correlation measure per variable pair:
    - Both continuous → nonparanormal Spearman sin-transform (Liu et al. 2009)
    - Both ordinal → polychoric MLE (Brent or Newton-Fisher solver)
    - Mixed → ad-hoc polyserial correlation
- **Full Meek rule application** — R1–R4 applied to maximally orient remaining undirected edges.

---

## Installation

### From source

```bash
git clone https://github.com/konstantingoe/mixed-pc.git
cd mixed-pc
pip install -e .
```

### Development installation

```bash
git clone https://github.com/konstantingoe/mixed-pc.git
cd mixed-pc
make sync-venv
```

`make sync-venv` creates a virtual environment, installs all pinned dev dependencies via `uv pip sync`, and installs the package in editable mode.

---

## Quick Start

### Continuous data (Fisher Z test)

```python
import numpy as np
from mixpc import PC

rng = np.random.default_rng(42)
n = 2000

# Ground truth: X0 -> X2 <- X1, X2 -> X3
x0 = rng.normal(size=(n, 1))
x1 = rng.normal(size=(n, 1))
x2 = x0 + x1 + 0.5 * rng.normal(size=(n, 1))
x3 = x2 + 0.3 * rng.normal(size=(n, 1))

data = {"X0": x0, "X1": x1, "X2": x2, "X3": x3}

pc = PC(alpha=0.05)
pdag = pc.learn_graph(data, v_structure_rule="conservative")

print("Directed edges:", pdag.dir_edges)
# → [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3')]
```

### Mixed data (continuous + ordinal)

```python
import numpy as np
from mixpc import PC, MixedFisherZ

rng = np.random.default_rng(0)
n = 3000

x0 = rng.normal(size=(n, 1))
x1 = rng.normal(size=(n, 1))
x2 = x0 + x1 + 0.5 * rng.normal(size=(n, 1))

# X3 is ordinal (5 categories)
quantiles = np.percentile(x2, [20, 40, 60, 80])
x3 = np.searchsorted(quantiles, x2).reshape(n, 1).astype(float)

data = {"X0": x0, "X1": x1, "X2": x2, "X3": x3}

pc = PC(alpha=0.05, test=MixedFisherZ)
pdag = pc.learn_graph(data, v_structure_rule="majority")

print("Directed edges:", pdag.dir_edges)
# → [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3')]
```

---

## Orientation strategies

Pass `v_structure_rule` to `learn_graph`:

| Strategy | Behaviour |
|---|---|
| `"conservative"` | Orient as v-structure only if **all** separating sets exclude the middle node |
| `"majority"` | Orient if **majority** of separating sets exclude the middle node |
| `"pc-max"` | Orient based on the highest p-value across separating sets |

---

## Project Structure

```
.
├── mixpc/
│   ├── __init__.py
│   ├── pc_algorithm.py
│   ├── independence_tests.py
│   ├── correlations.py
│   └── graphs.py
├── tests/
│   ├── test_pc_algorithm.py
│   ├── test_mixed_ci_test.py
│   ├── test_graphs.py
│   └── test_benchmark_causallearn.py
├── docs/
│   ├── index.md
│   └── reference.md
├── pyproject.toml
├── Makefile
├── mkdocs.yml
└── .pre-commit-config.yaml
```

---

## Testing

```bash
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov=mixpc --cov-report=html
```

---

## Development

### Managing dependencies

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

### Linting and type checking

```bash
ruff check mixpc/
mypy mixpc/
```

### Documentation

```bash
source venv_mixed-pc/bin/activate
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build   # static site in site/
```

### Deploy to GitHub Pages

```bash
mike deploy --push --update-aliases 0.1.0 latest
mike set-default --push latest   # first time only
```

---

## References

- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search.* MIT Press.
- Colombo, D. & Maathuis, M.H. (2014). Order-independent constraint-based causal structure learning. *JMLR* 15, 3741–3782.
- Ramsey, J., Zhang, J. & Spirtes, P. (2016). Adjacency-faithfulness and conservative causal inference. *UAI*.
- Liu, H., Lafferty, J. & Wasserman, L. (2009). The nonparanormal. *JMLR* 10, 2295–2328.

## License

MIT License — see [LICENSE](LICENSE) for details.
