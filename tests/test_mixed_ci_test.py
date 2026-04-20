"""Tests and simulation for MixedFisherZ and correlation utilities.

Covers:
- Unit tests for correlations.py (polychoric, polyserial, adhoc_polyserial, f_hat)
- Unit tests for MixedFisherZ independence test
- End-to-end simulation: PC with MixedFisherZ on mixed continuous/ordinal data
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module loading
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
_pc_mod = _load_module("pc_algorithm", "pc_algorithm.py")

PolychoricCorrelation = _corr_mod.PolychoricCorrelation
PolyserialCorrelation = _corr_mod.PolyserialCorrelation
adhoc_polyserial = _corr_mod.adhoc_polyserial
f_hat = _corr_mod.f_hat
spearman = _corr_mod.spearman
MixedFisherZ = _itest_mod.MixedFisherZ
PC = _pc_mod.PC


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

def _make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ordinal(cont: np.ndarray, n_cats: int = 4) -> np.ndarray:
    """Discretise a continuous array into n_cats ordinal categories."""
    quantiles = np.linspace(0, 100, n_cats + 1)[1:-1]
    thresholds = np.percentile(cont, quantiles)
    return np.searchsorted(thresholds, cont).astype(float)


# ---------------------------------------------------------------------------
# Tests: f_hat (nonparanormal transform)
# ---------------------------------------------------------------------------

class TestFHat:
    def test_output_length(self) -> None:
        rng = _make_rng()
        x = rng.normal(size=200)
        assert f_hat(x).shape == x.shape

    def test_unit_variance(self) -> None:
        rng = _make_rng()
        x = rng.normal(size=500)
        out = f_hat(x)
        assert abs(np.std(out, ddof=1) - 1.0) < 1e-10

    def test_monotone(self) -> None:
        """f_hat is weakly monotone: sorting by x yields non-decreasing transformed values.

        Winsorization clips the most extreme ranks to the same threshold, so
        ties at the tails are expected (non-strictly monotone at boundaries).
        """
        rng = _make_rng()
        x = rng.normal(size=300)
        out = f_hat(x)
        sorted_out = out[np.argsort(x)]
        assert np.all(np.diff(sorted_out) >= 0)

    def test_raises_on_small_input(self) -> None:
        with pytest.raises(ValueError):
            f_hat(np.array([1.0]))

    def test_raises_on_2d_input(self) -> None:
        with pytest.raises(ValueError):
            f_hat(np.ones((10, 2)))


# ---------------------------------------------------------------------------
# Tests: PolychoricCorrelation
# ---------------------------------------------------------------------------

class TestPolychoricCorrelation:
    """Tests for polychoric correlation estimator."""

    @pytest.fixture(scope="class")
    def corr_pair(self):
        """Two ordinal variables with known positive correlation."""
        rng = _make_rng(0)
        n = 2000
        latent_x = rng.normal(size=n)
        latent_y = 0.6 * latent_x + np.sqrt(1 - 0.36) * rng.normal(size=n)
        x = _ordinal(latent_x, n_cats=4)
        y = _ordinal(latent_y, n_cats=4)
        return x, y, 0.6  # true correlation

    def test_positive_correlation_sign(self, corr_pair) -> None:
        x, y, _ = corr_pair
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert rho > 0

    def test_correlation_in_range(self, corr_pair) -> None:
        x, y, _ = corr_pair
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert -1.0 <= rho <= 1.0

    def test_estimate_close_to_truth_brent(self, corr_pair) -> None:
        x, y, true_rho = corr_pair
        rho = PolychoricCorrelation(solver="brent").fit(x, y).correlation
        assert abs(rho - true_rho) < 0.08

    def test_estimate_close_to_truth_newton(self, corr_pair) -> None:
        x, y, true_rho = corr_pair
        rho = PolychoricCorrelation(solver="newton").fit(x, y).correlation
        assert abs(rho - true_rho) < 0.08

    def test_brent_and_newton_agree(self, corr_pair) -> None:
        x, y, _ = corr_pair
        r_brent = PolychoricCorrelation(solver="brent").fit(x, y).correlation
        r_newton = PolychoricCorrelation(solver="newton").fit(x, y).correlation
        assert abs(r_brent - r_newton) < 0.02

    def test_symmetric(self, corr_pair) -> None:
        x, y, _ = corr_pair
        rho_xy = PolychoricCorrelation().fit(x, y).correlation
        rho_yx = PolychoricCorrelation().fit(y, x).correlation
        assert abs(rho_xy - rho_yx) < 1e-6

    def test_negative_correlation(self) -> None:
        rng = _make_rng(1)
        n = 2000
        latent_x = rng.normal(size=n)
        latent_y = -0.7 * latent_x + np.sqrt(1 - 0.49) * rng.normal(size=n)
        x = _ordinal(latent_x, n_cats=3)
        y = _ordinal(latent_y, n_cats=3)
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert rho < 0

    def test_invalid_solver_raises(self) -> None:
        with pytest.raises(ValueError, match="solver"):
            PolychoricCorrelation(solver="bad")  # type: ignore[arg-type]

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            _ = PolychoricCorrelation().correlation


# ---------------------------------------------------------------------------
# Tests: PolyserialCorrelation
# ---------------------------------------------------------------------------

class TestPolyserialCorrelation:
    """Tests for polyserial correlation estimator."""

    @pytest.fixture(scope="class")
    def mixed_pair(self):
        """Continuous X, ordinal Y with known positive correlation."""
        rng = _make_rng(2)
        n = 2000
        x_cont = rng.normal(size=n)
        latent_y = 0.5 * x_cont + np.sqrt(1 - 0.25) * rng.normal(size=n)
        y_ord = _ordinal(latent_y, n_cats=4)
        return x_cont, y_ord, 0.5

    def test_positive_sign(self, mixed_pair) -> None:
        x, y, _ = mixed_pair
        rho = PolyserialCorrelation().fit(x, y).correlation
        assert rho > 0

    def test_in_range(self, mixed_pair) -> None:
        x, y, _ = mixed_pair
        rho = PolyserialCorrelation().fit(x, y).correlation
        assert -1.0 <= rho <= 1.0

    def test_close_to_truth(self, mixed_pair) -> None:
        x, y, true_rho = mixed_pair
        rho = PolyserialCorrelation().fit(x, y).correlation
        assert abs(rho - true_rho) < 0.12

    def test_symmetric_argument_order(self, mixed_pair) -> None:
        """fit(x_cont, y_ord) and fit(y_ord, x_cont) must give same result."""
        x, y, _ = mixed_pair
        r1 = PolyserialCorrelation().fit(x, y).correlation
        r2 = PolyserialCorrelation().fit(y, x).correlation
        assert abs(r1 - r2) < 1e-10

    def test_both_continuous_raises(self) -> None:
        rng = _make_rng()
        x = rng.normal(size=100)
        y = rng.normal(size=100)
        with pytest.raises(ValueError, match="continuous"):
            PolyserialCorrelation().fit(x, y)

    def test_both_ordinal_raises(self) -> None:
        rng = _make_rng()
        x = _ordinal(rng.normal(size=100), n_cats=4)
        y = _ordinal(rng.normal(size=100), n_cats=4)
        with pytest.raises(ValueError, match="ordinal"):
            PolyserialCorrelation().fit(x, y)


# ---------------------------------------------------------------------------
# Tests: adhoc_polyserial dispatcher
# ---------------------------------------------------------------------------

class TestAdhocPolyserial:
    """Tests for the adhoc_polyserial dispatcher."""

    def test_cont_cont_returns_npn_spearman(self) -> None:
        rng = _make_rng()
        x = rng.normal(size=500)
        y = 0.5 * x + rng.normal(size=500)
        r = adhoc_polyserial(x, y)
        assert -1.0 <= r <= 1.0
        assert r > 0  # should reflect positive association

    def test_ord_ord_delegates_to_polychoric(self) -> None:
        rng = _make_rng(3)
        n = 1000
        latent = rng.normal(size=(n, 2))
        latent[:, 1] = 0.6 * latent[:, 0] + np.sqrt(1 - 0.36) * rng.normal(size=n)
        x = _ordinal(latent[:, 0], n_cats=3)
        y = _ordinal(latent[:, 1], n_cats=3)
        r = adhoc_polyserial(x, y)
        assert r > 0.3  # should detect positive association

    def test_mixed_delegates_to_polyserial(self) -> None:
        rng = _make_rng(4)
        n = 1000
        x = rng.normal(size=n)
        y = _ordinal(0.5 * x + rng.normal(size=n), n_cats=4)
        r = adhoc_polyserial(x, y)
        assert r > 0

    def test_result_clipped(self) -> None:
        rng = _make_rng()
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        r = adhoc_polyserial(x, y, max_cor=0.9)
        assert abs(r) <= 0.9


# ---------------------------------------------------------------------------
# Tests: MixedFisherZ CI test
# ---------------------------------------------------------------------------

class TestMixedFisherZ:
    """Tests for the MixedFisherZ conditional independence test."""

    def test_independent_vars_high_pvalue(self) -> None:
        """Truly independent continuous variables should yield high p-value."""
        rng = _make_rng(5)
        n = 1000
        x = rng.normal(size=(n, 1))
        y = rng.normal(size=(n, 1))
        _, p = MixedFisherZ().test(x, y)
        assert p > 0.05

    def test_dependent_vars_low_pvalue(self) -> None:
        """Strongly correlated variables should yield low p-value."""
        rng = _make_rng(6)
        n = 1000
        x = rng.normal(size=(n, 1))
        y = 0.9 * x + 0.1 * rng.normal(size=(n, 1))
        _, p = MixedFisherZ().test(x, y)
        assert p < 0.01

    def test_conditional_independence(self) -> None:
        """X ⟂ Y | Z when X->Z->Y: p-value should be high given Z."""
        rng = _make_rng(7)
        n = 2000
        z = rng.normal(size=(n, 1))
        x = z + 0.3 * rng.normal(size=(n, 1))
        y = z + 0.3 * rng.normal(size=(n, 1))
        _, p_marginal = MixedFisherZ().test(x, y)
        _, p_given_z = MixedFisherZ().test(x, y, z_data=z)
        assert p_marginal < 0.05   # marginally dependent
        assert p_given_z > 0.05   # conditionally independent

    def test_mixed_marginal(self) -> None:
        """Test on continuous-ordinal pair without conditioning."""
        rng = _make_rng(8)
        n = 1000
        x_cont = rng.normal(size=(n, 1))
        y_ord = _ordinal(0.6 * x_cont.ravel() + rng.normal(size=n), n_cats=4).reshape(n, 1)
        _, p = MixedFisherZ().test(x_cont, y_ord)
        assert p < 0.01  # should detect association

    def test_mixed_conditional(self) -> None:
        """Conditional test on mixed data: X_cont ⟂ Y_ord | Z after accounting for Z."""
        rng = _make_rng(9)
        n = 2000
        z = rng.normal(size=(n, 1))
        x = z + 0.3 * rng.normal(size=(n, 1))
        y_lat = (z + 0.3 * rng.normal(size=(n, 1))).ravel()
        y_ord = _ordinal(y_lat, n_cats=4).reshape(n, 1)
        _, p_marginal = MixedFisherZ().test(x, y_ord)
        _, p_given_z = MixedFisherZ().test(x, y_ord, z_data=z)
        assert p_marginal < 0.05
        assert p_given_z > 0.05

    def test_returns_float_tuple(self) -> None:
        rng = _make_rng()
        x = rng.normal(size=(200, 1))
        y = rng.normal(size=(200, 1))
        result = MixedFisherZ().test(x, y)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_pvalue_in_unit_interval(self) -> None:
        rng = _make_rng()
        x = rng.normal(size=(200, 1))
        y = rng.normal(size=(200, 1))
        _, p = MixedFisherZ().test(x, y)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# End-to-end simulation: PC with MixedFisherZ on mixed data (task 7)
# ---------------------------------------------------------------------------

class TestMixedPCSimulation:
    """End-to-end simulation: PC algorithm with MixedFisherZ on mixed data.

    Ground truth: X0 -> X2 <- X1, X2 -> X3
    X0, X1: continuous (Gaussian)
    X2: continuous (Gaussian, child of X0 and X1)
    X3: ordinal (discretised child of X2)
    """

    @pytest.fixture(scope="class")
    def mixed_data(self):
        rng = _make_rng(42)
        n = 3000
        x0 = rng.normal(size=(n, 1))
        x1 = rng.normal(size=(n, 1))
        x2 = x0 + x1 + 0.5 * rng.normal(size=(n, 1))
        x3 = _ordinal((x2.ravel() + 0.3 * rng.normal(size=n)), n_cats=5).reshape(n, 1)
        return {"X0": x0, "X1": x1, "X2": x2, "X3": x3}

    def test_skeleton_correct(self, mixed_data) -> None:
        """Mixed PC should find the correct 3-edge skeleton."""
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pc._find_skeleton_stable(data=mixed_data, alpha=0.05)
        skel = pc.skel
        assert skel is not None
        assert skel.loc["X0", "X2"] == 1
        assert skel.loc["X1", "X2"] == 1
        assert skel.loc["X2", "X3"] == 1
        assert skel.loc["X0", "X1"] == 0

    def test_collider_oriented(self, mixed_data) -> None:
        """Mixed PC should orient X0->X2<-X1 as a v-structure."""
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pdag = pc.learn_graph(mixed_data, v_structure_rule="conservative")
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges

    def test_meek_propagates_x2_x3(self, mixed_data) -> None:
        """After orienting the collider, Meek R1 should orient X2->X3."""
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pdag = pc.learn_graph(mixed_data, v_structure_rule="conservative")
        assert ("X2", "X3") in pdag.dir_edges

    @pytest.mark.parametrize("rule", ["conservative", "majority", "pc-max"])
    def test_all_rules_find_collider(self, mixed_data, rule: str) -> None:
        """All three orientation rules should identify the v-structure."""
        pc = PC(alpha=0.05, test=MixedFisherZ)
        pdag = pc.learn_graph(mixed_data, v_structure_rule=rule)
        assert ("X0", "X2") in pdag.dir_edges
        assert ("X1", "X2") in pdag.dir_edges
