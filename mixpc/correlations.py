"""Correlation estimation for mixed continuous and ordinal data.

Ported from https://github.com/konstantingoe/mixed-gm (humlpy/correlation.py).

Provides:
* :class:`PolychoricCorrelation` — MLE polychoric for ordinal-ordinal pairs.
* :class:`PolyserialCorrelation` — ad-hoc polyserial for continuous-ordinal pairs.
* :func:`pairwise_latent_correlation` — dispatcher that selects the right estimator by
  variable type.
* :func:`f_hat` — winsorized nonparanormal transformation (Liu et al. 2009).

References:
    Liu, Han, Lafferty, John and Wasserman, Larry. (2009).
    The nonparanormal: Semi-parametric estimation of high dimensional
    undirected graphs. JMLR 10(80), 2295–2328.

    Göbler, Konstantin, Drton, Mathias, Mukherjee, Sach and Miloschewski, Anne.
    (2024). High-dimensional undirected graphical models for arbitrary mixed
    data. Electronic Journal of Statistics 18(1). doi:10.1214/24-EJS2254.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

_CELL_FLOOR: float = 1e-12
_CDF_CAP: float = 8.0  # Φ(±8) is within 1e-15 of {0, 1}; capping ±inf here avoids scipy versions that reject
# infinite inputs to mvn.cdf/mvn.pdf.


def _cap(v: float) -> float:
    if v > _CDF_CAP:
        return _CDF_CAP
    if v < -_CDF_CAP:
        return -_CDF_CAP
    return v


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _to_array(x: object) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D array, got shape {arr.shape}.")
    return arr


def _validate_pair(x: np.ndarray, y: np.ndarray, min_samples: int = 5) -> None:
    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if x.size != y.size:
        raise ValueError(f"Arrays must have the same length, got {x.size} and {y.size}.")
    if x.size < min_samples:
        raise ValueError(f"At least {min_samples} observations required, got {x.size}.")
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        raise ValueError("Input array is entirely NaN.")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        raise ValueError("Input array is constant — correlation is undefined.")


def _validate_ordinal(arr: np.ndarray, name: str = "array") -> None:
    if np.any(np.isinf(arr)):
        raise ValueError(f"Ordinal variable '{name}' contains infinite values.")
    n_unique = len(np.unique(arr[~np.isnan(arr)]))
    if n_unique < 2:
        raise ValueError(f"Ordinal variable '{name}' must have ≥ 2 categories, found {n_unique}.")


def _validate_continuous(arr: np.ndarray, name: str = "array") -> None:
    if np.any(np.isinf(arr)):
        raise ValueError(f"Continuous variable '{name}' contains infinite values.")
    n_nan = int(np.sum(np.isnan(arr)))
    if n_nan > 0:
        logger.warning("Continuous variable '%s' contains %d NaN(s); excluded from rank transform.", name, n_nan)


def _thresholds(disc_var: np.ndarray) -> np.ndarray:
    """Normal-score thresholds for an ordinal variable."""
    n = len(disc_var)
    _, counts = np.unique(disc_var, return_counts=True)
    return np.asarray(stats.norm.ppf(np.concatenate([[0], np.cumsum(counts / n)])))


def _f_hat(x: np.ndarray) -> np.ndarray:
    """Winsorized nonparanormal transformation (Liu et al. 2009)."""
    n = x.shape[0]
    npn_thresh = 1 / (4 * (n**0.25) * np.sqrt(np.pi * np.log(n)))
    ranks = stats.rankdata(x, method="average")
    ecdf_values = np.clip(ranks / (n + 1), npn_thresh, 1 - npn_thresh)
    transformed = np.asarray(stats.norm.ppf(ecdf_values))
    return np.asarray(transformed / np.std(transformed, ddof=1))


def _npn_pearson(cont: np.ndarray, disc: np.ndarray) -> float:
    return float(np.corrcoef(_f_hat(cont), disc)[0, 1])


def _pi_rs(lower: tuple[float, float], upper: tuple[float, float], corr: float) -> float:
    """Bivariate normal rectangle probability."""
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, corr], [corr, 1.0]])
    l1, l2 = _cap(lower[0]), _cap(lower[1])
    u1, u2 = _cap(upper[0]), _cap(upper[1])
    return float(mvn.cdf([u1, u2]) - mvn.cdf([l1, u2]) - mvn.cdf([u1, l2]) + mvn.cdf([l1, l2]))


def _safe_mvn_pdf(mvn: stats.multivariate_normal_frozen, x: np.ndarray) -> float:
    return 0.0 if np.any(np.isinf(x)) else float(mvn.pdf(x))


def _pi_rs_derivative(lower: np.ndarray, upper: np.ndarray, corr: float) -> float:
    """Derivative of bivariate normal rectangle probability w.r.t. corr."""
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, corr], [corr, 1.0]])
    l1, l2 = _cap(float(lower[0])), _cap(float(lower[1]))
    u1, u2 = _cap(float(upper[0])), _cap(float(upper[1]))
    return (
        _safe_mvn_pdf(mvn, np.array([u1, u2]))
        - _safe_mvn_pdf(mvn, np.array([l1, u2]))
        - _safe_mvn_pdf(mvn, np.array([u1, l2]))
        + _safe_mvn_pdf(mvn, np.array([l1, l2]))
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CorrelationMeasure(ABC):
    """Abstract base for latent-Gaussian copula correlation estimators."""

    def __init__(self, max_cor: float = 0.9999) -> None:
        """Init with clip bound max_cor in (0, 1)."""
        if not (0 < max_cor < 1):
            raise ValueError(f"`max_cor` must be in (0, 1), got {max_cor}.")
        self._max_cor = max_cor
        self._correlation: float | None = None

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> CorrelationMeasure:
        """Compute the correlation estimate. Returns self."""

    @property
    def correlation(self) -> float:
        """Estimated correlation in [-max_cor, max_cor]. Raises if not yet fitted."""
        if self._correlation is None:
            raise RuntimeError("Call .fit(x, y) before accessing .correlation.")
        return self._correlation

    @staticmethod
    def _prepare(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
        x_arr = _to_array(x)
        y_arr = _to_array(y)
        _validate_pair(x_arr, y_arr)
        return x_arr, y_arr

    def _clip(self, value: float) -> float:
        return float(np.clip(value, -self._max_cor, self._max_cor))


# ---------------------------------------------------------------------------
# Polychoric correlation
# ---------------------------------------------------------------------------


class PolychoricCorrelation(CorrelationMeasure):
    """MLE polychoric correlation between two ordinal variables.

    Args:
        max_cor: Clip bound for the estimate. Defaults to 0.9999.
        solver: ``"brent"`` (default, faster for ≥4 categories) or
            ``"newton"`` (Fisher scoring, faster for binary/ternary).
        max_iter: Max iterations for Newton solver.
        tol: Convergence tolerance for Newton solver.
    """

    def __init__(
        self,
        max_cor: float = 0.9999,
        solver: Literal["newton", "brent"] = "brent",
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        """Init. solver: 'brent' (default) or 'newton' (Fisher scoring)."""
        super().__init__(max_cor=max_cor)
        if solver not in {"newton", "brent"}:
            raise ValueError(f"`solver` must be 'newton' or 'brent', got '{solver}'.")
        self._solver = solver
        self._max_iter = max_iter
        self._tol = tol

    def fit(self, x: np.ndarray, y: np.ndarray) -> PolychoricCorrelation:
        """Fit polychoric correlation to two ordinal arrays. Returns self."""
        x_arr, y_arr = self._prepare(x, y)
        _validate_ordinal(x_arr, "x")
        _validate_ordinal(y_arr, "y")
        self._correlation = self._clip(self._polychoric(x_arr, y_arr))
        return self

    def _polychoric(self, x: np.ndarray, y: np.ndarray) -> float:
        n = x.size
        ux, uy = np.unique(x), np.unique(y)
        n_rs = np.array([[np.sum((x == xi) & (y == yj)) for yj in uy] for xi in ux], dtype=float)
        assert n_rs.sum() == n
        tx, ty = _thresholds(x), _thresholds(y)
        if self._solver == "newton":
            return self._polychoric_newton(n_rs, tx, ty, ux, uy)
        return self._polychoric_brent(n_rs, tx, ty, ux, uy)

    def _polychoric_newton(
        self, n_rs: np.ndarray, tx: np.ndarray, ty: np.ndarray, ux: np.ndarray, uy: np.ndarray
    ) -> float:
        rho = 0.0
        bound = self._max_cor
        score_val = 0.0
        for iteration in range(self._max_iter):
            score_val = info_val = 0.0
            for i in range(len(ux)):
                for j in range(len(uy)):
                    lower = (float(tx[i]), float(ty[j]))
                    upper = (float(tx[i + 1]), float(ty[j + 1]))
                    p = max(_pi_rs(lower=lower, upper=upper, corr=rho), _CELL_FLOOR)
                    dp = _pi_rs_derivative(lower=np.array(lower), upper=np.array(upper), corr=rho)
                    ratio = dp / p
                    score_val += n_rs[i, j] * ratio
                    info_val += n_rs[i, j] * ratio**2
            if abs(score_val) < self._tol:
                break
            if info_val < 1e-14:
                logger.warning(
                    "Fisher information ≈ 0 at ρ=%.4f after %d iter; returning current estimate.", rho, iteration
                )
                break
            step = score_val / info_val
            rho_new = float(np.clip(rho + step, -bound, bound))
            if abs(rho_new - rho) < self._tol:
                rho = rho_new
                break
            rho = rho_new
        else:
            logger.warning(
                "Fisher scoring did not converge in %d iterations (|score|=%.2e).", self._max_iter, abs(score_val)
            )
        return rho

    def _polychoric_brent(
        self, n_rs: np.ndarray, tx: np.ndarray, ty: np.ndarray, ux: np.ndarray, uy: np.ndarray
    ) -> float:
        bound = self._max_cor

        def neg_log_likelihood(rho: float) -> float:
            total = 0.0
            for i in range(len(ux)):
                for j in range(len(uy)):
                    if n_rs[i, j] == 0:
                        continue
                    lower = (float(tx[i]), float(ty[j]))
                    upper = (float(tx[i + 1]), float(ty[j + 1]))
                    p = max(_pi_rs(lower=lower, upper=upper, corr=rho), _CELL_FLOOR)
                    total += n_rs[i, j] * np.log(p)
            return -total

        result = minimize_scalar(neg_log_likelihood, bounds=(-bound, bound), method="bounded")
        return float(result.x)


# ---------------------------------------------------------------------------
# Polyserial correlation
# ---------------------------------------------------------------------------


class PolyserialCorrelation(CorrelationMeasure):
    """Ad-hoc polyserial correlation between one continuous and one ordinal variable.

    Args:
        max_cor: Clip bound. Defaults to 0.9999.
        n_levels_threshold: Variables with fewer unique values are treated as
            ordinal. Defaults to 20.
    """

    def __init__(self, max_cor: float = 0.9999, n_levels_threshold: int = 20) -> None:
        """Init. Variables with < n_levels_threshold unique values are treated as ordinal."""
        super().__init__(max_cor=max_cor)
        if n_levels_threshold < 2:
            raise ValueError("`n_levels_threshold` must be ≥ 2.")
        self._n_levels_threshold = n_levels_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> PolyserialCorrelation:
        """Fit polyserial correlation to a continuous/ordinal pair. Returns self."""
        x_arr, y_arr = self._prepare(x, y)
        x_is_ord = len(np.unique(x_arr)) < self._n_levels_threshold
        y_is_ord = len(np.unique(y_arr)) < self._n_levels_threshold
        if not x_is_ord and not y_is_ord:
            raise ValueError("Both variables appear continuous; use a Spearman/NPN estimator instead.")
        if x_is_ord and y_is_ord:
            raise ValueError("Both variables appear ordinal; use PolychoricCorrelation instead.")
        cont_arr, ord_arr = (y_arr, x_arr) if x_is_ord else (x_arr, y_arr)
        cont_name, ord_name = ("y", "x") if x_is_ord else ("x", "y")
        _validate_continuous(cont_arr, cont_name)
        _validate_ordinal(ord_arr, ord_name)
        self._correlation = self._clip(self._polyserial(cont_arr, ord_arr))
        return self

    def _polyserial(self, cont: np.ndarray, disc: np.ndarray) -> float:
        unique_vals = np.sort(np.unique(disc).astype(float))
        threshold_estimate = _thresholds(disc)
        interior_thresholds = threshold_estimate[1:-1]
        value_diff = np.diff(unique_vals)
        lambda_val = float(np.sum(stats.norm.pdf(interior_thresholds) * value_diff))
        if lambda_val == 0:
            raise ValueError("Denominator λ in polyserial estimator is zero.")
        s_disc = float(np.std(disc.astype(float), ddof=1))
        r = _npn_pearson(cont, disc)
        return r * s_disc / lambda_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def f_hat(x: np.ndarray) -> np.ndarray:
    """Winsorized nonparanormal transformation (Liu et al. 2009).

    Args:
        x: 1-D numeric array (≥ 2 observations).

    Returns:
        Transformed array scaled to unit variance.
    """
    x_arr = _to_array(x)
    if x_arr.size < 2:
        raise ValueError("f_hat requires at least 2 observations.")
    return _f_hat(x_arr)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation coefficient."""
    x_arr, y_arr = _to_array(x), _to_array(y)
    _validate_pair(x_arr, y_arr)
    rho, _ = stats.spearmanr(x_arr, y_arr)
    return float(rho)


def pairwise_latent_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_cor: float = 0.9999,
    n_levels_threshold: int = 20,
    verbose: bool = False,
) -> float:
    """Dispatch to the appropriate correlation estimator based on variable types.

    - Both continuous → nonparanormal Spearman sin-transform.
    - Both ordinal → polychoric MLE.
    - Mixed → ad-hoc polyserial.

    Args:
        x: First variable.
        y: Second variable (same length as x).
        max_cor: Clip bound for the result.
        n_levels_threshold: Unique-value count below which a variable is
            treated as ordinal.
        verbose: Log which estimator was selected.

    Returns:
        Correlation estimate in [−1, 1].
    """
    x_arr, y_arr = _to_array(x), _to_array(y)
    _validate_pair(x_arr, y_arr)

    x_is_disc = len(np.unique(x_arr)) < n_levels_threshold
    y_is_disc = len(np.unique(y_arr)) < n_levels_threshold

    if not x_is_disc and not y_is_disc:
        if verbose:
            logger.info("Both continuous — using nonparanormal Spearman.")
        rho = spearman(x_arr, y_arr)
        return float(np.clip(2 * np.sin(np.pi / 6 * rho), -max_cor, max_cor))

    if x_is_disc and y_is_disc:
        if verbose:
            logger.info("Both ordinal — using polychoric correlation.")
        return PolychoricCorrelation(max_cor=max_cor).fit(x_arr, y_arr).correlation

    if verbose:
        logger.info("Mixed pair — using polyserial correlation.")
    return PolyserialCorrelation(max_cor=max_cor, n_levels_threshold=n_levels_threshold).fit(x_arr, y_arr).correlation
