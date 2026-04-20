"""Necessary independence tests for the PC algorithm."""

from abc import ABCMeta, abstractmethod
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

from .correlations import adhoc_polyserial


class Itest(metaclass=ABCMeta):
    """Abstract meta class for independence tests."""

    def _check_input(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
    ) -> None:
        if not isinstance(x_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("x_data must be of type np.ndarray, pd.DataFrame, or pd.Series")
        if not isinstance(y_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("y_data must be of type np.ndarray, pd.DataFrame, or pd.Series")

    @abstractmethod
    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
    ) -> tuple[float, float | str]:
        """Abstract method for independence tests.

        Args:
            x_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test
            y_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test


        Returns:
            tuple[float, float | str]: Test statistic and corresponding pvalue (Test decision).
        """


class CItest(metaclass=ABCMeta):
    """Abstract meta class for independence tests."""

    def _check_input(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
        z_data: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> None:
        if not isinstance(x_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("x_data must be of type np.ndarray, pd.DataFrame, or pd.Series")
        if not isinstance(y_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("y_data must be of type np.ndarray, pd.DataFrame, or pd.Series")
        if z_data is not None and not isinstance(z_data, np.ndarray | pd.DataFrame | pd.Series):
            raise TypeError("z_data must be of type np.ndarray, pd.DataFrame, or pd.Series")

    @abstractmethod
    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
        z_data: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> tuple[float, float]:
        """Abstract method for independence tests.

        Args:
            x_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test
            y_data (np.ndarray | pd.DataFrame | pd.Series): Variables involved in the test
            z_data (np.ndarray | pd.DataFrame | pd.Series | None): Variables involved in the test


        Returns:
            tuple[float, float]: Test statistic and corresponding pvalue (Test decision).
        """


class FisherZVec(CItest):
    """Simple extension of standard Fisher-Z test for independence."""

    def __init__(self) -> None:
        """Init of the object."""
        pass

    @staticmethod
    def _as_2d_array(data: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
        """Convert supported tabular/array inputs to a 2D numpy array."""
        arr = data.to_numpy() if isinstance(data, pd.DataFrame | pd.Series) else data

        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        return arr

    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
        z_data: np.ndarray | pd.DataFrame | pd.Series | None = None,
        corr_threshold: float = 0.999,
    ) -> tuple[float, float]:
        """Retrieve (composite) p_value using Fisher z-transformation.

        Appropriate when data is jointly Gaussian.

        Args:
            x_data (np.ndarray): X_data.
            y_data (np.ndarray): Y_data.
            z_data (np.ndarray | None): Z_data. defaults to None.
            corr_threshold (float, optional): Threshold to make sure
                r in [-1,1]. Defaults to 0.999.

        Returns:
            tuple[float,float]: test_statistic, p_value
        """
        x_arr = self._as_2d_array(x_data)
        y_arr = self._as_2d_array(y_data)
        z_arr = None if z_data is None else self._as_2d_array(z_data)

        n = x_arr.shape[0]

        if z_arr is not None:
            sep_set_length = z_arr.shape[1]
            corrdata = np.empty((n, 2 + z_arr.shape[1], x_arr.shape[1] * y_arr.shape[1]))
            for k, (i, j) in enumerate(product(range(x_arr.shape[1]), range(y_arr.shape[1]))):
                corrdata[:, :, k] = np.concatenate(
                    [x_arr[:, i][:, np.newaxis], y_arr[:, j][:, np.newaxis], z_arr], axis=1
                )
            precision_matrices = np.empty((2 + z_arr.shape[1], 2 + z_arr.shape[1], x_arr.shape[1] * y_arr.shape[1]))
            for k in range(precision_matrices.shape[-1]):
                corrmat = np.corrcoef(corrdata[:, :, k].T)
                try:
                    precision_matrices[:, :, k] = np.linalg.inv(corrmat)
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        "The correlation matrix of your data is singular. \
                        Partial correlations cannot be estimated. Are there  \
                        collinearities in your data?"
                    ) from error

            r = np.empty(precision_matrices.shape[-1])
            for k in range(precision_matrices.shape[-1]):
                precision_matrix = precision_matrices[:, :, k]
                r[k] = -1 * precision_matrix[0, 1] / np.sqrt(np.abs(precision_matrix[0, 0] * precision_matrix[1, 1]))
        else:
            sep_set_length = 0
            uncond = []
            for i in range(x_arr.shape[1]):
                uncond.append(np.corrcoef(np.concatenate([x_arr[:, i][:, np.newaxis], y_arr], axis=1).T)[1:, 0])
            r = np.concatenate(uncond)

        r = np.minimum(corr_threshold, np.maximum(-1 * corr_threshold, r))  # make r between -1 and 1
        # Fisher’s z-transform
        factor = np.sqrt(n - sep_set_length - 3)
        z_transform = factor * 0.5 * np.log((1 + r) / (1 - r))
        test_stat = factor * z_transform
        p_value = 2 * (1 - norm.cdf(abs(z_transform)))

        final_test_stat = test_stat[np.argmin(np.abs(test_stat))]
        final_p_value = np.max(p_value)

        return (float(final_test_stat), float(final_p_value))


def _make_positive_definite(mat: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues to make a symmetric matrix positive definite."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


class MixedFisherZ(CItest):
    """Fisher Z conditional independence test for mixed continuous/ordinal data.

    Uses :func:`~mixpc.correlations.adhoc_polyserial` to build a pairwise
    correlation matrix that automatically selects the right estimator for each
    variable pair:

    - Both continuous → nonparanormal Spearman sin-transform.
    - Both ordinal → polychoric MLE.
    - Mixed → ad-hoc polyserial.

    The partial correlation of X and Y given Z is then derived from the
    precision matrix of the joint correlation matrix, and Fisher's Z transform
    is applied.

    Args:
        n_levels_threshold: Variables with fewer unique values than this are
            treated as ordinal. Defaults to 20.
        max_cor: Clip bound for individual pairwise correlations. Defaults to
            0.9999.
    """

    def __init__(self, n_levels_threshold: int = 20, max_cor: float = 0.9999) -> None:
        """Init. Variables with < n_levels_threshold unique values are treated as ordinal."""
        self._n_levels_threshold = n_levels_threshold
        self._max_cor = max_cor

    @staticmethod
    def _as_1d(data: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
        arr = data.to_numpy() if isinstance(data, pd.DataFrame | pd.Series) else np.asarray(data)
        return arr.ravel()

    def _build_corr_matrix(self, cols: list[np.ndarray]) -> np.ndarray:
        """Build a full pairwise correlation matrix from a list of 1-D arrays."""
        k = len(cols)
        corr = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                r = adhoc_polyserial(
                    cols[i],
                    cols[j],
                    max_cor=self._max_cor,
                    n_levels_threshold=self._n_levels_threshold,
                )
                corr[i, j] = corr[j, i] = r
        return corr

    def test(
        self,
        x_data: np.ndarray | pd.DataFrame | pd.Series,
        y_data: np.ndarray | pd.DataFrame | pd.Series,
        z_data: np.ndarray | pd.DataFrame | pd.Series | None = None,
        corr_threshold: float = 0.999,
    ) -> tuple[float, float]:
        """Test conditional independence of X and Y given Z.

        Args:
            x_data: Variable X — shape (n,) or (n, 1).
            y_data: Variable Y — shape (n,) or (n, 1).
            z_data: Conditioning set — shape (n, k) or None for marginal test.
            corr_threshold: Clip bound applied to the partial correlation
                before the Fisher Z transform.

        Returns:
            (test_statistic, p_value)
        """
        self._check_input(x_data, y_data, z_data)

        x_arr = self._as_1d(x_data)
        y_arr = self._as_1d(y_data)
        n = x_arr.shape[0]

        if z_data is None:
            # Marginal test: direct pairwise correlation
            r = adhoc_polyserial(
                x_arr, y_arr,
                max_cor=corr_threshold,
                n_levels_threshold=self._n_levels_threshold,
            )
            sep_set_length = 0
            cols = [x_arr, y_arr]
        else:
            z_arr = z_data.to_numpy() if isinstance(z_data, pd.DataFrame | pd.Series) else np.asarray(z_data)
            if z_arr.ndim == 1:
                z_arr = z_arr[:, np.newaxis]
            sep_set_length = z_arr.shape[1]

            # Build correlation matrix for [X, Y, Z_1, ..., Z_k]
            cols = [x_arr, y_arr] + [z_arr[:, i] for i in range(sep_set_length)]
            corr_mat = self._build_corr_matrix(cols)
            corr_mat = _make_positive_definite(corr_mat)

            try:
                precision = np.linalg.inv(corr_mat)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Correlation matrix is singular; check for collinearities.") from exc

            # Partial correlation r(X,Y|Z) via precision matrix
            r = -precision[0, 1] / np.sqrt(np.abs(precision[0, 0] * precision[1, 1]))

        r = float(np.clip(r, -corr_threshold, corr_threshold))
        factor = np.sqrt(n - sep_set_length - 3)
        z_stat = factor * 0.5 * np.log((1 + r) / (1 - r))
        p_value = float(2 * (1 - norm.cdf(abs(z_stat))))

        return (float(z_stat), p_value)
