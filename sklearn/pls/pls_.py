"""
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""

# Author: Edouard Duchesnay <edouard.duchesnay@cea.fr>
# License: BSD 3 clause

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_arrays
from ..utils import deprecated
from ._base import _PLS, _center_scale_xy
from .. import cca

import numpy as np
from scipy import linalg

__all__ = ['PLSCanonical', 'PLSRegression', 'PLSSVD']


class PLSRegression(_PLS):
    """PLS regression

    PLSRegression implements the PLS 2 blocks regression known as PLS2 or PLS1
    in case of one dimensional response.
    This class inherits from _PLS with mode="A", deflation_mode="regression",
    norm_y_weights=False and algorithm="nipals".

    Parameters
    ----------
    X : array-like of predictors, shape = [n_samples, p]
        Training vectors, where n_samples in the number of samples and
        p is the number of predictors.

    Y : array-like of response, shape = [n_samples, q]
        Training vectors, where n_samples in the number of samples and
        q is the number of response variables.

    n_components : int, (default 2)
        Number of components to keep.

    scale : boolean, (default True)
        whether to scale the data

    max_iter : an integer, (default 500)
        the maximum number of iterations of the NIPALS inner loop (used
        only if algorithm="nipals")

    tol : non-negative real
        Tolerance used in the iterative algorithm default 1e-06.

    copy : boolean, default True
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect

    Attributes
    ----------
    `x_weights_` : array, [p, n_components]
        X block weights vectors.

    `y_weights_` : array, [q, n_components]
        Y block weights vectors.

    `x_loadings_` : array, [p, n_components]
        X block loadings vectors.

    `y_loadings_` : array, [q, n_components]
        Y block loadings vectors.

    `x_scores_` : array, [n_samples, n_components]
        X scores.

    `y_scores_` : array, [n_samples, n_components]
        Y scores.

    `x_rotations_` : array, [p, n_components]
        X block to latents rotations.

    `y_rotations_` : array, [q, n_components]
        Y block to latents rotations.

    coefs: array, [p, q]
        The coefficients of the linear model: Y = X coefs + Err

    Notes
    -----
    For each component k, find weights u, v that optimizes:
    ``max corr(Xk u, Yk v) * var(Xk u) var(Yk u)``, such that ``|u| = 1``

    Note that it maximizes both the correlations between the scores and the
    intra-block variances.

    The residual matrix of X (Xk+1) block is obtained by the deflation on
    the current X score: x_score.

    The residual matrix of Y (Yk+1) block is obtained by deflation on the
    current X score. This performs the PLS regression known as PLS2. This
    mode is prediction oriented.

    This implementation provides the same results that 3 PLS packages
    provided in the R language (R-project):

        - "mixOmics" with function pls(X, Y, mode = "regression")
        - "plspm " with function plsreg2(X, Y)
        - "pls" with function oscorespls.fit(X, Y)

    Examples
    --------
    >>> from sklearn.pls import PLSCanonical, PLSRegression, CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> pls2 = PLSRegression(n_components=2)
    >>> pls2.fit(X, Y)
    ... # doctest: +NORMALIZE_WHITESPACE
    PLSRegression(copy=True, max_iter=500, n_components=2, scale=True,
            tol=1e-06)
    >>> Y_pred = pls2.predict(X)

    References
    ----------

    Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
    emphasis on the two-block case. Technical Report 371, Department of
    Statistics, University of Washington, Seattle, 2000.

    In french but still a reference:
    Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
    Editions Technic.
    """

    def __init__(self, n_components=2, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        _PLS.__init__(self, n_components=n_components, scale=scale,
                      deflation_mode="regression", mode="A",
                      norm_y_weights=False, max_iter=max_iter, tol=tol,
                      copy=copy)


class PLSCanonical(_PLS):
    """ PLSCanonical implements the 2 blocks canonical PLS of the original Wold
    algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000].

    This class inherits from PLS with mode="A" and deflation_mode="canonical",
    norm_y_weights=True and algorithm="nipals", but svd should provide similar
    results up to numerical errors.

    Parameters
    ----------
    X : array-like of predictors, shape = [n_samples, p]
        Training vectors, where n_samples is the number of samples and
        p is the number of predictors.

    Y : array-like of response, shape = [n_samples, q]
        Training vectors, where n_samples is the number of samples and
        q is the number of response variables.

    n_components : int, number of components to keep. (default 2).

    scale : boolean, scale data? (default True)

    algorithm : string, "nipals" or "svd"
        The algorithm used to estimate the weights. It will be called
        n_components times, i.e. once for each iteration of the outer loop.

    max_iter : an integer, (default 500)
        the maximum number of iterations of the NIPALS inner loop (used
        only if algorithm="nipals")

    tol : non-negative real, default 1e-06
        the tolerance used in the iterative algorithm

    copy : boolean, default True
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect

    Attributes
    ----------
    `x_weights_` : array, shape = [p, n_components]
        X block weights vectors.

    `y_weights_` : array, shape = [q, n_components]
        Y block weights vectors.

    `x_loadings_` : array, shape = [p, n_components]
        X block loadings vectors.

    `y_loadings_` : array, shape = [q, n_components]
        Y block loadings vectors.

    `x_scores_` : array, shape = [n_samples, n_components]
        X scores.

    `y_scores_` : array, shape = [n_samples, n_components]
        Y scores.

    `x_rotations_` : array, shape = [p, n_components]
        X block to latents rotations.

    `y_rotations_` : array, shape = [q, n_components]
        Y block to latents rotations.

    Notes
    -----
    For each component k, find weights u, v that optimize::
    max corr(Xk u, Yk v) * var(Xk u) var(Yk u), such that ``|u| = |v| = 1``

    Note that it maximizes both the correlations between the scores and the
    intra-block variances.

    The residual matrix of X (Xk+1) block is obtained by the deflation on the
    current X score: x_score.

    The residual matrix of Y (Yk+1) block is obtained by deflation on the
    current Y score. This performs a canonical symmetric version of the PLS
    regression. But slightly different than the CCA. This is mostly used
    for modeling.

    This implementation provides the same results that the "plspm" package
    provided in the R language (R-project), using the function plsca(X, Y).
    Results are equal or colinear with the function
    ``pls(..., mode = "canonical")`` of the "mixOmics" package. The difference
    relies in the fact that mixOmics implementation does not exactly implement
    the Wold algorithm since it does not normalize y_weights to one.

    Examples
    --------
    >>> from sklearn.pls import PLSCanonical, PLSRegression, CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, Y)
    ... # doctest: +NORMALIZE_WHITESPACE
    PLSCanonical(algorithm='nipals', copy=True, max_iter=500, n_components=2,
                 scale=True, tol=1e-06)
    >>> X_c, Y_c = plsca.transform(X, Y)

    References
    ----------

    Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
    emphasis on the two-block case. Technical Report 371, Department of
    Statistics, University of Washington, Seattle, 2000.

    Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
    Editions Technic.

    See also
    --------
    CCA
    PLSSVD
    """

    def __init__(self, n_components=2, scale=True, algorithm="nipals",
                 max_iter=500, tol=1e-06, copy=True):
        _PLS.__init__(self, n_components=n_components, scale=scale,
                      deflation_mode="canonical", mode="A",
                      norm_y_weights=True, algorithm=algorithm,
                      max_iter=max_iter, tol=tol, copy=copy)


class PLSSVD(BaseEstimator, TransformerMixin):
    """Partial Least Square SVD

    Simply perform a svd on the crosscovariance matrix: X'Y
    There are no iterative deflation here.

    Parameters
    ----------
    X : array-like of predictors, shape = [n_samples, p]
        Training vector, where n_samples is the number of samples and
        p is the number of predictors. X will be centered before any analysis.

    Y : array-like of response, shape = [n_samples, q]
        Training vector, where n_samples is the number of samples and
        q is the number of response variables. X will be centered before any
        analysis.

    n_components : int, (default 2).
        number of components to keep.

    scale : boolean, (default True)
        whether to scale X and Y.

    Attributes
    ----------
    `x_weights_` : array, [p, n_components]
        X block weights vectors.

    `y_weights_` : array, [q, n_components]
        Y block weights vectors.

    `x_scores_` : array, [n_samples, n_components]
        X scores.

    `y_scores_` : array, [n_samples, n_components]
        Y scores.

    See also
    --------
    PLSCanonical
    CCA
    """

    def __init__(self, n_components=2, scale=True, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy

    def fit(self, X, Y):
        # copy since this will contains the centered data
        X, Y = check_arrays(X, Y, dtype=np.float, copy=self.copy,
                            sparse_format='dense')

        n = X.shape[0]
        p = X.shape[1]

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        if n != Y.shape[0]:
            raise ValueError(
                'Incompatible shapes: X has %s samples, while Y '
                'has %s' % (X.shape[0], Y.shape[0]))

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('invalid number of components')

        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ =\
            _center_scale_xy(X, Y, self.scale)
        # svd(X'Y)
        C = np.dot(X.T, Y)
        U, s, V = linalg.svd(C, full_matrices=False)
        V = V.T
        self.x_scores_ = np.dot(X, U)
        self.y_scores_ = np.dot(Y, V)
        self.x_weights_ = U
        self.y_weights_ = V
        return self

    def transform(self, X, Y=None):
        """Apply the dimension reduction learned on the train data."""
        Xr = (X - self.x_mean_) / self.x_std_
        x_scores = np.dot(Xr, self.x_weights_)
        if Y is not None:
            Yr = (Y - self.y_mean_) / self.y_std_
            y_scores = np.dot(Yr, self.y_weights_)
            return x_scores, y_scores
        return x_scores

    def fit_transform(self, X, y=None, **fit_params):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.

        Y : array-like of response, shape = [n_samples, q], optional
            Training vectors, where n_samples in the number of samples and
            q is the number of response variables.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y, **fit_params).transform(X, y)


class CCA(cca.CCA):
    @deprecated("the CCA was moved to the cca module.")
    def __init__(self, n_components=2, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        cca.CCA.__init__(self, n_components=n_components, scale=scale,
                         max_iter=max_iter, tol=tol, copy=copy)
