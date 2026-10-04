"""Frozen diagonal preconditioners for generalised-linear models (NumPy only).

Proposition 5 (``docs/THEORY.md``) says SVRG with a *frozen* positive diagonal ``D`` has the
Johnson-Zhang linear rate in terms of the preconditioned constants ``(L_D, gamma_D)``. For
linear models the natural ``D`` is the diagonal of (an upper bound on) the Hessian:

* least squares:       ``D_j = mean_i x_ij^2``
* logistic regression: ``D_j = 0.25 * mean_i x_ij^2 + l2`` (``sigma(1 - sigma) <= 1/4``)

These helpers only compute diagonals and the Johnson-Zhang constants; the engines consume the
diagonal through ``set_frozen_preconditioner`` (PyTorch) or ``make_svrg_step(..., preconditioner=)``
(JAX).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def jacobi_diag_least_squares(features: FloatArray) -> FloatArray:
    """``diag(X^T X / n)``: the Hessian diagonal of 0.5 * mean (x.w - y)^2."""
    return np.mean(features ** 2, axis=0)


def jacobi_diag_logistic(features: FloatArray, l2: float) -> FloatArray:
    """Hessian-diagonal upper bound for L2-regularised logistic loss."""
    if l2 < 0.0:
        raise ValueError("l2 must be >= 0")
    return 0.25 * np.mean(features ** 2, axis=0) + l2


def johnson_zhang_recipe(features: FloatArray, diag: FloatArray) -> Dict[str, float]:
    """Hyper-parameters of the Johnson-Zhang theorem for least squares with preconditioner ``diag``.

    ``eta = 0.1 / L_D`` and ``m = ceil(50 L_D / gamma_D)`` give the guaranteed per-epoch contraction
    ``alpha = 1/2``. Requires every ``f_i`` convex and ``L_D``-smooth in the rescaled coordinates.
    """
    n: int = features.shape[0]
    scaled: FloatArray = features / np.sqrt(diag)
    l_d: float = float(np.max(np.sum(scaled ** 2, axis=1)))
    gamma_d: float = float(np.linalg.eigvalsh(scaled.T @ scaled / float(n))[0])
    eta: float = 0.1 / l_d
    inner: int = int(math.ceil(50.0 * l_d / gamma_d))
    alpha: float = (
        1.0 / (gamma_d * eta * (1.0 - 2.0 * l_d * eta) * inner) + 2.0 * l_d * eta / (1.0 - 2.0 * l_d * eta)
    )
    return {"L_D": l_d, "gamma_D": gamma_d, "eta": eta, "inner_steps": float(inner), "alpha": alpha}
