"""
Robust solution of the small nonlinear systems inside unit models.

All unit models used to call ``scipy.optimize.root`` or ``least_squares``
directly and returned whatever came back. :func:`solve` wraps both behind one
call that

* respects bounds (mole fractions stay in [0, 1], flows and temperatures stay
  positive), so that unphysical roots are not reported as solutions,
* falls back to a second method when the first one stalls,
* judges success by the residual actually reached, and
* reports failures through :class:`ConvergenceWarning` instead of ``print``.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize as spo

__all__ = ["ConvergenceWarning", "solve", "solve_scalar", "trivial_solution"]


class ConvergenceWarning(RuntimeWarning):
    """A unit model did not reach the requested residual tolerance."""


def _norm(f) -> float:
    f = np.atleast_1d(np.asarray(f, dtype=float))
    return float(np.max(np.abs(f))) if f.size and np.all(np.isfinite(f)) else np.inf


def _finish(res, fun, tol, method, name, warn):
    f = np.atleast_1d(fun(res.x))        # leaves the model at the returned point
    res.fun = f
    res.residual_norm = _norm(f)
    res.success = bool(res.residual_norm <= tol)
    res.method = method
    if not res.success and warn:
        warnings.warn(f"{name or 'solve'}: not converged (max |residual| = {res.residual_norm:.3e} > {tol:.1e}, "
                      f"method = {method})", ConvergenceWarning, stacklevel=3)
    return res


def solve(fun, x0, bounds=None, tol: float = 1e-7, scale: float = 1.0, name: str = "", warn: bool = True,
          **least_squares_kw):
    """
    Solve ``fun(x) = 0`` for a vector ``x``.

    Parameters
    ----------
    fun    : callable returning the residual vector (may have more equations than unknowns)
    x0     : initial guess
    bounds : optional ``(lower, upper)``; the guess is clipped into the bounds and
             the returned solution always satisfies them
    tol    : convergence criterion, ``max|fun(x)| <= tol*scale``
    scale  : typical magnitude of the residuals (e.g. the feed flow in mol/s)
    name   : label used in the warning when the solve fails
    warn   : emit a :class:`ConvergenceWarning` on failure

    Returns
    -------
    scipy.optimize.OptimizeResult with ``x``, ``fun``, ``success``,
    ``residual_norm`` and ``method``. ``fun`` has been evaluated at ``x`` last,
    so the model that owns ``fun`` is left in the solved state.

    Notes
    -----
    Square systems are tried with MINPACK's hybrid Powell method first (fast).
    If that fails, or lands outside the bounds, a bounded trust-region
    least-squares solve takes over, followed by a Newton polish.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    tol = tol*max(abs(scale), 1e-300)
    candidates = []
    done = False

    def inside(x):
        return bounds is None or bool(np.all(x >= np.asarray(bounds[0]) - 1e-12) and np.all(x <= np.asarray(bounds[1]) + 1e-12))

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        n_eq = np.atleast_1d(fun(x0)).size
        square = n_eq == x0.size

        if square:
            # fast path: MINPACK's hybrid method; with bounds its answer only counts if it respects them
            try:
                res = spo.root(fun, x0, method="hybr", options=dict(xtol=1e-13))
                if inside(res.x):
                    candidates.append((_norm(res.fun), "hybr", res))
                    # a small residual is not enough: badly scaled equations (tiny equilibrium constants) have
                    # small residuals far from the root, and MINPACK then reports that it stalled
                    done = bool(res.success) and candidates[-1][0] <= tol
            except Exception:
                pass

        if not done:
            lsq_bounds = (-np.inf, np.inf) if bounds is None else bounds
            start = x0 if bounds is None else np.clip(x0, np.asarray(bounds[0], float) + 1e-12, np.asarray(bounds[1], float) - 1e-12)
            kw = dict(xtol=1e-13, ftol=1e-13, gtol=1e-13, x_scale="jac")
            kw.update(least_squares_kw)
            try:
                res = spo.least_squares(fun, start, bounds=lsq_bounds, **kw)
                candidates.append((_norm(res.fun), "least_squares", res))
                if square and candidates[-1][0] > tol:
                    # polish with a Newton-type method; accept only if it stays inside the bounds
                    pol = spo.root(fun, res.x, method="hybr", options=dict(xtol=1e-13))
                    if inside(pol.x):
                        candidates.append((_norm(pol.fun), "least_squares+hybr", pol))
            except Exception:
                pass

    if not candidates:
        res = spo.OptimizeResult(x=x0, fun=np.full(n_eq, np.nan), status=-1, message="all solvers failed")
        return _finish(res, fun, tol, "none", name, warn)

    err, method, res = min(candidates, key=lambda c: c[0])
    return _finish(res, fun, tol, method, name, warn)


def solve_scalar(fun, lo: float, hi: float, x0: float | None = None, tol: float = 1e-10, name: str = "",
                 warn: bool = True):
    """
    Solve a scalar equation ``fun(x) = 0`` on ``[lo, hi]``.

    Uses Brent's method when the bracket changes sign (guaranteed convergence),
    otherwise a bounded least-squares search.
    """
    f_lo, f_hi = float(fun(lo)), float(fun(hi))
    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo*f_hi <= 0.0:
        x = spo.brentq(fun, lo, hi, xtol=1e-12, rtol=1e-13) if f_lo*f_hi < 0.0 else (lo if f_lo == 0.0 else hi)
        res = spo.OptimizeResult(x=np.array([x]), status=1, message="brentq")
        return _finish(res, lambda v: fun(float(v[0])), max(tol, 1e-300), "brentq", name, warn)
    start = 0.5*(lo + hi) if x0 is None else x0
    return solve(lambda v: np.atleast_1d(fun(float(v[0]))), [start], bounds=([lo], [hi]), tol=tol, name=name, warn=warn)


def trivial_solution(x=()):
    """Result object for models that are evaluated explicitly (no iteration needed)."""
    return spo.OptimizeResult(x=np.atleast_1d(np.asarray(x, dtype=float)), fun=np.zeros(0), success=True,
                              residual_norm=0.0, method="explicit", status=1, message="explicit evaluation")
