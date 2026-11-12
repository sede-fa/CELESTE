import warnings

import numpy as np

from ..._deprecation import warn_deprecated


class RecycleMixin:
    """
    Recycle-loop convergence.

    ``Recycle_Tear`` converges a tear stream: the component flows and the
    temperature of the flange that receives the recycle. It offers direct
    substitution, Wegstein, Aitken/Steffensen and Anderson acceleration (default).
    The former ``Recycle_*`` methods are thin wrappers around it.
    See docs/recycles.md for how to choose the tear stream and the method.
    """

    _TEAR_METHODS = ('anderson', 'wegstein', 'aitken', 'direct')

    def Recycle_Tear(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=300, method='anderson',
                     memory=5, q_bounds=(-5.0, 0.0), q=None, n_direct=0, print_err=False):
        """
        Converge a recycle by iterating on the full tear stream.

        The tear variables are the component molar flows and the temperature of
        ``F_sink`` (the flange that receives the recycle); the pressure is taken
        from ``F_source``. One pass sets ``F_sink``, calls ``flow_method`` and
        reads the result from ``F_source``.

        Parameters:
            F_source:    flange at the end of the loop (e.g. splitter outlet)
            F_sink:      flange at the start of the loop (e.g. mixer inlet); its current state is the first guess
            flow_method: name of the method (or a callable) that evaluates the loop once
            eps:         convergence tolerance on the largest relative change of a tear variable
            max_iter:    largest number of passes
            method:      'anderson' - Anderson acceleration over the last ``memory`` passes (default; accounts
                                      for the coupling between the tear variables)
                         'wegstein' - Wegstein step on each variable; factor q estimated from the last two
                                      passes and kept within ``q_bounds``, or fixed with ``q=``
                         'aitken'   - Aitken delta-squared (Steffensen) extrapolation every second pass,
                                      in the vector form of Irons and Tuck
                         'direct'   - direct substitution (fixed-point iteration)
            n_direct:    direct-substitution passes before the acceleration starts (stabilises the first guess)

        Returns:
            (number of passes, final error)
        """
        method = 'aitken' if method == 'steffensen' else method
        if method not in self._TEAR_METHODS:
            raise ValueError(f"method must be one of {self._TEAR_METHODS}")
        run = getattr(self, flow_method) if isinstance(flow_method, str) else flow_method
        self._record_link(F_source, F_sink)

        def pack(f):
            return np.concatenate((f.n*np.asarray(f.Xi, dtype=float), [f.T]))

        def unpack(x):
            flows = np.maximum(x[:-1], 0.0)
            total = flows.sum()
            F_sink.P = F_source.P
            F_sink.T = float(x[-1])
            F_sink.n = float(total)
            F_sink.Xi = flows/total if total > 0 else np.asarray(F_source.Xi, dtype=float)

        x = pack(F_sink)
        xs, gs = [], []
        error, count = np.inf, 0
        extrapolated = True                     # aitken: the next pass is a plain one
        while count < max_iter:
            unpack(x)
            x = pack(F_sink)                    # what was actually applied (negative flows are clipped)
            run()
            g = pack(F_source)
            count += 1
            scale = np.maximum(np.abs(g), 1e-9*max(np.abs(g[:-1]).sum(), 1e-300))
            error = float(np.max(np.abs(g - x)/scale))
            if print_err and count % 10 == 0:
                print(f"  tear pass {count}: error={error:.2e}, n={g[:-1].sum():.4f}")
            if error < eps:
                break
            xs.append(x); gs.append(g)
            xs, gs = xs[-(memory + 1):], gs[-(memory + 1):]
            x_new = g
            if count <= n_direct:
                pass
            elif method == 'wegstein' and (q is not None or len(xs) >= 2):
                if q is None:
                    dx = xs[-1] - xs[-2]
                    ok = np.abs(dx) > 1e-14*np.maximum(np.abs(x), 1.0)
                    slope = np.divide(gs[-1] - gs[-2], dx, out=np.zeros_like(dx), where=ok)
                    qv = np.divide(slope, slope - 1.0, out=np.zeros_like(slope), where=np.abs(slope - 1.0) > 1e-12)
                    qv = np.clip(qv, q_bounds[0], q_bounds[1])
                else:
                    qv = q
                x_new = qv*x + (1.0 - qv)*g
            elif method == 'aitken' and len(xs) >= 2:
                if extrapolated:
                    extrapolated = False        # one plain pass to get the second difference
                else:
                    # vector form (Irons-Tuck) on the scaled variables: one step length for all of them
                    d1, d2 = (gs[-2] - xs[-2])/scale, (g - x)/scale
                    dd = d2 - d1
                    den = float(dd @ dd)
                    if den > 0.0:
                        x_new = g - float(d2 @ dd)/den*(g - x)
                    extrapolated = True
            elif method == 'anderson' and len(xs) >= 2:
                F = (np.array(gs) - np.array(xs))/scale             # scaled residuals of the stored passes
                dF = (F[1:] - F[:-1]).T
                dG = (np.array(gs)[1:] - np.array(gs)[:-1]).T
                gamma, *_ = np.linalg.lstsq(dF, F[-1], rcond=1e-10)
                candidate = g - dG @ gamma
                if np.all(np.isfinite(candidate)):
                    x_new = candidate
            x = x_new
        unpack(g)
        if count >= max_iter and error >= eps:
            from ...solvers import ConvergenceWarning
            warnings.warn(f"{self.name}: recycle not converged after {max_iter} passes (error {error:.2e})",
                          ConvergenceWarning, stacklevel=2)
        elif print_err:
            print(f"tear stream converged in {count} passes, error: {error:.2e}")
        return count, error

    # ---- former interfaces ----------------------------------------------------------------------
    def _former_recycle(self, old, F_source, F_sink, flow_method, eps, max_iter, print_err, **tear):
        warn_deprecated(f"{old}() is deprecated; use Recycle_Tear(F_source, F_sink, method="
                        f"'{tear['method']}') - it converges the whole tear stream, not only the total flow",
                        stacklevel=4)
        return self.Recycle_Tear(F_source, F_sink, flow_method=flow_method, eps=eps, max_iter=max_iter,
                                 print_err=print_err, **tear)

    def Recycle_FPI(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=1000, print_err=False):
        """Deprecated: ``Recycle_Tear(..., method='direct')``."""
        return self._former_recycle('Recycle_FPI', F_source, F_sink, flow_method, eps, max_iter, print_err,
                                    method='direct')

    def Recycle_Wegstein_q(self, F_source, F_sink, flow_method='seq_sol', q=-5, eps=1e-8, max_iter=1000,
                           print_err=False):
        """Deprecated: ``Recycle_Tear(..., method='wegstein', q=q)``."""
        return self._former_recycle('Recycle_Wegstein_q', F_source, F_sink, flow_method, eps, max_iter, print_err,
                                    method='wegstein', q=q)

    def Recycle_Wegstein_Generic(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=1000,
                                 print_err=False):
        """Deprecated: ``Recycle_Tear(..., method='wegstein')``."""
        return self._former_recycle('Recycle_Wegstein_Generic', F_source, F_sink, flow_method, eps, max_iter,
                                    print_err, method='wegstein')

    def Recycle_Wegstein_Generic2(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=1000,
                                  print_err=False):
        """Deprecated: ``Recycle_Tear(..., method='wegstein')``."""
        return self._former_recycle('Recycle_Wegstein_Generic2', F_source, F_sink, flow_method, eps, max_iter,
                                    print_err, method='wegstein')

    def Recycle_Steffensen(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=1000, print_err=False):
        """Deprecated: ``Recycle_Tear(..., method='aitken')``."""
        return self._former_recycle('Recycle_Steffensen', F_source, F_sink, flow_method, eps, max_iter, print_err,
                                    method='aitken')

    def Recycle_Aitken(self, F_source, F_sink, flow_method='seq_sol', eps=1e-6, max_iter=1000, print_err=False):
        """Deprecated: ``Recycle_Tear(..., method='aitken')``."""
        return self._former_recycle('Recycle_Aitken', F_source, F_sink, flow_method, eps, max_iter, print_err,
                                    method='aitken')

    def Recycle_Hybrid_FPI_Aitken(self, F_source, F_sink, flow_method='seq_sol', eps=1e-6, max_iter=1000,
                                  print_err=False, n_fpi_initial=5, n_fpi_stabilize=5):
        """Deprecated: ``Recycle_Tear(..., method='aitken', n_direct=n_fpi_initial)``."""
        return self._former_recycle('Recycle_Hybrid_FPI_Aitken', F_source, F_sink, flow_method, eps, max_iter,
                                    print_err, method='aitken', n_direct=n_fpi_initial)

    def recycle_solver(self, x, F_source, F_sink, flow_method, print_err=False):
        """
        Deprecated: one pass of a loop for a given total recycle flow ``x``; returns the relative change of
        the total flow. ``Recycle_Tear`` replaces the solvers built on it.
        """
        warn_deprecated("recycle_solver() is deprecated; use Recycle_Tear()")
        self.connect_seq(F_source, F_sink)
        F_sink.n = x
        getattr(self, flow_method)() if isinstance(flow_method, str) else flow_method()
        error = np.abs((F_source.n - x)/F_source.n)
        if print_err:
            print(error)
        return error
