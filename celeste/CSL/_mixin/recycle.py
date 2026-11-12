import numpy as np


class RecycleMixin:
    """
    Recycle-loop convergence algorithms.

    ``Recycle_Tear`` converges the whole tear stream (component flows and
    temperature) with Anderson acceleration and is the recommended method. The older methods (FPI, Wegstein, Steffensen, Aitken) accelerate the
    total molar flow only; composition and temperature follow by direct
    substitution, which is slow for loops with a small purge.
    """

    def Recycle_Tear(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=300, method='anderson',
                     memory=5, q_bounds=(-5.0, 0.0), print_err=False):
        """
        Converge a recycle by iterating on the full tear stream.

        The tear variables are the component molar flows and the temperature of
        ``F_sink`` (the flange that receives the recycle); the pressure is taken
        from ``F_source``.

        Parameters:
            F_source:    flange at the end of the loop (e.g. splitter outlet)
            F_sink:      flange at the start of the loop (e.g. mixer inlet); its current state is the first guess
            flow_method: name of the method (or a callable) that evaluates the loop once
            eps:         convergence tolerance on the largest relative change of a tear variable
            method:      'anderson' - Anderson acceleration over the last ``memory`` passes (default; accounts
                                      for the coupling between the tear variables)
                         'wegstein' - bounded Wegstein step on each variable, factor q within ``q_bounds``
                         'direct'   - direct substitution

        Returns:
            (number of passes, final error)
        """
        if method not in ('anderson', 'wegstein', 'direct'):
            raise ValueError("method must be 'anderson', 'wegstein' or 'direct'")
        run = getattr(self, flow_method) if isinstance(flow_method, str) else flow_method

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
            if method == 'wegstein' and len(xs) >= 2:
                dx = xs[-1] - xs[-2]
                ok = np.abs(dx) > 1e-14*np.maximum(np.abs(x), 1.0)
                slope = np.divide(gs[-1] - gs[-2], dx, out=np.zeros_like(dx), where=ok)
                q = np.clip(np.divide(slope, slope - 1.0, out=np.zeros_like(slope), where=np.abs(slope - 1.0) > 1e-12),
                            q_bounds[0], q_bounds[1])
                x_new = q*x + (1.0 - q)*g
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
            import warnings
            from ...solvers import ConvergenceWarning
            warnings.warn(f"{self.name}: recycle not converged after {max_iter} passes (error {error:.2e})",
                          ConvergenceWarning, stacklevel=2)
        elif print_err:
            print(f"tear stream converged in {count} passes, error: {error:.2e}")
        return count, error

    def recycle_solver(self, x, F_source, F_sink, flow_method, print_err=False):
        """
        Generic method for recycle algorithms to use to complete recycles

        x, float:           guess for F_sink.n
        F_source, Flange:   Flange for source of recycle
        F_sink, Flange:     Flange for sink of recycle
        flow_method, str:   string name of method to run during recycle ("seq_sol" will run the entire system)
        """
        self.connect_seq(F_source, F_sink)
        F_sink.n = x

        if isinstance(flow_method, str):
            getattr(self, flow_method)()
        else:
            flow_method()

        error = np.abs((F_source.n - x) / F_source.n)
        if print_err:
            print(error)
        return error

    def Recycle_FPI(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=1000, print_err=False):
        """
        solve recycle by fixed point iteration method
        """
        x = F_source.n
        error = self.recycle_solver(x, F_source, F_sink, flow_method)
        count = 0

        while error >= eps and count < max_iter:
            error = self.recycle_solver(x, F_source, F_sink, flow_method)
            g_x = F_source.n
            x = g_x
            count += 1
            if print_err and count % 100 == 0:
                print(f"  FPI iter {count}: error={error:.2e}, n={float(np.ravel(x)[0]):.4f}")

        if print_err:
            if count >= max_iter:
                print(f'⚠️  Max FPI iterations ({max_iter}) reached, error: {error:.2e}')
            else:
                print(f'✓ FPI converged in {count} iterations, error: {error:.2e}')

        return count, error

    def Recycle_Wegstein_q(self, F_source, F_sink, flow_method='seq_sol', q=-5, eps=1e-8, max_iter=1000, print_err=False):
        """
        solve recycle by Wegsteins method with specific q factor
        """
        x = F_source.n
        error = self.recycle_solver(x, F_source, F_sink, flow_method)
        count = 0

        while error >= eps and count < max_iter:
            error = self.recycle_solver(x, F_source, F_sink, flow_method)
            x = q * x + (1 - q) * F_source.n
            count += 1
            if print_err and count % 100 == 0:
                print(f"  Wegstein q iter {count}: error={error:.2e}, n={float(np.ravel(x)[0]):.4f}")

        if print_err:
            if count >= max_iter:
                print(f'⚠️  Max Wegstein q iterations ({max_iter}) reached, error: {error:.2e}')
            else:
                print(f'✓ Wegstein q converged in {count} iterations, error: {error:.2e}')

        return count, error

    def Recycle_Wegstein_Generic(self, F_source, F_sink, flow_method='seq_sol', eps=1e-8, max_iter=1000, print_err=False):
        """ """
        x = np.zeros(3, dtype=float)
        g = np.zeros(2, dtype=float)

        x[0] = F_source.n
        error = self.recycle_solver(x[0], F_source, F_sink, flow_method)
        g[0] = F_source.n
        x[1] = F_source.n
        error = self.recycle_solver(x[1], F_source, F_sink, flow_method)
        g[1] = F_source.n
        x[2] = F_source.n
        count = 0

        while error >= eps and count < max_iter:
            error = self.recycle_solver(x[2], F_source, F_sink, flow_method, print_err)
            g = np.roll(g, -1)
            g[1] = F_source.n
            x = np.roll(x, -1)
            x[2] = abs((x[0] * g[1] - x[1] * g[0]) / (x[0] + g[1] - x[1] - g[0]))
            count += 1
            if print_err and count % 100 == 0:
                print(f"  Wegstein Generic iter {count}: error={error:.2e}, n={float(np.ravel(x)[0]):.4f}")

        if print_err:
            if count >= max_iter:
                print(f'⚠️  Max Wegstein Generic iterations ({max_iter}) reached, error: {error:.2e}')
            else:
                print(f'✓ Wegstein Generic converged in {count} iterations, error: {error:.2e}')

        return (count, error)

    def Recycle_Wegstein_Generic2(self, F_source, F_sink, flow_method="seq_sol", eps=1e-8, max_iter=1000, print_err=False):
        """
        Args:
            F_source (_type_): _description_
            F_sink (_type_): _description_
            flow_method (str, optional): _description_. Defaults to "seq_sol".
            eps (_type_, optional): _description_. Defaults to 1e-8.
            print_err (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        x = np.zeros(3, dtype=float)
        f = np.zeros(2, dtype=float)
        x[0] = F_source.n
        error = self.recycle_solver(x[0], F_source, F_sink, flow_method)
        f[0] = F_source.n
        x[1] = F_source.n
        error = self.recycle_solver(x[1], F_source, F_sink, flow_method)
        f[1] = F_source.n
        x[2] = F_source.n

        A = (f[1] - f[0]) / (x[1] - x[0])
        q = A / (A - 1)

        x = np.roll(x, -1)
        x[1] = F_source.n
        x[2] = x[0] * q + (1 - q) * x[1]
        count = 0
        while error >= eps and count < max_iter:
            error = self.recycle_solver(x[2], F_source, F_sink, flow_method)

            f = np.roll(f, -1)
            f[1] = F_source.n

            A = (f[1] - f[0]) / (x[2] - x[1])
            q = A / (A - 1)
            x = np.roll(x, -1)
            x[2] = x[1] * q + (1 - q) * f[1]
            count += 1
            if print_err and count % 100 == 0:
                print(f"  Wegstein Generic 2 iter {count}: error={error:.2e}, n={float(np.ravel(x)[0]):.4f}")

        if print_err:
            if count >= max_iter:
                print(f'⚠️  Max Wegstein Generic 2 iterations ({max_iter}) reached, error: {error:.2e}')
            else:
                print(f'✓ Wegstein Generic 2 converged in {count} iterations, error: {error:.2e}')

        return (count, error)

    def Recycle_Steffensen(self, F_source, F_sink, flow_method="seq_sol", eps=1e-8, max_iter=1000, print_err=False):
        """
        Args:
            F_source (_type_): _description_
            F_sink (_type_): _description_
            flow_method (str, optional): _description_. Defaults to "seq_sol".
            eps (_type_, optional): _description_. Defaults to 1e-8.

        Returns:
            _type_: _description_
        """
        x = np.zeros(3, dtype=float)
        x[0] = F_source.n
        error = self.recycle_solver(x[0], F_source, F_sink, flow_method, False)
        count = 0

        while error >= eps and count < max_iter:
            error = self.recycle_solver(x[0], F_source, F_sink, flow_method, False)
            x[1] = F_source.n
            error = self.recycle_solver(x[1], F_source, F_sink, flow_method, False)
            x[2] = F_source.n
            denominator = x[2] - 2 * x[1] + x[0]
            # a vanishing second difference means the iteration has converged: keep the last iterate
            x[0] = x[0] - ((x[1] - x[0]) ** 2) / denominator if abs(denominator) > 1e-14 else x[2]
            count += 1
            if print_err and count % 100 == 0:
                print(f"  Steffensen iter {count}: error={error:.2e}, n={float(np.ravel(x)[0]):.4f}")

        if print_err:
            if count >= max_iter:
                print(f'⚠️  Max Steffensen iterations ({max_iter}) reached, error: {error:.2e}')
            else:
                print(f'✓ Steffensen converged in {count} iterations, error: {error:.2e}')

        return (count, error)

    def Recycle_Aitken(self, F_source, F_sink, flow_method='seq_sol', eps=1e-6, max_iter=1000, print_err=False):
        """
        Aitken's delta-squared acceleration for oscillating recycles.
        Only tracks flow rate (n).
        """
        x = F_source.n
        count = 0
        error = 1.0

        while error >= eps and count < max_iter:
            self.recycle_solver(x, F_source, F_sink, flow_method, print_err=False)
            x1 = F_source.n

            self.recycle_solver(x1, F_source, F_sink, flow_method, print_err=False)
            x2 = F_source.n

            dx1 = x1 - x
            dx2 = x2 - x1
            denominator = dx2 - dx1

            if abs(denominator) > 1e-12:
                x_next = x - (dx1 * dx1) / denominator
            else:
                x_next = x2

            error = np.abs((x_next - x) / max(abs(x_next), 1e-10))
            count += 2

            if print_err and count % 100 == 0:
                print(f"  Aitken iter {count}: error={error:.2e}, n={x_next:.4f}")

            x = x_next

        if print_err:
            if count >= max_iter:
                print(f'⚠️  Max Aitken iterations ({max_iter}) reached, error: {error:.2e}')
            else:
                print(f'✓ Aitken converged in {count} iterations, error: {error:.2e}')
        return count, error

    def Recycle_Hybrid_FPI_Aitken(self, F_source, F_sink, flow_method='seq_sol', eps=1e-6, max_iter=1000,
                                   print_err=False, n_fpi_initial=5, n_fpi_stabilize=5):
        """
        Start with FPI iterations, then switch to Aitken.

        Parameters:
        -----------
        n_fpi_initial : int
            Number of FPI iterations before first Aitken step
        n_fpi_stabilize : int
            Number of FPI steps to run when oscillation detected
        """
        x = F_source.n
        error = 1.0
        count = 0

        if print_err:
            print(f"Starting with {n_fpi_initial} FPI iterations...")

        for i in range(n_fpi_initial):
            x_prev = x
            self.recycle_solver(x, F_source, F_sink, flow_method, print_err=False)
            x = F_source.n
            count += 1
            # change between successive iterates (comparing x with F_source.n is always zero here)
            error = np.abs((x - x_prev) / max(abs(x), 1e-10))

        if print_err:
            print("Switching to Aitken acceleration...")

        error_history = []
        oscillation_count = 0

        while error >= eps and count < max_iter:
            x_old = x

            self.recycle_solver(x, F_source, F_sink, flow_method, print_err=False)
            x1 = F_source.n

            self.recycle_solver(x1, F_source, F_sink, flow_method, print_err=False)
            x2 = F_source.n

            dx1 = x1 - x
            dx2 = x2 - x1
            denominator = dx2 - dx1

            if abs(denominator) > 1e-12:
                x_next = x - (dx1 * dx1) / denominator
            else:
                x_next = x2

            error = np.abs((x_next - x) / max(abs(x_next), 1e-10))
            error_history.append(error)
            count += 2

            if len(error_history) >= 3:
                if error_history[-1] > error_history[-2] and error_history[-2] > error_history[-3]:
                    oscillation_count += 1

                    if oscillation_count >= 2:
                        if print_err:
                            print(f"  ⚠️  Oscillation detected at iter {count}, running {n_fpi_stabilize} FPI steps...")

                        for _ in range(n_fpi_stabilize):
                            self.recycle_solver(x_next, F_source, F_sink, flow_method, print_err=False)
                            x_next = F_source.n
                            count += 1

                        oscillation_count = 0
                        error_history = []
                else:
                    oscillation_count = 0

            if print_err and count % 20 == 0:
                print(f"  Iter {count}: error={error:.2e}, x={x_next:.4f}")

            x = x_next

        if count >= max_iter:
            print(f'⚠️  Max iterations reached, error: {error:.2e}')
        elif print_err:
            print(f'✓ Converged in {count} iterations, error: {error:.2e}')

        return count, error
