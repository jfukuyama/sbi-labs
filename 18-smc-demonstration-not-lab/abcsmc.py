"""ABC-SMC with a weighted mixture proposal, and the two lecture-18 models.

Everything the sampler does is in this file:

  generation 0   rejection ABC at eps_0, proposing from the prior
  generation t   propose from
                     q_t(theta) = sum_j w_{t-1,j} N(theta; theta_{t-1,j}, tau^2 I),
                 simulate, accept if d(X, x_obs) < eps_t, until N acceptances;
                 weight each accepted theta by prior(theta) / q_t(theta).

The reference ABC posteriors (reference_density, reference_masses) are only
used for plots and checks. The sampler never calls them.

Proposals are drawn in fixed-size blocks from the generator, so the results
for a given seed do not depend on whether you step one proposal at a time or
run a whole generation at once.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2, ncx2, norm


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CubicModel:
    """Example 1. Theta ~ Unif(-2.5, 2.5), X = theta^3 - 3 theta + N(0, sigma^2)."""

    dim = 1
    low, high = -2.5, 2.5

    def __init__(self, sigma=0.15, x_obs=0.0):
        self.sigma = float(sigma)
        self.x_obs = np.array([float(x_obs)])
        # Modes: the solutions of f(theta) = x_obs inside the prior (three for
        # |x_obs| < 2, e.g. -sqrt(3), 0, sqrt(3) for x_obs = 0). The regions
        # used to define "represented" modes split the axis halfway between them.
        r = np.roots([1.0, 0.0, -3.0, -float(x_obs)])
        r = np.sort(r[np.abs(r.imag) < 1e-6].real)
        r = r[(r >= self.low) & (r <= self.high)]
        r = r[np.r_[True, np.diff(r) > 1e-3]]          # merge a double root
        self.mode_locations = r[:, None]
        self.mode_labels = [f"θ ≈ {v:.2f}".replace("-", "−") for v in r]
        self.cell_edges = (r[1:] + r[:-1]) / 2

    @staticmethod
    def f(theta):
        return theta ** 3 - 3 * theta

    def simulate(self, theta, noise):
        """theta, noise: arrays of shape (n, 1). Returns X of shape (n, 1)."""
        return self.f(theta) + self.sigma * noise

    def distance(self, x):
        return np.abs(x[:, 0] - self.x_obs[0])

    def cell(self, theta):
        """Index (0, 1, 2) of the mode region containing each theta."""
        return np.searchsorted(self.cell_edges, theta[:, 0])

    # -- reference (visualization and checking only) -----------------------
    def grid(self, n=1001):
        return np.linspace(self.low, self.high, n)

    def acceptance_prob(self, theta, eps):
        """P(|X - x_obs| < eps | theta), exactly."""
        m = self.f(np.asarray(theta, dtype=float))
        x = self.x_obs[0]
        return (norm.cdf((x + eps - m) / self.sigma)
                - norm.cdf((x - eps - m) / self.sigma))

    def reference_density(self, eps, grid=None):
        """ABC posterior pi_eps(theta | x_obs) on a grid, normalized numerically."""
        g = self.grid() if grid is None else grid
        dens = self.acceptance_prob(g, eps)  # prior is flat on the grid
        return dens / np.trapezoid(dens, g)

    def reference_masses(self, eps):
        g = self.grid(20001)
        dens = self.reference_density(eps, g)
        r = self.mode_labels
        cells = self.cell(g[:, None])
        dg = g[1] - g[0]
        return np.array([dens[cells == k].sum() * dg for k in range(len(r))]) / (dens.sum() * dg)


class SquaresModel:
    """Example 2. Theta ~ Unif([-4, 4]^2), X_i = theta_i^2 + N(0, sigma^2)."""

    dim = 2
    low, high = -4.0, 4.0

    def __init__(self, sigma=0.25, x_obs=(4.0, 4.0)):
        self.sigma = float(sigma)
        self.x_obs = np.asarray(x_obs, dtype=float)
        # Modes near (+-sqrt(x1), +-sqrt(x2)), one per quadrant (they merge
        # across an axis if x_i <= 0). Quadrants define "represented" modes.
        r1, r2 = np.sqrt(np.clip(self.x_obs, 0, None))
        self.mode_locations = np.array([[-r1, -r2], [r1, -r2], [-r1, r2], [r1, r2]])
        self.mode_labels = [f"({a:+.2g}, {b:+.2g})".replace("-", "−")
                            for a, b in self.mode_locations]

    def simulate(self, theta, noise):
        return theta ** 2 + self.sigma * noise

    def distance(self, x):
        return np.sqrt(((x - self.x_obs) ** 2).sum(axis=1))

    def cell(self, theta):
        """Quadrant index, in the order of mode_labels."""
        return (theta[:, 0] > 0).astype(int) + 2 * (theta[:, 1] > 0).astype(int)

    # -- reference (visualization and checking only) -----------------------
    def grid(self, n=241):
        return np.linspace(self.low, self.high, n)

    def acceptance_prob(self, theta, eps):
        """P(||X - x_obs|| < eps | theta), from the noncentral chi-squared(2) cdf."""
        theta = np.atleast_2d(theta)
        lam = (((theta ** 2 - self.x_obs) ** 2).sum(axis=1)) / self.sigma ** 2
        q = (eps / self.sigma) ** 2
        out = np.empty(len(lam))
        zero = lam < 1e-12
        out[zero] = chi2.cdf(q, 2)
        out[~zero] = ncx2.cdf(q, 2, lam[~zero])
        return out

    def reference_density(self, eps, grid=None):
        """ABC posterior on the grid x grid; entry [i, j] is (theta1 = g[i], theta2 = g[j])."""
        g = self.grid() if grid is None else grid
        t1, t2 = np.meshgrid(g, g, indexing="ij")
        p = self.acceptance_prob(np.column_stack([t1.ravel(), t2.ravel()]), eps)
        dens = p.reshape(t1.shape)
        dg = g[1] - g[0]
        return dens / (dens.sum() * dg * dg)

    def reference_masses(self, eps):
        g = self.grid(400)  # even count: no grid point sits on an axis
        dens = self.reference_density(eps, g)
        t1, t2 = np.meshgrid(g, g, indexing="ij")
        cells = self.cell(np.column_stack([t1.ravel(), t2.ravel()]))
        flat = dens.ravel()
        return np.array([flat[cells == k].sum() for k in range(4)]) / flat.sum()


MODELS = {1: CubicModel, 2: SquaresModel}


# ---------------------------------------------------------------------------
# Tolerance schedules
# ---------------------------------------------------------------------------

SCHEDULE_RATIOS = {"medium": 0.5, "fast": 0.1}


def next_tolerance(schedule, t, eps0, eps_min):
    """Tolerance for generation t (generation 0 uses eps0).

    a list:             eps_t = schedule[t] (the last entry repeats); this is what
                        the app's replicates use, copying the tolerances you typed.
    "medium", "fast":   eps_t = max(eps_min, eps0 * ratio^t), used by checks/.
    """
    if isinstance(schedule, (list, tuple)):
        return float(schedule[min(t, len(schedule) - 1)])
    if t == 0:
        return eps0
    return max(eps_min, eps0 * SCHEDULE_RATIOS[schedule] ** t)


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

class ABCSMC:
    """ABC-SMC that can be advanced one proposal, a batch, or a generation at a time.

    Parameters
    ----------
    model : CubicModel or SquaresModel
    N : particles per generation
    kernel_sd : standard deviation tau of the Gaussian perturbation kernel
    schedule : a list of tolerances, or "medium" / "fast" (see next_tolerance)
    eps0, eps_min : first tolerance and smallest allowed tolerance
    n_generations : number of generations, counting generation 0
    max_sims_per_generation : simulation budget; a generation that exceeds it stops
    """

    def __init__(self, model, N=200, kernel_sd=0.2, schedule="medium",
                 eps0=2.0, eps_min=0.1, n_generations=6, seed=1,
                 max_sims_per_generation=200_000, block_size=256, keep_log=True):
        self.model = model
        self.N = int(N)
        self.kernel_sd = float(kernel_sd)
        self.schedule = schedule
        self.eps0, self.eps_min = float(eps0), float(eps_min)
        self.n_generations = int(n_generations)
        self.seed = seed
        self.max_sims = int(max_sims_per_generation)
        self.block_size = int(block_size)
        self.keep_log = keep_log

        self.rng = np.random.default_rng(seed)
        self.history = []          # one dict per completed generation
        self.total_sims = 0
        self.stopped_reason = None  # "done" or "budget" once the run has ended
        self._begin_generation(0, prev=None)

    # -- state --------------------------------------------------------------
    @property
    def finished(self):
        return self.stopped_reason is not None

    @property
    def generation_complete(self):
        return self.phase == "complete"

    def _begin_generation(self, t, prev, eps=None):
        self.t = t
        self.eps = (float(eps) if eps is not None else
                    next_tolerance(self.schedule, t, self.eps0, self.eps_min))
        self.prev = prev               # (theta, w) of generation t-1, or None
        self.phase = "proposing"
        self.n_acc = 0
        self.n_sims = 0
        self.n_props = 0
        self._buffer = None
        self._pos = 0
        self.log = {k: [] for k in ("theta", "x", "d", "accepted", "ancestor",
                                    "source", "logw")}
        self.acc_theta, self.acc_d, self.acc_logw = [], [], []

    # -- the proposal distribution -------------------------------------------
    def log_q(self, theta):
        """log q_t(theta) for an (n, dim) array."""
        theta = np.atleast_2d(theta)
        log_prior = self._log_prior(theta)
        if self.prev is None:
            return log_prior  # generation 0 proposes from the prior
        th_prev, w_prev = self.prev
        tau, d = self.kernel_sd, self.model.dim
        out = np.empty(len(theta))
        logw = np.log(w_prev)
        for s in range(0, len(theta), 2000):   # chunks keep memory bounded
            diff = theta[s:s + 2000, None, :] - th_prev[None, :, :]
            log_k = (-0.5 * (diff ** 2).sum(axis=2) / tau ** 2
                     - d * np.log(tau * np.sqrt(2 * np.pi)))
            out[s:s + 2000] = logsumexp(log_k + logw[None, :], axis=1)
        return out

    def _log_prior(self, theta):
        m = self.model
        inside = np.all((theta >= m.low) & (theta <= m.high), axis=1)
        return np.where(inside, -m.dim * np.log(m.high - m.low), -np.inf)

    def proposal_density_grid(self, grid):
        """q_t on a plotting grid. 1D: values at grid. 2D: matrix [i, j] at (g[i], g[j])."""
        m = self.model
        inside = ((grid >= m.low) & (grid <= m.high)).astype(float)
        if m.dim == 2:
            inside = np.outer(inside, inside)
        prior_val = inside * (1 / (m.high - m.low)) ** m.dim
        if self.prev is None:  # generation 0 proposes from the prior
            return prior_val
        th_prev, w_prev = self.prev
        tau = self.kernel_sd
        if m.dim == 1:
            local = (w_prev[None, :] *
                     norm.pdf(grid[:, None], th_prev[None, :, 0], tau)).sum(axis=1)
        else:  # the Gaussian kernel factorizes over coordinates
            A = norm.pdf(grid[None, :], th_prev[:, 0, None], tau)
            B = norm.pdf(grid[None, :], th_prev[:, 1, None], tau)
            local = (A * w_prev[:, None]).T @ B
        return local

    # -- drawing proposals ----------------------------------------------------
    def _draw_block(self):
        m, B, rng = self.model, self.block_size, self.rng
        if self.prev is None:
            theta = rng.uniform(m.low, m.high, size=(B, m.dim))  # the prior
            ancestor = np.full(B, -1)
            source = np.full(B, "prior", dtype=object)
        else:
            th_prev, w_prev = self.prev
            ancestor = rng.choice(len(w_prev), size=B, p=w_prev)
            z = rng.standard_normal((B, m.dim))
            theta = th_prev[ancestor] + self.kernel_sd * z
            source = np.full(B, "local", dtype=object)
        noise = rng.standard_normal((B, m.dim))
        inside = np.isfinite(self._log_prior(theta))
        x = m.simulate(theta, noise)
        x[~inside] = np.nan          # outside the prior: never simulated
        d = m.distance(x)
        d[~inside] = np.inf
        source[~inside] = "outside prior"
        self._buffer = dict(theta=theta, x=x, d=d, ancestor=ancestor, source=source,
                            inside=inside)
        self._pos = 0

    def _consume(self, max_count):
        """Process up to max_count proposals from the buffer; stop at N acceptances."""
        processed = []
        remaining = max_count
        while remaining > 0 and self.phase == "proposing":
            if self._buffer is None or self._pos >= self.block_size:
                self._draw_block()
            b, s = self._buffer, self._pos
            e = min(self.block_size, s + remaining)
            acc = b["d"][s:e] < self.eps
            sims = b["inside"][s:e]
            # cut the slice where we reach N acceptances or the simulation budget
            cut = e - s
            need = self.N - self.n_acc
            cum_acc = np.cumsum(acc)
            if cum_acc[-1] >= need:
                cut = min(cut, int(np.argmax(cum_acc >= need)) + 1)
            cum_sims = np.cumsum(sims)
            budget_left = self.max_sims - self.n_sims
            if cum_sims[-1] >= budget_left:
                cut = min(cut, int(np.argmax(cum_sims >= budget_left)) + 1)
            e = s + cut
            sl = slice(s, e)
            acc = acc[:cut]
            theta = b["theta"][sl]
            logw = np.full(cut, np.nan)
            if acc.any():
                logw[acc] = self._log_prior(theta[acc]) - self.log_q(theta[acc])
                self.acc_theta.append(theta[acc])
                self.acc_d.append(b["d"][sl][acc])
                self.acc_logw.append(logw[acc])
            if self.keep_log:
                for k in ("theta", "x", "d", "ancestor", "source"):
                    self.log[k].append(b[k][sl])
                self.log["accepted"].append(acc)
                self.log["logw"].append(logw)
            self.n_acc += int(acc.sum())
            self.n_sims += int(sims[:cut].sum())
            self.n_props += cut
            self.total_sims += int(sims[:cut].sum())
            self._pos = e
            remaining -= cut
            processed.append(cut)
            if self.n_acc >= self.N:
                self._complete_generation(budget_hit=False)
            elif self.n_sims >= self.max_sims:
                self._complete_generation(budget_hit=True)
        return sum(processed)

    def _complete_generation(self, budget_hit):
        theta = (np.concatenate(self.acc_theta) if self.acc_theta
                 else np.empty((0, self.model.dim)))
        logw = np.concatenate(self.acc_logw) if self.acc_logw else np.empty(0)
        dists = np.concatenate(self.acc_d) if self.acc_d else np.empty(0)
        if len(theta):
            w = np.exp(logw - logsumexp(logw))
        else:
            w = np.empty(0)
        cells = self.model.cell(theta) if len(theta) else np.empty(0, dtype=int)
        n_cells = len(self.model.mode_labels)
        self.history.append(dict(
            t=self.t, eps=self.eps, theta=theta, w=w, logw=logw, distances=dists,
            ess=1.0 / np.sum(w ** 2) if len(w) else 0.0,
            n_accepted=len(theta), n_sims=self.n_sims, n_proposals=self.n_props,
            acc_rate=len(theta) / max(self.n_sims, 1),
            counts=np.bincount(cells, minlength=n_cells),
            masses=np.bincount(cells, weights=w, minlength=n_cells),
            total_sims=self.total_sims, budget_hit=budget_hit,
            ended_early=len(theta) < self.N,
            prev=self.prev, kernel_sd=self.kernel_sd,
        ))
        self.phase = "complete"
        if budget_hit:
            self.stopped_reason = "budget"
        elif self.t + 1 >= self.n_generations:
            self.stopped_reason = "done"

    # -- public stepping interface -------------------------------------------
    def end_generation(self):
        """End the current generation now, keeping the particles accepted so far.

        The weights are pi/q_t as usual; the next generation simply has fewer
        particles to build its proposal from. Needs at least one acceptance.
        """
        if self.phase != "proposing" or self.n_acc == 0:
            return False
        self._complete_generation(budget_hit=False)
        return True

    def start_next_generation(self, eps=None):
        """Move from a completed generation to the next one (no proposals drawn).

        eps, if given, overrides the schedule for the new generation.
        """
        if self.phase != "complete" or self.finished:
            return False
        last = self.history[-1]
        self._begin_generation(self.t + 1, prev=(last["theta"], last["w"]), eps=eps)
        return True

    def step(self, n=1):
        """Process up to n proposals in the current generation."""
        if self.phase != "proposing":
            return 0
        return self._consume(n)

    def run_generation(self):
        while self.phase == "proposing":
            self._consume(10 ** 9)

    def run_all(self):
        while not self.finished:
            if self.phase == "complete":
                self.start_next_generation()
            self.run_generation()
        return self

    # -- the current generation's proposals as arrays ------------------------
    def current_log(self):
        if not self.keep_log or not self.log["theta"]:
            return None
        return {k: np.concatenate(v) for k, v in self.log.items()}

    def current_population(self):
        """Accepted particles of the current generation, with normalized weights so far."""
        if not self.acc_theta:
            return np.empty((0, self.model.dim)), np.empty(0)
        theta = np.concatenate(self.acc_theta)
        logw = np.concatenate(self.acc_logw)
        return theta, np.exp(logw - logsumexp(logw))


# ---------------------------------------------------------------------------
# Repeated runs
# ---------------------------------------------------------------------------

WEIGHT_THRESHOLD = 0.05  # a mode "has weight" if its estimated mass is at least this


def run_replicates(model, n_reps=20, first_seed=1, **settings):
    """Run n_reps independent ABC-SMC runs (seeds first_seed, first_seed+1, ...).

    A mode is *found* if at least one final particle lies in its region
    (the quadrant in Example 2), and *weighted* if its estimated posterior mass
    is at least WEIGHT_THRESHOLD.
    """
    rows = []
    for r in range(n_reps):
        s = ABCSMC(model, seed=first_seed + r, keep_log=False, **settings).run_all()
        h = s.history[-1]
        rows.append(dict(seed=first_seed + r, counts=h["counts"], masses=h["masses"],
                         found=int((h["counts"] > 0).sum()),
                         weighted=int((h["masses"] >= WEIGHT_THRESHOLD).sum()),
                         total_sims=s.total_sims, ess=h["ess"], final_eps=h["eps"],
                         budget_hit=s.stopped_reason == "budget",
                         generations=len(s.history)))
    return rows
