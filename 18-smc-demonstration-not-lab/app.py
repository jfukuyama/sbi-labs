"""Interactive displays for the lecture-18 ABC-SMC demonstrations.

    from app import PriorDraws, Demo
    PriorDraws(1)   # draw from the prior and the simulator, no rejection
    Demo(1)         # Example 1: X = theta^3 - 3 theta + noise
    Demo(2)         # Example 2: X_i = theta_i^2 + noise, four modes

The algorithm lives in abcsmc.py; this file only draws pictures and wires up
buttons. Figures are rendered to PNG and shown in an ipywidgets Image, so the
app works with the ordinary inline matplotlib backend.
"""

import io

import ipywidgets as W
import numpy as np
from IPython.display import display
from matplotlib.figure import Figure

from abcsmc import ABCSMC, MODELS

BLUE, ORANGE, GREEN = "#386b94", "#d98032", "#528c73"
PURPLE, GRAY, DARK, LIGHT = "#8a5a9e", "#777777", "#333333", "#c4c4c4"

RC = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
      "axes.spines.top": False, "axes.spines.right": False}

# Starting settings. eps_min sets the fixed height of Example 1's density axis
# (the smallest tolerance the walkthrough uses). N is not used by the app; it is
# the number of particles per generation in checks/check_presets.py.
DEFAULTS = {
    1: dict(N=200, kernel_sd=0.2, eps0=2.0, eps_min=0.1, seed=1),
    2: dict(N=200, kernel_sd=0.2, eps0=3.0, eps_min=0.25, seed=1),
}

# Interactive runs have no particle target: a generation ends when you press
# "Next generation".
UNLIMITED = 10 ** 9

# Example 2 colors rejected proposals by their distance to x_obs (log scale,
# 0.1 to 10): dark = close, light = far.
DIST_SCALE = [[0.0, "#08214a"], [0.45, "#3f78b0"], [1.0, "#dfe6ee"]]
DIST_TICKS = [0.1, 0.3, 1, 3, 10]

SETTING_KEYS = ["kernel_sd", "eps0", "seed"]


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    return buf.getvalue()


def _sizes(w, N, base=28.0, cap=600.0):
    """Marker areas proportional to weight; equal weights give `base`."""
    return np.clip(base * np.asarray(w) * N, 4.0, cap)


class Demo:
    """One example's controls and plots."""

    def __init__(self, example=1, show=True):
        self.ex = example
        self.ModelCls = MODELS[example]
        self._ref_cache = {}
        self._build_widgets()
        self.reset()
        if show:
            display(self.ui)

    # ------------------------------------------------------------------ widgets
    def _build_widgets(self):
        d = DEFAULTS[self.ex]
        lay = W.Layout(width="230px")
        style = {"description_width": "95px"}
        self.w = {
            "kernel_sd": W.BoundedFloatText(value=d["kernel_sd"], min=0.001, max=20, step=0.05,
                                            description="kernel τ", layout=lay, style=style),
            "eps0": W.BoundedFloatText(value=d["eps0"], min=0.01, max=20, step=0.1,
                                       description="ε₀", layout=lay, style=style),
            "seed": W.IntText(value=d["seed"], description="seed", layout=lay, style=style),
        }
        for wid in self.w.values():
            wid.observe(self._settings_changed, names="value")
        # observed data: changing it restarts the run from generation 0
        if self.ex == 1:
            self.xobs_w = [W.BoundedFloatText(value=0.0, min=-8, max=8, step=0.5,
                                              description="x_obs", layout=lay, style=style)]
        else:
            self.xobs_w = [W.BoundedFloatText(value=v, min=-1, max=16, step=0.5,
                                              description=f"x_obs,{i}", layout=lay,
                                              style=style)
                           for i, v in ((1, 4.0), (2, 4.0))]
        for wid in self.xobs_w:
            wid.observe(lambda _: self.reset(), names="value")


        def button(label, cb, width="150px", style=""):
            b = W.Button(description=label, layout=W.Layout(width=width, height="38px"),
                         button_style=style)
            b.on_click(cb)
            return b

        self.b_reset = button("Reset", self.reset, style="warning")
        self.b_one = button("+1 proposal", lambda _: self._advance("one"))
        self.batch = W.BoundedIntText(value=100, min=1, max=200_000,
                                      layout=W.Layout(width="100px"))
        self.b_batch = button("+ batch of", lambda _: self._advance("batch"), "110px")
        self.b_next = button("Next generation ▸", lambda _: self._advance("next"),
                             "170px", style="info")
        self.next_eps = W.BoundedFloatText(value=1.0, min=0.001, max=50, step=0.05,
                                           description="with ε =",
                                           layout=W.Layout(width="170px"),
                                           style={"description_width": "60px"})

        self.show_solutions = W.Checkbox(value=True, description="show solutions",
                                         indent=False, layout=W.Layout(width="150px"))
        self.show_solutions.observe(lambda _: self.render(), names="value")
        self.show_ref = W.Checkbox(value=False, description="show exact ABC posterior with tolerance ε",
                                   indent=False, layout=W.Layout(width="340px"))
        self.show_ref.observe(lambda _: self.render(), names="value")
        self.view = W.ToggleButtons(options=["particles", "proposal", "side by side"],
                                    value="side by side")
        self.view.observe(lambda _: self.render(), names="value")

        if self.ex == 1:
            self.image = W.Image(format="png", layout=W.Layout(width="100%"))
            self.figure_area = self.image
        else:
            # Example 2: interactive plotly maps (hover, click) + acceptance panel
            self.pfig, self._pviews = None, None
            self.map_box = W.Box()
            self.image = W.Image(format="png", layout=W.Layout(width="330px"))
            self.figure_area = W.HBox([self.map_box, self.image])
        self.status = W.HTML(layout=W.Layout(width="100%"))
        self.changed_note = W.HTML()

        css = W.HTML("<style>.abcsmc, .abcsmc .widget-label, .abcsmc button, "
                     ".abcsmc select, .abcsmc input, .abcsmc .widget-readout "
                     "{font-size: 15px !important;}</style>")
        ws = self.w
        controls = W.VBox([
            W.HBox([ws["kernel_sd"], ws["eps0"], ws["seed"]]),
            W.HBox(self.xobs_w),
            self.changed_note,
        ])
        steps = W.HBox([self.b_reset, self.b_one, self.b_batch, self.batch,
                        self.b_next, self.next_eps])
        extras = [self.show_solutions] if self.ex == 1 else [self.view]
        # the posterior toggle sits right under the figure (below the density panel)
        self.ui = W.VBox([css, controls, steps, W.HBox(extras), self.figure_area,
                          W.HBox([self.show_ref], layout=W.Layout(padding="0 0 0 60px")),
                          self.status])
        self.ui.add_class("abcsmc")

    # ----------------------------------------------------------- control logic
    def settings(self):
        return {k: self.w[k].value for k in SETTING_KEYS}

    def _settings_changed(self, _=None):
        self._update_changed_note()

    def _update_changed_note(self):
        if self.settings() != getattr(self, "applied", None):
            self.changed_note.value = ("<b style='color:#b0533c'>Settings changed: "
                                       "press Reset to apply.</b>")
        else:
            self.changed_note.value = ""

    def reset(self, _=None):
        self.applied = self.settings()
        self.model = self._new_model()
        self.smc = ABCSMC(self.model, N=UNLIMITED, n_generations=UNLIMITED, **self.applied)
        self.last_action = "reset"
        # fixed axis limits for this run
        if self.ex == 1:
            ref = self.reference(min(DEFAULTS[1]["eps_min"], self.applied["eps0"]))
            self.dens_ymax = 1.3 * ref.max()
        self._prefill_next_eps()
        self._update_changed_note()
        self.render()

    def _advance(self, kind):
        s = self.smc
        if kind == "next":
            # end the generation with whatever has been accepted, then move on
            s.end_generation()
            s.start_next_generation(eps=self.next_eps.value)
            self._prefill_next_eps()
            self.last_action = "next"
        elif s.phase == "proposing":
            s.step(1 if kind == "one" else self.batch.value)
            self.last_action = kind
        self.render()

    def _new_model(self):
        x = [w.value for w in self.xobs_w]
        return self.ModelCls(x_obs=x[0] if self.ex == 1 else tuple(x))

    def _prefill_next_eps(self):
        """Suggest the next tolerance: half the current one."""
        self.next_eps.value = round(self.smc.eps / 2, 6)

    # --------------------------------------------------------------- helpers
    def reference(self, eps):
        key = (round(eps, 10), self.model.sigma, tuple(self.model.x_obs))
        if key not in self._ref_cache:
            self._ref_cache[key] = self.model.reference_density(eps)
        return self._ref_cache[key]

    def _population_now(self):
        """(theta, w, label) for the bar chart: accepted so far, else last generation."""
        s = self.smc
        th, w = s.current_population()
        if len(th):
            return th, w, f"generation {s.t}" + ("" if s.generation_complete
                                                 else ", so far")
        if s.prev is not None:
            return s.prev[0], s.prev[1], f"generation {s.t - 1}"
        return th, w, "no particles yet"

    # ------------------------------------------------------------- rendering
    def render(self):
        s = self.smc
        log = s.current_log()
        n = 0 if log is None else len(log["d"])
        sel = n - 1 if n else None        # the highlighted proposal: always the latest
        if self.ex == 1:
            self.image.value = _png(self._figure_ex1(log, sel))
        else:
            self._render_ex2(log, sel)
        self.status.value = self._status_html(log, sel)
        proposing = s.phase == "proposing"
        for b in (self.b_one, self.b_batch):
            b.disabled = not proposing
        self.b_next.disabled = s.finished or (proposing and s.n_acc == 0)

    def _status_html(self, log, sel):
        s = self.smc
        _, w, _ = self._population_now()
        ess = 1 / np.sum(w ** 2) if len(w) else 0
        acc = f"{s.n_acc / s.n_sims:.1%}" if s.n_sims else "–"
        q_desc = ("the prior" if s.t == 0 else
                  f"Σ w<sub>j</sub> N(θ<sub>j</sub>, {s.kernel_sd:g}²)")

        # what just happened
        if s.finished and s.generation_complete:
            why = ("Run finished." if s.stopped_reason == "done" else
                   f"<b>Simulation budget ({s.max_sims:,} per generation) exhausted</b> "
                   f"with {s.n_acc} particles accepted; the run stops here.")
            step = (f"{why} Generation {s.t} accepted {s.n_acc} particles from "
                    f"{s.n_sims:,} simulations.")
        elif s.generation_complete:
            step = (f"<b>Generation {s.t} complete.</b> {s.n_acc} particles accepted from "
                    f"{s.n_sims:,} simulations. Each has weight π(θ)/q<sub>{s.t}</sub>(θ), "
                    "normalized to sum to 1." if s.t > 0 else
                    f"<b>Generation 0 complete.</b> Rejection ABC: {s.n_acc} accepted from "
                    f"{s.n_sims:,} simulations, all with equal weight.")
            step += (" Press <b>Next generation</b> to build the next proposal from "
                     "these weighted particles.")
        elif n_props(log) == 0:
            if s.t == 0:
                step = (f"<b>Generation 0: rejection ABC.</b> Draw θ from {q_desc}, simulate X, "
                        f"keep θ if d(X, x<sub>obs</sub>) &lt; ε<sub>0</sub> = {s.eps:g}.")
            else:
                prev_eps = s.history[-1]["eps"]
                step = (f"<b>Generation {s.t}.</b> The proposal is q<sub>{s.t}</sub> = {q_desc}, "
                        f"built from the {len(s.prev[1])} weighted particles of generation "
                        f"{s.t - 1}. The tolerance goes from {prev_eps:g} to "
                        f"ε<sub>{s.t}</sub> = {s.eps:g}.")
        else:
            step = (f"<b>Generation {s.t}</b>, proposing from q<sub>{s.t}</sub> = {q_desc}, "
                    f"ε<sub>{s.t}</sub> = {s.eps:g}.")
            if s.n_acc:
                step += (f" <b>Next generation ▸</b> ends this generation with its "
                         f"{s.n_acc} accepted particles.")

        diag = (f"generation <b>{s.t}</b> &nbsp;|&nbsp; ε = <b>{s.eps:g}</b> "
                f"&nbsp;|&nbsp; accepted {s.n_acc} &nbsp;|&nbsp; "
                f"simulations this generation {s.n_sims:,} (total {s.total_sims:,}) "
                f"&nbsp;|&nbsp; acceptance rate {acc} &nbsp;|&nbsp; ESS {ess:.0f}")
        insp = self._latest_html(log, sel)
        return (f"<div style='font-size:17px; line-height:1.45'>{step}<br>{diag}"
                f"{insp}</div>")

    def _latest_html(self, log, sel):
        if log is None or sel is None:
            return ""
        s, m = self.smc, self.model
        fmt = lambda v: ", ".join(f"{x:.3f}" for x in np.atleast_1d(v))
        th = log["theta"][sel]
        src = log["source"][sel]
        anc = log["ancestor"][sel]
        if s.t == 0:
            how = "drawn from the generation-0 search distribution"
        else:
            how = (f"ancestor j = {anc} (θ<sub>j</sub> = ({fmt(s.prev[0][anc])}), "
                   f"weight {s.prev[1][anc]:.4f}) perturbed by N(0, {s.kernel_sd:g}²)")
        if src == "outside prior":
            res = "outside the prior: <b>rejected without simulating</b>"
        else:
            ok = bool(log["accepted"][sel])
            res = (f"simulated X = ({fmt(log['x'][sel])}), x<sub>obs</sub> = "
                   f"({fmt(m.x_obs)}), d = {log['d'][sel]:.3f} vs ε = {s.eps:g}: "
                   + ("<b style='color:#386b94'>accepted</b>" if ok else
                      "<b style='color:#777'>rejected</b>"))
            if ok:
                lw = log["logw"][sel]
                acc_lw = log["logw"][log["accepted"]]
                share = np.exp(lw - np.logaddexp.reduce(acc_lw))
                res += (f"; weight π/q = {np.exp(lw):.3g}" if s.t > 0 else
                        "; equal weight") + f" (normalized {share:.4f})"
        return (f"<div style='margin-top:6px; padding:6px 10px; background:#f3f1ec'>"
                f"Most recent proposal (#{sel} of this generation): θ* = ({fmt(th)}), {how}. "
                f"{res}.</div>")

    # ---- shared right-hand panels -------------------------------------------
    def _acc_panel(self, ax):
        s = self.smc
        T = max(6, s.t + 1)  # rows/bars reserved; grows past 6
        rates = [h["acc_rate"] for h in s.history]
        x = np.arange(len(rates))
        ax.bar(x, rates, color=GRAY, width=0.6)
        if s.phase == "proposing" and s.n_sims:
            ax.bar([s.t], [max(s.n_acc / s.n_sims, 1e-4)], color=LIGHT, width=0.6)
        for xi, r in zip(x, rates):
            ax.text(xi, r * 1.2, f"{r:.0%}" if r >= 0.01 else f"{r:.1%}",
                    ha="center", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.5)
        ax.set_xlim(-0.6, T - 0.4)
        ax.set_xticks(range(T))
        ax.set_xlabel("generation")
        ax.set_title("Acceptance rate (log scale)", fontsize=12)

    # ---- Example 1 ------------------------------------------------------------
    def _figure_ex1(self, log, sel):
        import matplotlib
        with matplotlib.rc_context(RC):
            fig = Figure(figsize=(15, 9.8), layout="constrained")
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 0.75], height_ratios=[1.1, 1.15])
            ax_top = fig.add_subplot(gs[0, 0])
            ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
            self._ex1_data_panel(ax_top, log, sel)
            self._ex1_theta_panel(ax_bot, log, sel)
            self._acc_panel(fig.add_subplot(gs[:, 1]))
        return fig

    def _ex1_data_panel(self, ax, log, sel):
        m, s = self.model, self.smc
        xo = m.x_obs[0]
        ylo, yhi = min(-4.5, xo - 2.5), max(4.5, xo + 2.5)
        g = m.grid(600)
        ax.axhspan(m.x_obs[0] - s.eps, m.x_obs[0] + s.eps, color=ORANGE, alpha=0.18, lw=0)
        ax.axhline(m.x_obs[0], color=ORANGE, lw=2)
        ax.text(m.low + 0.04, m.x_obs[0] + s.eps + 0.05,
                f"$x_{{obs}} \\pm \\epsilon$,  ε = {s.eps:g}", color=ORANGE,
                ha="left", va="bottom", fontsize=13)
        if self.show_solutions.value:
            for r in m.mode_locations[:, 0]:
                ax.axvline(r, color=LIGHT, lw=1.2, ls="--")
        ax.plot(g, m.f(g), color=DARK, lw=2, label="f(θ) = θ³ − 3θ (noise-free)")
        if log is not None:
            keep = slice(max(0, len(log["d"]) - 3000), None)  # draw at most 3000
            th = log["theta"][keep, 0]
            x = log["x"][keep, 0]
            acc = log["accepted"][keep]
            ok = np.isfinite(x)
            xc = np.clip(x, ylo + 0.1, yhi - 0.1)
            off = ok & (x != xc)
            rej = ok & ~acc
            ax.scatter(th[rej & ~off], xc[rej & ~off], s=14, facecolor="none",
                       edgecolor=GRAY, lw=0.8, alpha=0.6, label="rejected")
            ax.scatter(th[rej & off], xc[rej & off], s=14, color=GRAY, alpha=0.4,
                       marker=6, lw=0)
            ax.scatter(th[acc], x[acc], s=22, color=BLUE, label="accepted", zorder=3)
            if sel is not None and np.isfinite(log["x"][sel, 0]):
                t0, x0 = log["theta"][sel, 0], log["x"][sel, 0]
                ax.plot([t0, t0], [m.f(t0), np.clip(x0, ylo, yhi)], color=GREEN, lw=1.5)
                ax.scatter([t0], [np.clip(x0, ylo + 0.1, yhi - 0.1)], s=260,
                           facecolor="none", edgecolor=GREEN, lw=2.5, zorder=4,
                           label="most recent proposal")
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(m.low, m.high)
        ax.set_ylabel("simulated X")
        ax.set_title(f"Generation {s.t}: proposals θ and their simulated X "
                     f"(noise σ = {m.sigma:g})")
        ax.legend(loc="upper center", fontsize=11, frameon=False, ncol=2)
        ax.tick_params(labelbottom=False)

    def _ex1_theta_panel(self, ax, log, sel):
        m, s = self.model, self.smc
        g = m.grid()
        ymax = self.dens_ymax
        q = s.proposal_density_grid(g)
        th, w = s.current_population()
        if len(th):
            bins = np.linspace(m.low, m.high, 101)
            h, _ = np.histogram(th[:, 0], bins=bins, weights=w, density=True)
            ax.stairs(h, bins, fill=True, color=BLUE, alpha=0.35,
                      label="accepted, weighted (this generation)")
            if h.max() > ymax:
                ax.text(m.high, ymax, "histogram\noff scale ↑", fontsize=10,
                        color=BLUE, ha="right", va="top")
        ax.plot(g, np.full_like(g, 1 / (m.high - m.low)), color=GRAY, ls=":", lw=1.5,
                label="prior")
        ax.plot(g, q, color=PURPLE, lw=2, label=f"proposal q{sub(s.t)}")
        if self.show_ref.value:
            ax.plot(g, self.reference(s.eps), color=DARK, lw=2, ls="--",
                    label=f"exact ABC posterior, ε={s.eps:g}")

        # one row of particles per generation, generation 0 at the top
        T = max(6, s.t + 1)  # rows/bars reserved; grows past 6
        row_h = min(0.1, 0.6 / T) * ymax
        row_y = lambda k: -(k + 0.9) * row_h
        pops = [(h["theta"], h["w"]) for h in s.history[:s.t]] + [(th, w)]
        for k, (th_k, w_k) in enumerate(pops):
            col = BLUE if k == s.t else PURPLE if k == s.t - 1 else GRAY
            if len(th_k):
                ax.scatter(th_k[:, 0], np.full(len(w_k), row_y(k)), s=_sizes(w_k, len(w_k)),
                           color=col, alpha=0.5 if k >= s.t - 1 else 0.35, lw=0)
            ax.text(m.high, row_y(k), f" gen {k}", va="center", fontsize=10,
                    color=col, clip_on=False)
        if sel is not None:
            t0 = log["theta"][sel, 0]
            anc = log["ancestor"][sel]
            if anc >= 0 and s.prev is not None:
                a0 = s.prev[0][anc, 0]
                ax.annotate("", xy=(t0, row_y(s.t)), xytext=(a0, row_y(s.t - 1)),
                            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
                ax.scatter([a0], [row_y(s.t - 1)], s=180, facecolor="none",
                           edgecolor=GREEN, lw=2)
            ax.scatter([t0], [row_y(s.t)], s=180, facecolor="none", edgecolor=GREEN, lw=2)
        ax.axhline(0, color=LIGHT, lw=0.8)
        ax.set_ylim(row_y(T - 1) - 0.6 * row_h, ymax)
        ax.set_yticks([v for v in ax.get_yticks() if 0 <= v <= ymax])
        ax.set_xlabel("θ")
        ax.set_ylabel("density")
        ax.set_title("Parameter axis; below: particles of each generation "
                     "(dot area ∝ weight)", fontsize=13)
        ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2,
                  bbox_to_anchor=(0, 1.02))

    # ---- Example 2 (interactive plotly maps + matplotlib acceptance panel) ----
    def _render_ex2(self, log, sel):
        import matplotlib
        views = {"particles": ["particles"], "proposal": ["proposal"],
                 "side by side": ["particles", "proposal"]}[self.view.value]
        if self.pfig is None or self._pviews != views:
            self._build_pfig(views)
        self._update_pfig(log, sel)
        with matplotlib.rc_context(RC):
            fig = Figure(figsize=(4.6, 6.4), layout="constrained")
            self._acc_panel(fig.add_subplot())
        self.image.value = _png(fig)

    def _build_pfig(self, views):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        m = self.model
        n = len(views)
        base = make_subplots(rows=1, cols=n, horizontal_spacing=0.09,
                             subplot_titles=[" "] * n)
        self._ptr = []
        for c, v in enumerate(views, start=1):
            idx = {}

            def add(trace, key):
                base.add_trace(trace, row=1, col=c)
                idx[key] = len(base.data) - 1

            add(go.Heatmap(x=m.grid(), y=m.grid(), z=None, showscale=False,
                           hoverinfo="skip", zmin=0,
                           colorscale="Greys" if v == "particles" else "Purples"), "heat")
            add(go.Scatter(x=[], y=[], mode="markers", hoverinfo="text",
                           name="rejected: no outline (color = distance to x_obs)",
                           showlegend=(c == 1),
                           marker=dict(size=7, color=[], colorscale=DIST_SCALE,
                                       cmin=-1, cmax=1, showscale=(c == n),
                                       colorbar=dict(
                                           title=dict(text="d(X, x_obs)", side="right"),
                                           tickvals=np.log10(DIST_TICKS),
                                           ticktext=[f"{t:g}" for t in DIST_TICKS],
                                           len=0.85, thickness=14))), "rej")
            add(go.Scatter(x=[], y=[], mode="markers", hoverinfo="text",
                           name="accepted: orange outline, size ∝ weight",
                           showlegend=(c == 1),
                           marker=dict(color=[], colorscale=DIST_SCALE, cmin=-1, cmax=1,
                                       line=dict(color=ORANGE, width=2.5))),
                "acc")
            add(go.Scatter(x=[], y=[], mode="markers", name="previous generation",
                           showlegend=(v == "proposal"), hoverinfo="text",
                           marker=dict(symbol="circle-open", color=PURPLE,
                                       line=dict(width=1.5))), "prev")
            add(go.Scatter(x=[], y=[], mode="lines+markers", hoverinfo="skip",
                           name="most recent proposal (ancestor → proposal)",
                           showlegend=(c == 1),
                           line=dict(color=GREEN, width=2.5),
                           marker=dict(symbol="circle-open", size=[], color=GREEN,
                                       line=dict(width=3))), "sel")
            add(go.Scatter(x=m.mode_locations[:, 0], y=m.mode_locations[:, 1],
                           mode="markers", hoverinfo="skip", showlegend=False,
                           marker=dict(symbol="cross-thin", size=14,
                                       line=dict(color=DARK, width=2))), "modes")
            ax = "" if c == 1 else str(c)
            base.update_xaxes(range=[m.low, m.high], title_text="θ₁", zeroline=True,
                              zerolinecolor=LIGHT, showgrid=False, row=1, col=c)
            base.update_yaxes(range=[m.low, m.high], title_text="θ₂", zeroline=True,
                              zerolinecolor=LIGHT, showgrid=False,
                              scaleanchor=f"x{ax}", scaleratio=1, row=1, col=c)
            self._ptr.append((v, idx))
        base.update_layout(height=640, width=560 * n + 120, plot_bgcolor="white",
                           margin=dict(l=60, r=20, t=50, b=60), font=dict(size=14),
                           legend=dict(orientation="h", y=-0.13, x=0),
                           hoverlabel=dict(font_size=15))
        self.pfig = go.FigureWidget(base)
        self._pviews = views
        self.map_box.children = [self.pfig]

    def _update_pfig(self, log, sel):
        m, s = self.model, self.smc
        f = lambda v: ", ".join(f"{x:.3f}" for x in np.atleast_1d(v))
        obs = f"({m.x_obs[0]:g}, {m.x_obs[1]:g})"
        # proposals of this generation (latest 4000), outside-prior ones left out
        if log is not None:
            first = max(0, len(log["d"]) - 4000)
            ids = np.arange(first, len(log["d"]))
            ids = ids[np.isfinite(log["d"][ids])]
            acc_all = log["accepted"]
            lw_acc = log["logw"][acc_all]
            norm_c = np.logaddexp.reduce(lw_acc) if len(lw_acc) else 0.0
        else:
            ids = np.empty(0, int)

        def hover(i):
            txt = (f"proposal #{i}<br>θ = ({f(log['theta'][i])})<br>"
                   f"simulated X = ({f(log['x'][i])})<br>observed x = {obs}<br>"
                   f"distance d = {log['d'][i]:.3f}  (ε = {s.eps:g})<br>")
            if log["accepted"][i]:
                share = np.exp(log["logw"][i] - norm_c)
                return txt + f"<b>accepted</b>, normalized weight {share:.4f}"
            return txt + "rejected"

        rej = ids[~log["accepted"][ids]] if len(ids) else ids
        acc = ids[log["accepted"][ids]] if len(ids) else ids
        if len(acc):
            share = np.exp(log["logw"][acc] - norm_c)
            acc_size = np.clip(9 * np.sqrt(share * acc_all.sum()), 5, 34)
        else:
            acc_size = []
        with self.pfig.batch_update():
            for c, (v, idx) in enumerate(self._ptr, start=1):
                d = self.pfig.data
                heat = d[idx["heat"]]
                if v == "particles":
                    show = self.show_ref.value
                    z = self.reference(s.eps).T if show else None
                    heat.update(z=z, visible=show, zmax=None if z is None else z.max() * 1.6)
                    title = f"Generation {s.t}: proposals" + (
                        f" and exact ABC posterior at ε = {s.eps:g}" if show else "")
                else:
                    z = s.proposal_density_grid(m.grid()).T
                    heat.update(z=z, visible=True, zmax=z.max())
                    title = f"Proposal q{sub(s.t)}" + (" = prior" if s.t == 0 else
                                                        " and previous generation")
                self.pfig.layout.annotations[c - 1].text = title
                if v == "proposal" and s.prev is not None:
                    pt, pw = s.prev
                    d[idx["prev"]].update(
                        x=pt[:, 0], y=pt[:, 1],
                        marker=dict(size=np.clip(9 * np.sqrt(pw * len(pw)), 4, 34)),
                        hovertext=[f"generation {s.t - 1} particle {j}<br>"
                                   f"θ = ({f(pt[j])})<br>weight {pw[j]:.4f}"
                                   for j in range(len(pw))])
                else:
                    d[idx["prev"]].update(x=[], y=[], hovertext=[])
                d[idx["rej"]].update(
                    x=log["theta"][rej, 0] if len(rej) else [],
                    y=log["theta"][rej, 1] if len(rej) else [],
                    hovertext=[hover(i) for i in rej],
                    marker=dict(color=np.log10(np.clip(log["d"][rej], 0.05, 20))
                                if len(rej) else []))
                if c == len(self._ptr):
                    d[idx["rej"]].marker.colorbar.title.text = (
                        f"distance d(X, x_obs)   (ε = {s.eps:g})")
                d[idx["acc"]].update(
                    x=log["theta"][acc, 0] if len(acc) else [],
                    y=log["theta"][acc, 1] if len(acc) else [],
                    hovertext=[hover(i) for i in acc],
                    marker=dict(size=acc_size,
                                color=np.log10(np.clip(log["d"][acc], 0.05, 20))
                                if len(acc) else []))
                if sel is not None and np.isfinite(log["d"][sel]):
                    t0, anc = log["theta"][sel], log["ancestor"][sel]
                    if anc >= 0 and s.prev is not None:
                        a0 = s.prev[0][anc]
                        d[idx["sel"]].update(x=[a0[0], t0[0]], y=[a0[1], t0[1]],
                                             marker=dict(size=[12, 18]))
                    else:
                        d[idx["sel"]].update(x=[t0[0]], y=[t0[1]], marker=dict(size=[18]))
                else:
                    d[idx["sel"]].update(x=[], y=[])

class PriorDraws:
    """Draw theta from the prior and simulate X, rejecting nothing."""

    def __init__(self, example=1, seed=0, show=True):
        self.ex = example
        self.model = MODELS[example]()
        self.seed = seed

        def button(label, cb, width):
            b = W.Button(description=label, layout=W.Layout(width=width, height="38px"))
            b.on_click(cb)
            return b

        self.batch = W.BoundedIntText(value=100, min=1, max=100_000,
                                      layout=W.Layout(width="100px"))
        self.image = W.Image(format="png", layout=W.Layout(width="100%"))
        self.text = W.HTML()
        css = W.HTML("<style>.abcsmc, .abcsmc .widget-label, .abcsmc button, "
                     ".abcsmc input {font-size: 15px !important;}</style>")
        row = W.HBox([button("Clear", self.clear, "90px"),
                      button("+1 draw", lambda _: self.add(1), "110px"),
                      button("+ batch of", lambda _: self.add(self.batch.value), "110px"),
                      self.batch])
        self.ui = W.VBox([css, row, self.image, self.text])
        self.ui.add_class("abcsmc")
        self.clear()
        if show:
            display(self.ui)

    def clear(self, _=None):
        self.rng = np.random.default_rng(self.seed)
        self.theta = np.empty((0, self.model.dim))
        self.x = np.empty((0, self.model.dim))
        self.render()

    def add(self, n):
        m = self.model
        th = self.rng.uniform(m.low, m.high, size=(n, m.dim))
        x = m.simulate(th, self.rng.standard_normal(th.shape))
        self.theta = np.vstack([self.theta, th])
        self.x = np.vstack([self.x, x])
        self.render()

    def render(self):
        import matplotlib
        m, n = self.model, len(self.theta)
        with matplotlib.rc_context(RC):
            if self.ex == 1:
                fig = Figure(figsize=(13, 6.6), layout="constrained")
                ax = fig.subplots()
                g = m.grid(600)
                ax.plot(g, m.f(g), color=DARK, lw=2, label="f(θ) = θ³ − 3θ (noise-free)")
                ax.axhline(m.x_obs[0], color=ORANGE, lw=2, label="$x_{obs}$ = 0")
                if n:
                    ax.scatter(self.theta[:, 0], self.x[:, 0], s=16, color=BLUE,
                               alpha=0.6, lw=0, label="draws (θ, X)")
                    t0, x0 = self.theta[-1, 0], self.x[-1, 0]
                    ax.plot([t0, t0], [m.f(t0), x0], color=GREEN, lw=1.5)
                    ax.scatter([t0], [x0], s=220, facecolor="none", edgecolor=GREEN,
                               lw=2.5, label="most recent draw")
                ax.set_ylim(-9, 9)
                ax.set_xlim(m.low, m.high)
                ax.set_xlabel("θ")
                ax.set_ylabel("simulated X")
                ax.set_title(f"Draws from the prior and the simulator (noise σ = {m.sigma:g}); "
                             "nothing rejected")
                ax.legend(loc="upper left", fontsize=11, frameon=False)
            else:
                fig = Figure(figsize=(14, 7), layout="constrained")
                a1, a2 = fig.subplots(1, 2)
                if n:
                    a1.scatter(self.theta[:, 0], self.theta[:, 1], s=12, color=BLUE,
                               alpha=0.6, lw=0)
                    a2.scatter(self.x[:, 0], self.x[:, 1], s=12, color=BLUE, alpha=0.6,
                               lw=0, label="simulated X")
                    for a, pt in ((a1, self.theta[-1]), (a2, self.x[-1])):
                        a.scatter([pt[0]], [pt[1]], s=220, facecolor="none",
                                  edgecolor=GREEN, lw=2.5)
                a1.axhline(0, color=LIGHT, lw=0.8)
                a1.axvline(0, color=LIGHT, lw=0.8)
                a1.set_xlim(m.low, m.high)
                a1.set_ylim(m.low, m.high)
                a1.set_aspect("equal")
                a1.set_xlabel("θ₁")
                a1.set_ylabel("θ₂")
                a1.set_title("θ drawn from the prior")
                a2.scatter([m.x_obs[0]], [m.x_obs[1]], marker="+", s=300, lw=3,
                           color=ORANGE, zorder=5, label="$x_{obs}$ = (4, 4)")
                a2.set_xlim(-1.5, 17.5)
                a2.set_ylim(-1.5, 17.5)
                a2.set_aspect("equal")
                a2.set_xlabel("X₁")
                a2.set_ylabel("X₂")
                a2.set_title("Simulated data X; nothing rejected")
                a2.legend(loc="upper right", fontsize=11, frameon=False)
            self.image.value = _png(fig)
        if n:
            fmt = lambda v: ", ".join(f"{x:.3f}" for x in v)
            latest = (f" Latest: θ = ({fmt(self.theta[-1])}), X = ({fmt(self.x[-1])}), "
                      f"distance to x<sub>obs</sub> = "
                      f"{m.distance(self.x[-1:])[0]:.3f}.")
        else:
            latest = " Press <b>+1 draw</b> to start."
        self.text.value = (f"<div style='font-size:17px'>{n:,} draws.{latest}</div>")


def sub(t):
    return str(t).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def n_props(log):
    return 0 if log is None else len(log["d"])
