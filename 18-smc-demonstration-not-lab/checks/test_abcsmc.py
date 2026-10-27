"""Checks of the sampler's mathematics. Run with:  python -m pytest checks/"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abcsmc import ABCSMC, CubicModel, SquaresModel  # noqa: E402


def advance_to_generation(s, t):
    while s.t < t:
        s.run_generation()
        s.start_next_generation()


@pytest.mark.parametrize("Model,tau", [(CubicModel, 0.2), (CubicModel, 0.6),
                                       (SquaresModel, 0.4)])
def test_weights_are_prior_over_mixture_proposal(Model, tau):
    m = Model()
    s = ABCSMC(m, N=150, kernel_sd=tau, eps0=3.0, seed=3)
    advance_to_generation(s, 2)
    s.run_generation()
    h = s.history[-1]
    th_prev, w_prev = h["prev"]
    vol = (m.high - m.low) ** m.dim
    # brute-force q_t(theta)
    kern = np.prod(norm.pdf(h["theta"][:, None, :], th_prev[None, :, :], tau), axis=2)
    q = kern @ w_prev
    w_expected = (1 / vol) / q
    w_expected /= w_expected.sum()
    np.testing.assert_allclose(h["w"], w_expected, rtol=1e-10)


def test_generation_zero_has_equal_weights():
    s = ABCSMC(CubicModel(), N=100, seed=1)
    s.run_generation()
    np.testing.assert_allclose(s.history[0]["w"], 1 / 100)


@pytest.mark.parametrize("Model", [CubicModel, SquaresModel])
def test_weights_normalized_and_particles_inside_prior(Model):
    m = Model()
    s = ABCSMC(m, N=200, kernel_sd=1.5, n_generations=4, eps0=3.0,
               seed=5).run_all()
    for h in s.history:
        assert np.isclose(h["w"].sum(), 1.0)
        assert np.all(h["w"] >= 0)
        assert np.all((h["theta"] >= m.low) & (h["theta"] <= m.high))
        assert np.all(h["distances"] < h["eps"])


def test_out_of_prior_proposals_are_not_simulated():
    s = ABCSMC(CubicModel(), N=100, kernel_sd=2.0, n_generations=3, seed=2)
    advance_to_generation(s, 1)
    s.run_generation()
    log = s.current_log()
    outside = log["source"] == "outside prior"
    assert outside.any()
    assert np.all(np.isnan(log["x"][outside])) and not log["accepted"][outside].any()
    assert s.n_sims == (~outside).sum()


@pytest.mark.parametrize("Model", [CubicModel, SquaresModel])
def test_proposal_density_integrates_to_one(Model):
    m = Model()
    s = ABCSMC(m, N=100, kernel_sd=0.3, eps0=3.0, seed=4)
    advance_to_generation(s, 1)
    g = np.linspace(-7, 7, 1401)
    q = s.proposal_density_grid(g)
    dg = g[1] - g[0]
    assert np.isclose(q.sum() * dg ** m.dim, 1.0, atol=5e-3)
    # agrees with the density used for the weights
    pts = np.array([[0.1] * m.dim, [1.7] * m.dim])
    i = np.searchsorted(g, pts[:, 0])
    vals = q[i] if m.dim == 1 else q[i, i]
    np.testing.assert_allclose(np.log(vals), s.log_q(np.c_[g[i]][:, [0] * m.dim]),
                               rtol=1e-8)


@pytest.mark.parametrize("Model", [CubicModel, SquaresModel])
def test_same_seed_same_result(Model):
    kw = dict(N=100, kernel_sd=0.3, n_generations=4, eps0=3.0, seed=11)
    a = ABCSMC(Model(), **kw).run_all()
    b = ABCSMC(Model(), **kw).run_all()
    for ha, hb in zip(a.history, b.history):
        np.testing.assert_array_equal(ha["theta"], hb["theta"])
        np.testing.assert_array_equal(ha["w"], hb["w"])
    assert a.total_sims == b.total_sims


@pytest.mark.parametrize("Model", [CubicModel, SquaresModel])
def test_stepping_granularity_does_not_change_results(Model):
    kw = dict(N=60, kernel_sd=0.3, n_generations=3, eps0=3.0, seed=7)
    a = ABCSMC(Model(), **kw).run_all()
    b = ABCSMC(Model(), **kw)
    sizes = [1, 1, 7, 25, 1, 100]
    i = 0
    while not b.finished:
        if b.generation_complete:
            b.start_next_generation()
        b.step(sizes[i % len(sizes)])
        i += 1
    for ha, hb in zip(a.history, b.history):
        np.testing.assert_array_equal(ha["theta"], hb["theta"])
        np.testing.assert_allclose(ha["w"], hb["w"], rtol=1e-12)
        assert ha["n_sims"] == hb["n_sims"]


def test_cubic_reference_matches_rejection_abc():
    m = CubicModel()
    rng = np.random.default_rng(0)
    for eps in (1.0, 0.1):
        th = rng.uniform(m.low, m.high, size=(2_000_000, 1))
        x = m.simulate(th, rng.standard_normal(th.shape))
        keep = m.distance(x) < eps
        mc = np.bincount(m.cell(th[keep]), minlength=3) / keep.sum()
        np.testing.assert_allclose(mc, m.reference_masses(eps), atol=0.01)
        # the density itself: compare a histogram of accepted draws
        g = m.grid(2001)
        ref = m.reference_density(eps, g)
        bins = np.linspace(m.low, m.high, 51)
        h, _ = np.histogram(th[keep, 0], bins=bins, density=True)
        ref_bins = [ref[(g >= a) & (g < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
        assert np.max(np.abs(h - ref_bins)) < 0.08 * max(ref_bins)


def test_squares_acceptance_probability_matches_simulation():
    m = SquaresModel()
    rng = np.random.default_rng(1)
    for theta in ([2.0, 2.0], [1.9, -2.1], [0.0, 2.0], [2.2, 1.6]):
        for eps in (0.25, 1.0):
            th = np.tile(theta, (400_000, 1))
            x = m.simulate(th, rng.standard_normal(th.shape))
            mc = np.mean(m.distance(x) < eps)
            exact = m.acceptance_prob(np.array([theta]), eps)[0]
            assert abs(mc - exact) < 4 * np.sqrt(exact * (1 - exact) / 400_000) + 1e-4


def test_squares_reference_masses_are_quarters():
    np.testing.assert_allclose(SquaresModel().reference_masses(0.5), 0.25, atol=1e-6)


@pytest.mark.parametrize("Model,eps_min", [(CubicModel, 0.1), (SquaresModel, 0.25)])
def test_importance_weighted_masses_approach_reference(Model, eps_min):
    m = Model()
    s = ABCSMC(m, N=1000, kernel_sd=0.3, eps0=3.0, eps_min=eps_min,
               n_generations=6, seed=21).run_all()
    h = s.history[-1]
    np.testing.assert_allclose(h["masses"], m.reference_masses(h["eps"]), atol=0.05)


def test_sampler_never_uses_the_reference():
    m = CubicModel()

    def forbidden(*a, **k):
        raise AssertionError("the sampler must not use the reference posterior")

    m.reference_density = m.reference_masses = m.acceptance_prob = forbidden
    ABCSMC(m, N=50, n_generations=3, seed=1).run_all()


def test_budget_stops_the_run():
    s = ABCSMC(SquaresModel(), N=200, kernel_sd=3.0, eps0=3.0, eps_min=0.05,
               schedule="fast", max_sims_per_generation=5_000, seed=1).run_all()
    assert s.stopped_reason == "budget"
    assert s.history[-1]["n_sims"] == 5_000
    assert s.history[-1]["n_accepted"] < 200



def test_ending_a_generation_early():
    m = CubicModel()
    s = ABCSMC(m, N=200, kernel_sd=0.3, n_generations=3, seed=8)
    assert not s.end_generation()          # nothing accepted yet
    s.step(40)
    k = s.n_acc
    assert 0 < k < 200 and s.end_generation()
    assert s.history[0]["n_accepted"] == k and s.history[0]["ended_early"]
    s.start_next_generation()
    assert len(s.prev[1]) == k
    s.run_generation()
    h = s.history[-1]
    th_prev, w_prev = h["prev"]
    q = norm.pdf(h["theta"], th_prev[:, 0], 0.3) @ w_prev
    w = (1 / 5) / q
    np.testing.assert_allclose(h["w"], w / w.sum(), rtol=1e-10)


@pytest.mark.parametrize("x_obs,n_modes", [(0.0, 3), (1.0, 3), (2.0, 2), (3.0, 1), (-5.0, 1)])
def test_cubic_modes_follow_x_obs(x_obs, n_modes):
    m = CubicModel(x_obs=x_obs)
    r = m.mode_locations[:, 0]
    assert len(r) == n_modes
    np.testing.assert_allclose(m.f(r), x_obs, atol=1e-4)
    masses = m.reference_masses(0.1)
    assert len(masses) == n_modes and np.isclose(masses.sum(), 1)


def test_squares_modes_follow_x_obs():
    m = SquaresModel(x_obs=(9.0, 1.0))
    np.testing.assert_allclose(np.abs(m.mode_locations), [[3, 1]] * 4)
    np.testing.assert_allclose(m.reference_masses(0.5), 0.25, atol=1e-6)
