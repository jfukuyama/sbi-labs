# Scaled coalescent and stepwise-mutation simulator for the ABC lab.
# Python version of population_growth_sim.R. The two use different random
# number generators, so the same seed does not give the same datasets, but the
# model and the summaries are the same.
#
# n sampled Y chromosomes share one ancestry tree across all loci. Time is
# measured backward in units of N_A generations. The population is e^gamma
# times its ancestral size today, shrinks exponentially (looking backward) at
# rate beta, and is at its ancestral size from u = gamma / beta onward.

import numpy as np
import pandas as pd


def scaled_hazard(u, beta, gamma):
    """Lambda(u): the integral of 1 / q(w) from 0 to u."""
    u = np.asarray(u, dtype=float)
    if beta == 0:
        assert gamma == 0
        return u
    tau = gamma / beta
    end_hazard = -np.expm1(-gamma) / beta
    growing = u <= tau
    ans = np.empty_like(u)
    ans[growing] = np.exp(-gamma) * np.expm1(beta * u[growing]) / beta
    ans[~growing] = end_hazard + u[~growing] - tau
    return ans


def inverse_scaled_hazard(h, beta, gamma):
    """Lambda^{-1}(h): converts constant-size time to scaled time."""
    if beta == 0:
        assert gamma == 0
        return h
    tau = gamma / beta
    end_hazard = -np.expm1(-gamma) / beta
    if h > end_hazard:
        return tau + h - end_hazard
    if h <= 0:
        return 0.0
    # log(1 + exp(z)) / beta, computed without overflow for large growth.
    z = np.log(beta) + np.log(h) + gamma
    return np.logaddexp(0.0, z) / beta


def simulate_data(theta, beta, gamma, n, loci=8, root=None, rng=None):
    """Simulate an n x loci integer matrix of repeat counts.

    root gives the root's repeat count at each locus (default: all zero).
    """
    rng = np.random.default_rng() if rng is None else rng
    assert theta >= 0 and beta >= 0 and gamma >= 0 and n >= 2
    root = np.zeros(loci, dtype=int) if root is None else np.asarray(root)
    assert root.shape == (loci,)
    nodes = 2 * n - 1
    time = np.zeros(nodes)
    child1 = np.zeros(nodes, dtype=int)
    child2 = np.zeros(nodes, dtype=int)
    active = list(range(n))
    hazard_clock = 0.0

    # Build the tree backward: constant-size waiting times on the Lambda
    # clock, a uniformly chosen pair, then rescale the time by Lambda^{-1}.
    for parent in range(n, nodes):
        k = len(active)
        hazard_clock += rng.exponential(1 / (k * (k - 1) / 2))
        time[parent] = inverse_scaled_hazard(hazard_clock, beta, gamma)
        i, j = rng.choice(k, size=2, replace=False)
        child1[parent], child2[parent] = active[i], active[j]
        active = [a for m, a in enumerate(active) if m not in (i, j)] + [parent]

    # Drop mutations down the tree, starting from the root's counts.
    states = np.zeros((nodes, loci), dtype=int)
    states[nodes - 1] = root
    for parent in range(nodes - 1, n - 1, -1):
        for child in (child1[parent], child2[parent]):
            half_mean = theta * (time[parent] - time[child]) / 4
            states[child] = (states[parent] + rng.poisson(half_mean, loci)
                             - rng.poisson(half_mean, loci))
    return states[:n]


def summary_stats(x):
    """Mean variance, mean diversity, and number of distinct haplotypes.

    Matches summary_stats() in population_growth_sim.R: variances use the
    n - 1 denominator, and diversity uses uncorrected allele frequencies.
    """
    x = np.asarray(x)
    diversity = []
    for j in range(x.shape[1]):
        _, counts = np.unique(x[:, j], return_counts=True)
        p = counts / x.shape[0]
        diversity.append(1 - np.sum(p ** 2))
    return pd.Series({
        "mean_variance": np.mean(np.var(x, axis=0, ddof=1)),
        "mean_diversity": np.mean(diversity),
        "distinct_profiles": len(np.unique(x, axis=0)),
    })
