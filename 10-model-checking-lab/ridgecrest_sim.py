# Simulator for the Ridgecrest model-checking lab: aftershocks in one band
# around one gently curved fault, plus a small amount of broadly scattered
# activity. The model and fitted values are described in 10-model-description.html.
#
# One simulated catalog, in this order:
#   1. N ~ Poisson(Lambda), Lambda = integral of K (t + c)^(-p) over [t_start, t_end)
#   2. times: N iid draws from the density proportional to (t + c)^(-p) on
#      [t_start, t_end), by inverting its CDF; then sorted
#   3. magnitudes: M = m_min + Exponential(rate = b * log(10)), iid
#   4. scatter indicator S ~ Bernoulli(eps), iid
#   5. along-fault position U:
#        S = 0 (band):    U ~ g, the fitted smoothed histogram (pick a 1-km bin
#                         with its probability, then a uniform point inside it)
#        S = 1 (scatter): U ~ Uniform(u_lo, u_hi)
#   6. across-fault position V:
#        S = 0 (band):    V ~ Normal(m(U), sigma_v^2): centred on the fault
#                         curve m, with the same spread wherever U is
#        S = 1 (scatter): V ~ Uniform(-v_scatter, v_scatter)
#   7. apply the same observation window as the observed catalog
# Events are independent; for each event, time, magnitude and location are
# independent.

import numpy as np
import pandas as pd

fit = dict(
    t_start=0.25, t_end=30,      # days after the M7.1 mainshock (fixed design)
    m_min=2.5,                   # magnitude threshold (fixed design)
    K=268.7, c=0.01, p=1.160,    # Omori-Utsu rate, events/day; c is fixed, not fitted
    b=1.069,                     # Gutenberg-Richter b-value
    g=pd.read_csv("data/along_fault_density.csv"),  # 1-km bins: u_left, u_right, u_mid,
                                 # prob (g) and centre_km (the fault curve m at u_mid)
    sigma_v=1.86,                # width of the band around the fault curve, km
    eps=0.069,                   # share of broadly scattered events
    u_lo=-33.66, u_hi=23.40,     # along-fault extent of the scatter, km
    v_scatter=12,                # scatter fills |v| <= 12 km
)

# Fixed observation window in (U, V) coordinates (km), shared with the data
window = dict(u=(-42, 26), v=(-12, 12))


def fault_centre(u):
    """The fault curve m(u): linear interpolation between 1-km bin centres."""
    return np.interp(u, fit["g"]["u_mid"], fit["g"]["centre_km"])


def offset_from_fault(cat):
    """Signed distance across the fault from the curve (NE positive)."""
    return cat["v_km"] - fault_centre(cat["u_km"])


def apply_window(cat):
    keep = ((cat["t_days"] >= fit["t_start"]) & (cat["t_days"] < fit["t_end"])
            & (cat["mag"] >= fit["m_min"])
            & (cat["u_km"] >= window["u"][0]) & (cat["u_km"] <= window["u"][1])
            & (cat["v_km"] >= window["v"][0]) & (cat["v_km"] <= window["v"][1]))
    return cat[keep].reset_index(drop=True)


def simulate_catalog(fit, rng):
    # --- 1-2. times: nonhomogeneous Poisson process with Omori-Utsu rate
    A = (fit["t_start"] + fit["c"]) ** (1 - fit["p"])
    B = (fit["t_end"] + fit["c"]) ** (1 - fit["p"])
    expected_n = fit["K"] * (B - A) / (1 - fit["p"])
    n = rng.poisson(expected_n)
    t_days = np.sort((A + rng.uniform(size=n) * (B - A)) ** (1 / (1 - fit["p"])) - fit["c"])

    # --- 3. magnitudes: Gutenberg-Richter above the threshold
    mag = fit["m_min"] + rng.exponential(scale=1 / (fit["b"] * np.log(10)), size=n)

    # --- 4. which events are broadly scattered
    scatter = rng.uniform(size=n) < fit["eps"]

    # --- 5. along-fault position
    # (np.where draws a candidate from both branches and keeps the matching one)
    g = fit["g"]
    bin_ = rng.choice(len(g), size=n, p=g["prob"] / g["prob"].sum())
    left, right = g["u_left"].to_numpy()[bin_], g["u_right"].to_numpy()[bin_]
    u_km = np.where(scatter, rng.uniform(fit["u_lo"], fit["u_hi"], size=n),
                             left + rng.uniform(size=n) * (right - left))

    # --- 6. across-fault position
    v_km = np.where(scatter, rng.uniform(-fit["v_scatter"], fit["v_scatter"], size=n),
                             fault_centre(u_km) + rng.normal(0, fit["sigma_v"], size=n))

    # --- 7. same filters as the observed catalog
    return apply_window(pd.DataFrame(dict(t_days=t_days, mag=mag, u_km=u_km, v_km=v_km)))
