"""Check the numbers the notebook's walkthrough quotes for each scenario.

    python checks/check_presets.py

Each scenario starts from the app's defaults, halves ε each generation (the
app's pre-filled value) until it reaches 0.1 (Example 1) or 0.25 (Example 2),
then stays there, and stops each generation at N accepted particles. It is run with 50
seeds (1-50). The script prints a summary and exits with an error if a claim
fails.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abcsmc import ABCSMC, MODELS, run_replicates  # noqa: E402
from app import DEFAULTS as APP_DEFAULTS  # noqa: E402

# the automatic stand-in for clicking through a run in class
DEFAULTS = {ex: dict(d, schedule="medium", n_generations=6)
            for ex, d in APP_DEFAULTS.items()}

# Each scenario: (changes from DEFAULTS, what the walkthrough says it shows).
SCENARIOS = {
    1: {
        "Three-mode recovery": ({}, (
            "Start from the prior at ε = 2 and halve ε each generation down to 0.1. "
            "All three solutions keep particles, and the weighted masses approach the "
            "reference (about 0.25, 0.5, 0.25).")),
        "Small-sample mode loss": ({"N": 20}, (
            "Same settings, but keep generations small (about 20 accepted particles: "
            "one batch of 25 at first, larger batches as acceptance falls). A mode can "
            "lose its last particle, and the local kernels almost never bring it back. "
            "Run replicates (N = 20): roughly 40% of runs miss a mode.")),
        "Oversized kernel, poor acceptance": ({"kernel_sd": 2.0}, (
            "τ = 2 spreads proposals over the whole prior range, so the sequence no "
            "longer focuses the search. Estimates stay fine, but the last generations "
            "accept about 3% instead of about 10%; in replicates the run needs roughly "
            "three times as many simulations.")),
    },
    2: {
        "Four-mode recovery": ({}, (
            "Start from the prior at ε = 3 and halve ε each generation down to 0.25. "
            "All four quadrants keep particles, with masses near 1/4 each.")),
        "Small-sample mode loss": ({"N": 20}, (
            "Same settings, but keep generations small (about 20 accepted particles). "
            "Quadrants lose their last particle and are not recovered. Run replicates "
            "(N = 20): more than half the runs end with a mode missing.")),
        "Oversized kernel, poor acceptance": ({"kernel_sd": 1.0}, (
            "τ = 1 proposes far from the existing particles. The answer is still "
            "right, but the last generations accept about 0.2% of proposals instead "
            "of about 4%; in replicates the run costs more than ten times as many "
            "simulations.")),
        "Aggressive tolerance reduction": ({"schedule": "fast"}, (
            "ε drops from 3 straight to 0.3. The population from ε = 3 is a poor proposal "
            "for the much smaller target: the first SMC generation accepts about 1% "
            "instead of about 20%, and reaching the final ε costs more in total.")),
    },
}



def preset_settings(example, name):
    """Full settings dict for a scenario."""
    return {**DEFAULTS[example], **SCENARIOS[example][name][0]}

R = 50


def runs(example, name):
    st = dict(preset_settings(example, name))
    model = MODELS[example]()
    st.pop("seed")
    return model, run_replicates(model, n_reps=R, first_seed=1, **st)


def single(example, name, seed=1):
    st = dict(preset_settings(example, name))
    model = MODELS[example]()
    st["seed"] = seed
    return ABCSMC(model, **st).run_all()


def per_generation(example, name, key):
    """Median over seeds of history[t][key], for each generation t."""
    vals = []
    for seed in range(1, 11):
        vals.append([h[key] for h in single(example, name, seed).history])
    return np.median(np.array(vals), axis=0)


def summary(model, rows):
    k = len(model.mode_labels)
    found = np.array([r["found"] for r in rows])
    M = np.array([r["masses"] for r in rows])
    sims = np.array([r["total_sims"] for r in rows])
    return dict(all_found=np.mean(found == k), mean_mass=M.mean(0),
                median_sims=np.median(sims), budget=sum(r["budget_hit"] for r in rows),
                found=found, masses=M)


failures = []


def check(label, ok, detail):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- Example 1
print("Example 1")
m, rows = runs(1, "Three-mode recovery")
base1 = summary(m, rows)
ref = m.reference_masses(DEFAULTS[1]["eps_min"])
check("three-mode: all modes found", base1["all_found"] == 1,
      f"{base1['all_found']:.0%} of runs")
check("three-mode: masses near reference",
      np.all(np.abs(base1["mean_mass"] - ref) < 0.03),
      f"mean {base1['mean_mass'].round(3)} vs {ref.round(3)}")

m, rows = runs(1, "Small-sample mode loss")
s = summary(m, rows)
check("N = 20: roughly 40% of runs lose a mode", 0.25 <= 1 - s["all_found"] <= 0.6,
      f"{1 - s['all_found']:.0%} of runs miss at least one mode")

m, rows = runs(1, "Oversized kernel, poor acceptance")
s = summary(m, rows)
acc_big = per_generation(1, "Oversized kernel, poor acceptance", "acc_rate")
acc_base = per_generation(1, "Three-mode recovery", "acc_rate")
check("oversized: final acceptance far lower",
      acc_big[-1] < 0.5 * acc_base[-1] and 0.015 < acc_big[-1] < 0.045,
      f"{acc_big[-1]:.1%} vs {acc_base[-1]:.1%}")
check("oversized: about three times the simulations",
      2.3 < s["median_sims"] / base1["median_sims"] < 4,
      f"{s['median_sims']:.0f} vs {base1['median_sims']:.0f}")
check("oversized: estimates still fine", np.all(np.abs(s["mean_mass"] - ref) < 0.03),
      f"mean {s['mean_mass'].round(3)}")

# ---------------------------------------------------------------- Example 2
print("Example 2")
m, rows = runs(2, "Four-mode recovery")
base2 = summary(m, rows)
check("four-mode: all modes found", base2["all_found"] == 1, f"{base2['all_found']:.0%}")
check("four-mode: masses near 1/4", np.all(np.abs(base2["mean_mass"] - 0.25) < 0.03),
      f"mean {base2['mean_mass'].round(3)}")

m, rows = runs(2, "Small-sample mode loss")
s = summary(m, rows)
check("N = 20: more than half the runs lose a mode", 0.45 <= 1 - s["all_found"] <= 0.85,
      f"{1 - s['all_found']:.0%}")

m, rows = runs(2, "Oversized kernel, poor acceptance")
s = summary(m, rows)
acc_big = per_generation(2, "Oversized kernel, poor acceptance", "acc_rate")
acc_base = per_generation(2, "Four-mode recovery", "acc_rate")
check("oversized: final acceptance about 0.2% (vs about 4%)",
      acc_big[-1] < 0.004 and acc_base[-1] > 0.03,
      f"{acc_big[-1]:.2%} vs {acc_base[-1]:.1%}")
check("oversized: more than ten times the simulations",
      s["median_sims"] > 10 * base2["median_sims"],
      f"{s['median_sims']:.0f} vs {base2['median_sims']:.0f}")
check("oversized: no run hits the budget", s["budget"] == 0, f"{s['budget']} runs")
check("oversized: estimates still fine", np.all(np.abs(s["mean_mass"] - 0.25) < 0.03),
      f"mean {s['mean_mass'].round(3)}")

m, rows = runs(2, "Aggressive tolerance reduction")
s = summary(m, rows)
acc_fast = per_generation(2, "Aggressive tolerance reduction", "acc_rate")
check("aggressive: first SMC generation accepts about 1% (vs about 20%)",
      acc_fast[1] < 0.02 and acc_base[1] > 0.15,
      f"{acc_fast[1]:.1%} vs {acc_base[1]:.1%}")
cost_fast, cost_base = [], []
for seed in range(1, 11):
    for name, out in (("Aggressive tolerance reduction", cost_fast),
                      ("Four-mode recovery", cost_base)):
        h = single(2, name, seed).history
        out.append(next(x["total_sims"] for x in h if x["eps"] <= DEFAULTS[2]["eps_min"]))
check("aggressive: reaching the final ε costs more",
      np.median(cost_fast) > 1.2 * np.median(cost_base),
      f"median {np.median(cost_fast):.0f} vs {np.median(cost_base):.0f} simulations")

print()
if failures:
    print("FAILED:", *failures, sep="\n  ")
    sys.exit(1)
print("All preset claims hold.")
