"""Recompute the printed truths of interventions.ipynb from the structural equations.

The equations are typed here from ``nonlinear_dgp``, ``nonlinear_bounded_dgp``, and
``shift_dgp`` in ``src/cleverly/datasets/synthetic.py``.  This script does not import
``cleverly``, so it is an independent check of the package truths that the page prints.

Binary law (the regime and incremental axes, ``navigation_data``): Monte Carlo over 2 x 10^7
standard-normal draws of W1 to W4, in 40 chunks.

- the ATE, ``E[Q1 - Q0]``
- the screen contrast, ``E[(Q1 - Q0) 1(W1 > 0)]``
- the double-odds incremental contrast, ``E[(2 g Q1 + (1 - g) Q0) / (2 g + 1 - g)] - E[g Q1 +
  (1 - g) Q0]``.  It depends on the propensity g, so it moved when g was bounded to
  [0.05, 0.95].

Dose law (the modified-treatment-policy axis, ``make_shift_dose``): Monte Carlo over 2 x 10^7
draws of (W1, W2, W3, A), in 40 chunks, with ``A | W ~ N(2 + 0.7 W1 - 0.3 W2, 1)``.

- each shift contrast, with a capped unit held at its own dose
- the closed form ``1.5 delta + 0.25 delta^2`` of an uncapped shift
- ``E[Y]``, ``P(A < 0)``, ``P(A + 0.5 > 5)``, and the Kish fraction ``1 / E[r^2]`` of the true
  density ratio ``r = g(A - delta | W) / g(A | W)`` for each uncapped shift

Usage: ``python truth.py``.  Output: ``truth.log``.
"""

import numpy as np
from scipy.special import expit

CHUNKS, SIZE = 40, 500_000


def propensity(w):
    u = 0.6 * w[:, 0] - 0.4 * w[:, 1] ** 2 + 0.5 * w[:, 1] * w[:, 2] + 0.3 * (w[:, 3] > 0)
    return 0.05 + 0.90 * expit(u)


def bounded_mean(w, a):
    baseline = (
        -0.5
        + 0.8 * np.sin(1.5 * w[:, 0])
        + 0.5 * np.tanh(w[:, 1] ** 2 - 1.0)
        - 0.4 * np.tanh(w[:, 2] * w[:, 3])
        + 0.3 * np.tanh(np.abs(w[:, 3]))
    )
    effect = 0.9 + 0.4 * np.tanh(w[:, 0]) - 0.3 * (w[:, 1] > 0)
    return expit(baseline + effect * a)


def dose_mean(w):
    return 2.0 + 0.7 * w[:, 0] - 0.3 * w[:, 1]


def dose_outcome(w, a):
    return 1.0 + 0.5 * a + 0.25 * a**2 + w[:, 0] - 0.5 * w[:, 1] + 0.2 * w[:, 2]


def shifted(a, delta, cap):
    moved = a + delta
    return moved if cap is None else np.where(moved > cap, a, moved)


def normal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def report(name, values):
    values = np.asarray(values)
    se = values.std(ddof=1) / np.sqrt(len(values))
    print(f"  {name:44s} {values.mean():.6f}  (MC SE {se:.1e})")


rng = np.random.default_rng(20261002)
binary = {"ate": [], "screen": [], "ipsi x2 vs x1": [], "min g": [], "max g": []}
dose = {
    "+0.5 capped at 5": [],
    "+0.5 uncapped": [],
    "+1.0 uncapped": [],
    "gap uncapped - capped": [],
    "E[Y]": [],
    "P(A < 0)": [],
    "P(A + 0.5 > 5)": [],
    "Kish fraction, true ratio, +0.5": [],
    "Kish fraction, true ratio, +1.0": [],
}
for _ in range(CHUNKS):
    w = rng.standard_normal((SIZE, 4))
    g = propensity(w)
    q1, q0 = bounded_mean(w, 1.0), bounded_mean(w, 0.0)
    binary["ate"].append((q1 - q0).mean())
    binary["screen"].append(((q1 - q0) * (w[:, 0] > 0)).mean())
    tilted = (2.0 * g * q1 + (1.0 - g) * q0) / (2.0 * g + 1.0 - g)
    binary["ipsi x2 vs x1"].append((tilted - (g * q1 + (1.0 - g) * q0)).mean())
    binary["min g"].append(g.min())
    binary["max g"].append(g.max())

    v = rng.standard_normal((SIZE, 3))
    mu = dose_mean(v)
    a = mu + rng.standard_normal(SIZE)
    base = dose_outcome(v, a)
    capped = dose_outcome(v, shifted(a, 0.5, 5.0)) - base
    uncapped = dose_outcome(v, shifted(a, 0.5, None)) - base
    dose["+0.5 capped at 5"].append(capped.mean())
    dose["+0.5 uncapped"].append(uncapped.mean())
    dose["+1.0 uncapped"].append((dose_outcome(v, shifted(a, 1.0, None)) - base).mean())
    dose["gap uncapped - capped"].append((uncapped - capped).mean())
    dose["E[Y]"].append(base.mean())
    dose["P(A < 0)"].append((a < 0).mean())
    dose["P(A + 0.5 > 5)"].append((a + 0.5 > 5.0).mean())
    for delta in (0.5, 1.0):
        ratio = normal_pdf(a - delta - mu) / normal_pdf(a - mu)
        dose[f"Kish fraction, true ratio, +{delta}"].append(ratio.mean() ** 2 / (ratio**2).mean())

print(f"binary law: Monte Carlo, {CHUNKS} chunks of {SIZE}")
for key, values in binary.items():
    if key == "min g":
        print(f"  {key:44s} {min(values):.6f}")
    elif key == "max g":
        print(f"  {key:44s} {max(values):.6f}")
    else:
        report(key, values)
print(f"dose law: Monte Carlo, {CHUNKS} chunks of {SIZE}")
for key, values in dose.items():
    report(key, values)
for delta in (0.5, 1.0):
    print(f"  closed form, uncapped +{delta}: {1.5 * delta + 0.25 * delta**2:.6f}")
    print(f"  closed form, Kish fraction exp(-delta^2), +{delta}: {np.exp(-(delta**2)):.6f}")
