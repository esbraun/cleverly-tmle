"""An independent transcription of the pricing law and its truths. It imports nothing of cleverly.

The tutorial page ``docs/examples/sequential-pricing.ipynb`` takes its truths from
``cleverly.datasets.pricing_truth``. This script re-derives them from the law table in the
docstring of ``make_sequential_pricing``, with its own Gauss--Legendre rule: ten nodes for each
piece of a price band, thirty for usage, and a different split of the bands. ``summarize.py``
compares its output with the package, and the page cites that comparison.

Usage: ``python truth.py`` prints every truth.
"""

import itertools

import numpy as np
from scipy.stats import norm

SD, BOUND = 1.5, 2.5


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def centre(e):
    return 6.0 + 2.0 * e


def lo(e):
    return 4.0 + 2.0 * e


def hi(e):
    return 8.0 + 2.0 * e


def gauss(order, a, b):
    x, w = np.polynomial.legendre.leggauss(order)
    half = (b - a) / 2.0
    return (a + b) / 2.0 + half * x, half * w


def trunc_pdf(price, mean, e):
    mass = norm.cdf((hi(e) - mean) / SD) - norm.cdf((lo(e) - mean) / SD)
    return norm.pdf((price - mean) / SD) / (SD * mass)


def plan(kind, factor):
    if kind == "up":
        return lambda a, e: np.where(a < hi(e) / factor, factor * a, a)
    if kind == "down":
        return lambda a, e: np.where(a < lo(e) / factor, a, factor * a)
    return lambda a, e: a


def edges(kind, factor, e):
    cuts = {lo(e), hi(e), hi(e) / 1.12, lo(e) / 0.85}
    if kind == "up":
        cuts.add(hi(e) / factor)
    if kind == "down":
        cuts.add(lo(e) / factor)
    return sorted(cuts)


def mean_outcomes(kind, factor, hold_engagement=False, hold_engine=False, orders=(10, 30)):
    """``(E[renewed], E[revenue])`` by explicit nested sums over three prices."""
    d = plan(kind, factor)
    nodes_w, weights_w = gauss(orders[1], -BOUND, BOUND)
    weights_w = weights_w * norm.pdf(nodes_w) / (norm.cdf(BOUND) - norm.cdf(-BOUND))
    ey = er = 0.0
    for path in itertools.product((0.0, 1.0), repeat=3):

        def pieces(e):
            xs, ws = [], []
            for a, b in itertools.pairwise(edges(kind, factor, e)):
                x, w = gauss(orders[0], a, b)
                xs.append(x)
                ws.append(w)
            return np.concatenate(xs), np.concatenate(ws)

        (p1, w1), (p2, w2), (p3, w3) = (pieces(e) for e in path)
        wu = nodes_w[:, None, None, None]
        # Month 1: the start is A0 = c0 = 6 and L0 = 0, so the gaps are zero.
        chance1 = expit(0.3 + 0.6 * nodes_w)
        m1 = weights_w * (chance1 if path[0] else 1 - chance1)
        a1 = p1[None, :, None, None]
        dens1 = trunc_pdf(a1, centre(path[0]) + 0.3 * 0.0 + 0.3 * wu, path[0])
        gap1_assigned = d(a1, path[0]) - centre(path[0])
        gap1_natural = a1 - centre(path[0])
        # Month 2.
        read_l = gap1_natural if hold_engagement else gap1_assigned
        chance2 = expit(0.3 + 0.6 * wu - 0.35 * read_l + 0.8 * path[0])
        read_a = gap1_natural if hold_engine else gap1_assigned
        a2 = p2[None, None, :, None]
        dens2 = trunc_pdf(a2, centre(path[1]) + 0.3 * read_a + 0.3 * wu, path[1])
        gap2_assigned = d(a2, path[1]) - centre(path[1])
        gap2_natural = a2 - centre(path[1])
        # Month 3.
        read_l = gap2_natural if hold_engagement else gap2_assigned
        chance3 = expit(0.3 + 0.6 * wu - 0.35 * read_l + 0.8 * path[1])
        read_a = gap2_natural if hold_engine else gap2_assigned
        a3 = p3[None, None, None, :]
        dens3 = trunc_pdf(a3, centre(path[2]) + 0.3 * read_a + 0.3 * wu, path[2])
        assigned3 = d(a3, path[2])
        gap3_assigned = assigned3 - centre(path[2])
        renew = expit(
            0.5
            - 0.45 * gap3_assigned
            + 0.9 * path[2]
            + 0.3 * wu
            - 0.1 * (gap1_assigned + gap2_assigned)
        )
        mass = (
            m1[:, None, None, None]
            * (chance2 if path[1] else 1 - chance2)
            * (chance3 if path[2] else 1 - chance3)
            * dens1
            * dens2
            * dens3
            * w1[None, :, None, None]
            * w2[None, None, :, None]
            * w3[None, None, None, :]
        )
        ey += float(np.sum(mass * renew))
        er += float(np.sum(mass * renew * assigned3))
    return ey, er


def truths():
    """Every key of ``pricing_truth()``, from this independent transcription."""
    reference = "natural_prices"
    specs = {
        reference: ("natural", 1.0),
        "price_up": ("up", 1.12),
        "price_down": ("down", 0.85),
        "up_6": ("up", 1.06),
        "up_25": ("up", 1.25),
    }
    free = {label: mean_outcomes(*spec) for label, spec in specs.items()}
    out = {}
    for label, (renew, revenue) in free.items():
        out[f"ey_regimen[{label}]"], out[f"ey_revenue[{label}]"] = renew, revenue
        if label != reference:
            out[f"ate_regimen[{label} vs {reference}]"] = renew - free[reference][0]
            out[f"ate_revenue[{label} vs {reference}]"] = revenue - free[reference][1]
    for name, (hold_engagement, hold_engine) in {
        "ate_engagement_held": (True, False),
        "ate_engine_held": (False, True),
        "ate_both_held": (True, True),
    }.items():
        for label in ("price_up", "price_down"):
            out[f"{name}[{label}]"] = (
                mean_outcomes(*specs[label], hold_engagement, hold_engine)[0] - free[reference][0]
            )
    return out


if __name__ == "__main__":
    for key, value in truths().items():
        print(f"{key:44s} {value:.8f}")
