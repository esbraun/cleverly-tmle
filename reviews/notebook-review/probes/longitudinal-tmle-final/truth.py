"""Recompute the printed truths of docs/examples/longitudinal-tmle.ipynb without cleverly.

The structural equations are read from ``make_longitudinal`` and written out here:
``W1, W2, U ~ N(0, 1)`` independent, ``L2 = 0.6 W1 + 0.9 A1 + U``, and
``logit P(Y = 1) = -0.4 + 0.5 A1 + 0.8 A2 + 0.4 L2 + 0.3 W1 - 0.2 W2 + 0.5 tanh(L2)``.
The cluster construction keeps the marginal of ``U`` standard normal, so the truth holds on a
clustered draw. Censoring does not enter the counterfactual mean. The rule sets ``A1 = 1`` and
``A2 = 1{L2 > 0}``. The script integrates over ``(W1, W2)`` by a Gauss-Hermite rule. It integrates
over ``U`` with ``scipy.integrate.quad``, split where the rule switches, because a Gauss-Hermite
rule is inaccurate across that step.

Usage: ``python truth.py > truth.log``.
"""

import numpy as np
from scipy.special import expit

from scipy.integrate import quad
from scipy.stats import norm

nodes, weights = np.polynomial.hermite_e.hermegauss(40)
weights = weights / weights.sum()


def mean(a1, rule=None):
    """E[Y] with A1 = a1 and A2 = rule: 0, 1, or "engaged" for 1{L2 > 0}."""
    total = 0.0
    for x1, v1 in zip(nodes, weights):
        for x2, v2 in zip(nodes, weights):

            def integrand(u, a2):
                l2 = 0.6 * x1 + 0.9 * a1 + u
                index = (
                    -0.4 + 0.5 * a1 + 0.8 * a2 + 0.4 * l2 + 0.3 * x1 - 0.2 * x2 + 0.5 * np.tanh(l2)
                )
                return expit(index) * norm.pdf(u)

            if rule == "engaged":
                cut = -(0.6 * x1 + 0.9 * a1)
                inner = quad(integrand, -np.inf, cut, args=(0.0,))[0]
                inner += quad(integrand, cut, np.inf, args=(1.0,))[0]
            else:
                inner = quad(integrand, -np.inf, np.inf, args=(float(rule),))[0]
            total += v1 * v2 * inner
    return total


always = mean(1.0, 1)
never = mean(0.0, 0)
early = mean(1.0, 0)
late = mean(0.0, 1)
rule = mean(1.0, "engaged")
print(f"ey_regimen[always] {always:.6f}")
print(f"ey_regimen[never] {never:.6f}")
print(f"ey_regimen[early] {early:.6f}")
print(f"ey_regimen[late] {late:.6f}")
print(f"ate_regimen[always vs never] {always - never:.6f}")
print(f"ey_regimen[rule] {rule:.6f}")
print(f"ate_regimen[rule vs never] {rule - never:.6f}")
print(f"rule vs always {rule - always:.6f}")
