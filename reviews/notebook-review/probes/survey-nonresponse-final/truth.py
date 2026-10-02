"""Recompute every truth that docs/examples/survey-nonresponse.ipynb prints or quotes.

The structural equations are typed here from ``missing_outcome_dgp`` and
``missing_outcome_binary_dgp`` in ``src/cleverly/datasets/synthetic.py``; the script does not
import ``cleverly``.  Three independent standard-normal covariates (W1, W2, W3).

Continuous law at strength s (e = s - 1):

* ``P(A = 1 | W) = expit(0.4 W1 - 0.3 W2)``
* ``E[Y | A, W] = 1 + 1.2 A + 0.9 W1 + 0.6 W2 - 0.4 W3
  + e (1.1 tanh(1.5 W1) + 0.8 W2^2 - 0.9 A W1)``, Gaussian noise with SD 1
* ``P(Delta = 1 | A, W) = expit(1.2 + 0.6 A - (0.8 + 0.6 e) W1 + 0.3 W3)``

Binary law: ``P(A = 1 | W) = expit(0.4 W1 - 0.3 W2 + 0.2 W3)``,
``P(Y = 1 | A, W) = expit(-0.6 + 0.9 A + 0.7 W1 - 0.5 W2 + 0.4 W3)``.

Every expectation is a three-dimensional Gauss-Hermite quadrature (60 nodes per axis) over the
standard-normal covariates, so no Monte Carlo error enters.  The complete-case target is the
effect standardized to the respondents' covariate distribution,
``E[CATE(W) P(Delta = 1 | W)] / P(Delta = 1)``, with
``P(Delta = 1 | W) = g(W) pi(1, W) + (1 - g(W)) pi(0, W)``.  Output: ``truth.log``.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
nodes, weights = np.polynomial.hermite_e.hermegauss(60)
weights = weights / weights.sum()
w1, w2, w3 = (axis.ravel() for axis in np.meshgrid(nodes, nodes, nodes, indexing="ij"))
mass = np.einsum("i,j,k->ijk", weights, weights, weights).ravel()


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expect(values: np.ndarray) -> float:
    return float(np.sum(mass * values))


lines = []
for strength in (1.0, 2.0):
    e = strength - 1.0

    def q(a: float, e: float = e) -> np.ndarray:
        linear = 1.0 + 1.2 * a + 0.9 * w1 + 0.6 * w2 - 0.4 * w3
        return linear + e * (1.1 * np.tanh(1.5 * w1) + 0.8 * w2**2 - 0.9 * a * w1)

    def pi(a: float, e: float = e) -> np.ndarray:
        return expit(1.2 + 0.6 * a - (0.8 + 0.6 * e) * w1 + 0.3 * w3)

    g = expit(0.4 * w1 - 0.3 * w2)
    cate = q(1.0) - q(0.0)
    respond = g * pi(1.0) + (1.0 - g) * pi(0.0)
    p_respond = expect(respond)
    lines += [
        f"strength {strength:g}: EY1 {expect(q(1.0)):.6f}, EY0 {expect(q(0.0)):.6f}, "
        f"ATE {expect(cate):.6f}",
        f"strength {strength:g}: P(Delta = 1) {p_respond:.6f}, "
        f"P(Delta = 1 | A = 0) {expect((1 - g) * pi(0.0)) / expect(1 - g):.6f}, "
        f"P(Delta = 1 | A = 1) {expect(g * pi(1.0)) / expect(g):.6f}",
        f"strength {strength:g}: E[W1 | Delta = 1] {expect(w1 * respond) / p_respond:.6f}, "
        f"respondent-standardized effect {expect(cate * respond) / p_respond:.6f}",
    ]

gb = expit(0.4 * w1 - 0.3 * w2 + 0.2 * w3)
ey1 = expect(expit(-0.6 + 0.9 + 0.7 * w1 - 0.5 * w2 + 0.4 * w3))
ey0 = expect(expit(-0.6 + 0.7 * w1 - 0.5 * w2 + 0.4 * w3))
lines.append(
    f"binary: ey0 {ey0:.6f}, ey1 {ey1:.6f}, ate {ey1 - ey0:.6f}, rr {ey1 / ey0:.6f}, "
    f"or {(ey1 / (1 - ey1)) / (ey0 / (1 - ey0)):.6f}"
)
(HERE / "truth.log").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))
