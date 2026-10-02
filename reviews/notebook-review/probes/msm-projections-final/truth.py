"""Truths of the msm-projections page, recomputed from the structural equations of multi_arm_dgp.

The law (``src/cleverly/datasets/synthetic.py``, ``multi_arm_dgp``, gaussian family):
W ~ N(0, I_3); arm logits (0, 0.8 W1 - 0.4 W2, -0.5 W1 + 0.8 W2) for (low, medium, high);
Y = 0.6 step[a] + W1 - 0.5 W2 + 0.2 W3 + N(0, 0.8^2), with step = (0, 1, 2.4).

Run: .venv/Scripts/python.exe reviews/notebook-review/probes/msm-projections-final/truth.py
"""

import numpy as np

ARMS = ("low", "medium", "high")
STEP = np.array([0.0, 1.0, 2.4])
MEANS = 0.6 * STEP  # the covariate terms have mean zero
CONTACTS = np.array([1.0, 2.0, 6.0])
CADENCE_STEP = np.array([0.0, 1.0, 2.0])
# Logit coefficients on (W1, W2) per arm.
COEF = np.array([[0.0, 0.0], [0.8, -0.4], [-0.5, 0.8]])


def projection(x, weights):
    design = np.column_stack([np.ones(3), x])
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * MEANS))


def main():
    print("counterfactual means:", dict(zip(ARMS, MEANS.round(6), strict=True)))
    uniform = projection(CONTACTS, np.ones(3))
    print(f"per-contact projection, uniform weight: intercept {uniform[0]:.6f}, slope {uniform[1]:.6f}")
    print(f"slope as -2, -1, 3 over 14 contrast: {(-2 * MEANS[0] - MEANS[1] + 3 * MEANS[2]) / 14:.6f}")
    fixed = projection(CONTACTS, np.array([1.0, 10.0, 1.0]))
    print(f"per-contact projection, 1:10:1 weight: intercept {fixed[0]:.6f}, slope {fixed[1]:.6f}")
    step = projection(CADENCE_STEP, np.ones(3))
    print(f"per-step projection, uniform weight: intercept {step[0]:.6f}, slope {step[1]:.6f}")
    print("per-step line minus mean:", (np.column_stack([np.ones(3), CADENCE_STEP]) @ step - MEANS).round(6))

    # E[1/g_a] in closed form: 1/g_a = sum_b exp((c_b - c_a)' W), and E[exp(d' W)] = exp(|d|^2 / 2).
    for a, arm in enumerate(ARMS):
        value = sum(np.exp(np.sum((COEF[b] - COEF[a]) ** 2) / 2) for b in range(3))
        print(f"E[1/g({arm})] = {value:.4f}")

    rng = np.random.default_rng(20261002)
    w = rng.normal(size=(2_000_000, 2))
    logits = w @ COEF.T
    g = np.exp(logits - logits.max(1, keepdims=True))
    g /= g.sum(1, keepdims=True)
    for a, arm in enumerate(ARMS):
        column = g[:, a]
        print(
            f"true g({arm}) over 2e6 patients: min {column.min():.2e}, "
            f"share below 0.001 {np.mean(column < 0.001):.5f}, share below 0.01 {np.mean(column < 0.01):.4f}"
        )
    monte_carlo = (1 / g).mean(0)
    print("Monte Carlo E[1/g]:", dict(zip(ARMS, monte_carlo.round(3), strict=True)))


if __name__ == "__main__":
    main()
