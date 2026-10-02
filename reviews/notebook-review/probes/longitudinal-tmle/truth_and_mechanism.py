"""Independent truth recomputation and failure-mode mechanism probe.

Re-implements the structural equations of make_longitudinal from the source text
(src/cleverly/datasets/longitudinal.py lines 140-170, 365-381). Does NOT import cleverly.
"""

import numpy as np

rng = np.random.default_rng(20261001)


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def p_y(w1, w2, l2, a1, a2):
    return expit(-0.4 + 0.5 * a1 + 0.8 * a2 + 0.4 * l2 + 0.3 * w1 - 0.2 * w2 + 0.5 * np.tanh(l2))


# ---- 1. Counterfactual truth by Rao-Blackwellized Monte Carlo (10^7 draws) ----
def cf_means(n_total=10_000_000, chunk=1_000_000, l2_a1=0.9):
    sums = {k: 0.0 for k in ("always", "never", "early", "late", "rule")}
    sq = {k: 0.0 for k in sums}
    ys = {k: 0.0 for k in sums}
    for _ in range(n_total // chunk):
        w1 = rng.standard_normal(chunk)
        w2 = rng.standard_normal(chunk)
        e = rng.standard_normal(chunk)
        out = {}
        for name, a1, a2 in (("always", 1, 1), ("never", 0, 0), ("early", 1, 0), ("late", 0, 1)):
            l2 = 0.6 * w1 + l2_a1 * a1 + e
            out[name] = p_y(w1, w2, l2, a1, a2)
        l2 = 0.6 * w1 + l2_a1 * 1 + e
        out["rule"] = p_y(w1, w2, l2, 1, (l2 > 0).astype(float))
        for k, v in out.items():
            sums[k] += v.sum()
            sq[k] += (v ** 2).sum()
            # a direct Bernoulli draw too, as a fully simulated counterfactual
            ys[k] += rng.binomial(1, v).sum()
    res = {}
    for k in sums:
        m = sums[k] / n_total
        sd = np.sqrt(sq[k] / n_total - m * m)
        res[k] = (m, sd / np.sqrt(n_total), ys[k] / n_total)
    return res


res = cf_means()
print("== counterfactual means, 1e7 draws (RB mean, MC se, Bernoulli-sim mean) ==")
for k, (m, se, yb) in res.items():
    print(f"  {k:7s} {m:.5f}  se {se:.6f}  bernoulli {yb:.5f}")
print(f"  always-never {res['always'][0] - res['never'][0]:.5f}")
print(f"  rule-never   {res['rule'][0] - res['never'][0]:.5f}")
print(f"  rule-always  {res['rule'][0] - res['always'][0]:.5f}")

# ---- 2. Mechanism probe for the failure mode ----
# Large observational draw (no clusters: clustering preserves the L2 marginal and only
# adds dependence; it does not move population limits).
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(64)
gh_w = gh_w / np.sqrt(2 * np.pi)


def observational(n, l2_a1=0.9, g2_l2=0.5, c2_l2=0.2):
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    a1 = rng.binomial(1, expit(0.3 * w1 - 0.4 * w2)).astype(float)
    c1 = rng.binomial(1, expit(2.2 + 0.3 * w1 - 0.3 * a1)).astype(float)
    l2 = 0.6 * w1 + l2_a1 * a1 + rng.standard_normal(n)
    a2 = rng.binomial(1, expit(g2_l2 * l2 + 0.6 * a1 - 0.2 * w2)).astype(float)
    c2 = rng.binomial(1, expit(2.4 + c2_l2 * l2)).astype(float)
    return w1, w2, a1, c1, l2, a2, c2


def limits(n=400_000, l2_a1=0.9, g2_l2=0.5, c2_l2=0.2):
    w1, w2, a1, c1, l2, a2, c2 = observational(n, l2_a1, g2_l2, c2_l2)
    s = (c1 == 1) & (c2 == 1) & (a1 == a2)
    w1s, w2s, l2s = w1[s], w2[s], l2[s]
    # (i) adjusted for L2, true conditional mean, averaged over the selected sample S
    adj_s = np.mean(p_y(w1s, w2s, l2s, 1, 1) - p_y(w1s, w2s, l2s, 0, 0))
    # (ii) the same L2-held contrast over the whole population (no selection)
    adj_pop = np.mean(p_y(w1, w2, l2, 1, 1) - p_y(w1, w2, l2, 0, 0))
    # (iii) baseline-only limit: E[Y | A1=A2=a, C1=C2=1, W] by quadrature over L2 | W, a
    def m_given_w(a):
        mu = 0.6 * w1s[:, None] + l2_a1 * a + gh_x[None, :]
        g2 = expit(g2_l2 * mu + 0.6 * a - 0.2 * w2s[:, None])
        g2 = g2 if a == 1 else 1 - g2
        c = expit(2.4 + c2_l2 * mu)
        num = (p_y(w1s[:, None], w2s[:, None], mu, a, a) * g2 * c * gh_w).sum(1)
        den = (g2 * c * gh_w).sum(1)
        return num / den
    base_s = np.mean(m_given_w(1) - m_given_w(0))
    # truth under these coefficients
    e = rng.standard_normal(n)
    t = np.mean(p_y(w1, w2, 0.6 * w1 + l2_a1 + e, 1, 1) - p_y(w1, w2, 0.6 * w1 + e, 0, 0))
    return dict(truth=t, adjusted_limit_S=adj_s, adjusted_contrast_pop=adj_pop,
                baseline_limit_S=base_s, share_S=s.mean())


print()
print("== failure-mode limits (true conditional means; no learner error) ==")
for label, kw in (
    ("notebook law", {}),
    ("A1->L2 removed (l2_a1=0)", {"l2_a1": 0.0}),
    ("L2->A2 removed (g2_l2=0)", {"g2_l2": 0.0}),
    ("L2->A2 and L2->C2 removed", {"g2_l2": 0.0, "c2_l2": 0.0}),
):
    r = limits(**kw)
    print(f"  {label}")
    for k, v in r.items():
        print(f"     {k:24s} {v:+.4f}")
    print(f"     adjusted bias {r['adjusted_limit_S'] - r['truth']:+.4f}   "
          f"baseline bias {r['baseline_limit_S'] - r['truth']:+.4f}")
