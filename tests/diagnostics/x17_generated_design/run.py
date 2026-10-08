"""The generated-design term that the per-arm outcome-adaptive curve omits.

The registered study ``ctmle-oat-per-arm`` measured reported standard errors 8 to 24 percent
below the Monte Carlo spread of the per-arm ``strategy="oat"`` fit, flat in ``n``, while its
oracle-design cells were calibrated.  Condition (v) of Benkeser, Cai and van der Laan (2020),
Theorem 1 (preprint arXiv:1901.05056v1, Appendix F), fails for an estimated outcome regression,
so the remainder ``R24`` is first order.  For a parametric outcome model it equals ``c' IF_beta``,
with ``c = E[(1 - g0 / G) dQbar / dbeta]`` for each arm and ``IF_beta`` the influence function of
the logistic maximum-likelihood fit.

This diagnostic fits the law ``binary_active`` of :mod:`tests.studies.oat_per_arm_laws` in
sample, ``REPLICATES`` times at each ``n`` in ``SIZES``, and reads the ``ate`` four ways.

=========================  ===================================================================
row                        fit
=========================  ===================================================================
``per-arm estimated Q``    the study subject: the interaction GLM outcome, the per-arm design
``+ generated-design term``  the same fit, with ``c' IF_beta`` added to the reported curve. It
                           uses the true mechanism ``g0``, so it is a check, not an estimator
``per-arm oracle Q``       the outcome pinned to the truth, so the design is fixed in advance
``TMLE known g0``          ordinary TMLE on the true mechanism, a direction control
=========================  ===================================================================

The reported standard error is ``plugin_std_error``: every ``oat`` fit withholds ``std_error``.
Replication ``r`` draws its frame with seed ``SEED_BASE + r``.  The reading goes to
``reading.txt`` in this directory.  It is a diagnostic only, and no test or study reads it.

    python -m tests.diagnostics.x17_generated_design.run
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from cleverly.estimators import CTMLE, TMLE
from tests.studies import oat_per_arm_laws as laws

HERE = Path(__file__).resolve().parent
LAW = laws.BINARY_ACTIVE
SIZES = (1_500, 6_000)
REPLICATES = 400
SEED_BASE = 900_000
#: The two-sided 95% normal critical value.
Z = 1.959964


def per_arm(frame: Any, outcome_learner: Any) -> Any:
    """The per-arm outcome-adaptive fit of the study's binary law, in sample."""
    return (
        CTMLE(
            strategy="oat",
            outcome_learner=outcome_learner,
            treatment_learner=laws.cubic_logit_learner(),
            cross_fit=False,
            simultaneous=False,
            estimands=("ey0", "ey1", "ate"),
            g_bounds=laws.G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def corrected_se(result: Any, frame: Any) -> float:
    """The reported ``ate`` curve plus ``c' IF_beta`` for the interaction GLM."""
    data = result.data
    n = data.n
    design = laws.arm_interactions(data.treatment_design(), indicators=1)
    x = np.column_stack([np.ones(n), design])
    y = np.asarray(data.outcome, dtype=float)
    q_obs = np.asarray(result.nuisance.outcome.observed)
    information = (x * (q_obs * (1 - q_obs))[:, None]).T @ x / n
    if_beta = np.linalg.solve(information, (x * (y - q_obs)[:, None]).T).T
    g0 = LAW.mechanism(frame["W1"].to_numpy(), frame["W3"].to_numpy())
    g = result.nuisance.propensity.bounded(result.config.g_bounds)
    total = np.zeros(n)
    for j, arm in enumerate(data.arm_codes):
        xa = np.column_stack(
            [np.ones(n), laws.arm_interactions(data.counterfactual_design(arm), indicators=1)]
        )
        qa = np.asarray(result.nuisance.outcome.arms[arm])
        gradient = xa * (qa * (1 - qa))[:, None]
        c = np.mean(gradient * (1.0 - g0[:, j] / g[:, j])[:, None], axis=0)
        sign = 1.0 if j == 1 else -1.0
        total = total + sign * (if_beta @ c)
    curve = np.asarray(result["ate"].influence_curve) + total
    return float(np.sqrt(np.var(curve, ddof=1) / n))


def known_g(frame: Any) -> Any:
    """Ordinary TMLE on the true mechanism of the law."""
    g0 = LAW.mechanism(frame["W1"].to_numpy(), frame["W3"].to_numpy())
    frame = frame.assign(p0=g0[:, 0], p1=g0[:, 1])
    return (
        TMLE(
            outcome_learner=laws.interaction_outcome(2),
            cross_fit=False,
            simultaneous=False,
            estimands=("ate",),
            g_bounds=(1e-6, 1 - 1e-6),
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", treatment_probabilities={0.0: "p0", 1.0: "p1"})
        .single()
    )


def _line(label: str, estimates: np.ndarray, errors: np.ndarray, truth: float) -> str:
    spread = float(np.std(estimates, ddof=1))
    ratio = float(errors.mean()) / spread
    coverage = float(np.mean(np.abs(estimates - truth) <= Z * errors))
    return (
        f"  {label:<25} sd {spread:.5f}  mean se {errors.mean():.5f}  "
        f"ratio {ratio:.3f}  cover {coverage:.3f}"
    )


def reading(n: int, replicates: int) -> list[str]:
    """The four rows at one sample size."""
    truth = float(LAW.truth()["ate"])
    keys = ("est", "se", "corrected", "oracle_est", "oracle_se", "known_est", "known_se")
    rows: dict[str, list[float]] = {key: [] for key in keys}
    for replicate in range(replicates):
        frame, _ = LAW.sample(n, SEED_BASE + replicate)
        fit = per_arm(frame, laws.interaction_outcome(2))
        rows["est"].append(fit["ate"].psi)
        rows["se"].append(fit["ate"].plugin_std_error)
        rows["corrected"].append(corrected_se(fit, frame))
        oracle = per_arm(frame, laws.OracleOutcome(LAW.name))
        rows["oracle_est"].append(oracle["ate"].psi)
        rows["oracle_se"].append(oracle["ate"].plugin_std_error)
        known = known_g(frame)
        rows["known_est"].append(known["ate"].psi)
        rows["known_se"].append(known["ate"].std_error)
    a = {key: np.asarray(value) for key, value in rows.items()}
    return [
        f"n={n} R={replicates}",
        _line("per-arm estimated Q", a["est"], a["se"], truth),
        _line("+ generated-design term", a["est"], a["corrected"], truth),
        _line("per-arm oracle Q", a["oracle_est"], a["oracle_se"], truth),
        _line("TMLE known g0", a["known_est"], a["known_se"], truth),
        f"  sqrt(n) sd of the per-arm estimate {np.std(a['est'], ddof=1) * np.sqrt(n):.4f}",
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--output", type=Path, default=HERE / "reading.txt")
    arguments = parser.parse_args(argv)
    lines: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in SIZES:
            lines.extend(reading(n, arguments.replicates))
            print("\n".join(lines[-6:]), flush=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
