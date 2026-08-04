r"""Is Tier 1's regime-entry column the quantity the estimator's bias actually is?

**It is not, and this is the script that measured it.**  ``benchmarks/drtmle_coverage.py``
reports ``n^alpha R2`` against the coefficient
:mod:`benchmarks.drtmle_injection` committed to, and C3's pilot found that column landing on
``+0.400`` to four decimals while the plain ``TMLE``'s :math:`\sqrt n` bias came in at ``0.1``
to ``0.6`` -- where ``docs/drtmle/coverage-study.md`` had sized ``2.5`` to ``4.2`` from the same
coefficient.  A column that is exactly right beside a prediction that is an order of magnitude
wrong is a sign that the two are not about the same quantity, and they are not:

.. math::

    R_2(\hat Q)   &= P_0\!\left[\tfrac{\hat g_a - g_{0,a}}{\hat g_a}(\hat Q_a - \bar Q_{0,a})\right]
                     \quad\text{the \emph{plug-in} remainder, at the initial regression} \\
    \hat\psi - \psi_0 &= (P_n - P_0)D^* + R_2(\bar Q^*)
                     \quad\text{the estimator's bias, at the \emph{targeted} one}

:func:`benchmarks.drtmle_injection.exact_remainder` computes the first -- its docstring says
so, and it is the honest name for what it integrates.  What was wrong was the *reading*: a
design note treated it as the bias, and nothing measured the difference.  This script does,
by evaluating one expression at two regressions on the same rows of the same fits, so nothing
varies between the two columns but which array goes into them.

:math:`(P_n - P_0)D^*` is mean-zero, so averaging over draws leaves :math:`R_2(\bar Q^*)`; the
replication count is sized to the gap being resolved, which is ``0.08`` against ``0`` at a
per-draw standard deviation near ``0.10``.

**What it found**, at ``q-drift`` over 24 draws -- the mean bias tracks :math:`R_2(\bar Q^*)`
and not :math:`R_2(\hat Q)`, which overstates it roughly twentyfold:

===========  =======================  ===============  =================
``n``        mean bias                :math:`R_2(\hat Q)`  :math:`R_2(\bar Q^*)`
===========  =======================  ===============  =================
600          ``-0.0036 +/- 0.0203``   ``+0.08082``     ``-0.0039``
1,200        ``-0.0054 +/- 0.0100``   ``+0.06797``     ``+0.0105``
2,400        ``+0.0083 +/- 0.0091``   ``+0.05716``     ``-0.0015``
===========  =======================  ===============  =================

The consequence is a **design** finding rather than a defect: Tier 1 injects its drift into
:math:`\hat Q`, where the fluctuation's own free parameter can absorb it, so the perturbation
never reaches the estimate and no choice of ``c`` makes that tier produce a coverage gap.  The
measured :math:`R_2(\hat Q)` column reproduces
:func:`~benchmarks.drtmle_injection.exact_remainder`'s quadrature to five decimals, which is
what says the two arms of this comparison are computed correctly rather than merely
differently.

Usage::

    python benchmarks/drtmle_tier1_bias.py
    python benchmarks/drtmle_tier1_bias.py --cell g-drift --draws 12 --sizes 600
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:  # the benchmarks package is importable either way, depending on the entry point
    from benchmarks import drtmle_injection as injection
except ImportError:  # pragma: no cover - direct `python benchmarks/drtmle_tier1_bias.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks import drtmle_injection as injection

from cleverly import TMLE
from cleverly.estimators.base import format_table
from cleverly.utils.bounds import OutcomeScaler

ARMS = (1.0, 0.0)

#: Enough to resolve ``0.08`` against ``0`` at a per-draw standard deviation near ``0.10``:
#: the standard error of the mean is then ``0.02``, so the two readings separate by four of
#: them.  Sized to the gap, as ``CLAUDE.md`` requires of any simulation.
DEFAULT_DRAWS = 24


def remainder(cell: str, n: int, covariates: Any, arms: dict[float, Any]) -> float:
    r"""``R_2`` for the ATE at whatever regression ``arms`` holds, on the raw outcome scale.

    One expression, called twice per fit -- once at the initial regression and once at the
    targeted one.  Writing it once is the point: two functions would be two chances to differ
    somewhere other than in the array under test.
    """
    estimated = injection.injected_mechanism(cell, n, covariates)
    true = injection._mechanism(injection.base_law(), covariates)
    per_arm = {}
    for arm in ARMS:
        ghat = injection._arm(estimated, arm)
        g0 = injection._arm(true, arm)
        # The truth is the injected regression minus its own perturbation, so it is read off
        # the same module the fit was fed from rather than recomputed a second way.
        truth = injection.injected_outcome(
            cell, n, covariates, arm
        ) - injection.outcome_perturbation(cell, n, covariates, arm)
        per_arm[arm] = float(np.mean((ghat - g0) / ghat * (arms[arm] - truth)))
    return per_arm[1.0] - per_arm[0.0]


def one_size(cell: str, n: int, draws: int, seed: int) -> list[str]:
    """One row: the mean bias against the same remainder at both regressions."""
    dgp = injection.base_law()
    truth = dgp.truth()["ate"]
    scaler = OutcomeScaler(*injection.Q_BOUNDS)
    collected = []
    for data_seed in np.random.SeedSequence(seed).generate_state(draws):
        frame, _ = dgp.sample(n, seed=int(data_seed))
        fit = (
            TMLE(**injection.settings(cell, n), random_state=0)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        covariates = np.asarray(fit.data.covariates, dtype=float)
        targeted_arms = fit.repeats[0].fluctuations["mean"].targeted.arms
        initial = {
            a: scaler.unscale_level(np.asarray(fit.nuisance.outcome.arms[a], float)) for a in ARMS
        }
        targeted = {a: scaler.unscale_level(np.asarray(targeted_arms[a], float)) for a in ARMS}
        collected.append(
            (
                float(fit.estimates["ate"].psi) - truth,
                remainder(cell, n, covariates, initial),
                remainder(cell, n, covariates, targeted),
            )
        )
    bias, initial_r2, targeted_r2 = (np.asarray(column) for column in zip(*collected, strict=True))
    error = float(np.std(bias, ddof=1) / np.sqrt(draws))
    root = np.sqrt(n)
    return [
        cell,
        f"{n:,}",
        str(draws),
        f"{bias.mean():+.5f} +/- {error:.5f}",
        f"{initial_r2.mean():+.5f}",
        f"{targeted_r2.mean():+.5f}",
        f"{injection.exact_remainder(cell, n)['r2_ate']:+.5f}",
        f"{root * bias.mean():+.3f}",
        f"{root * initial_r2.mean():+.3f}",
        f"{root * targeted_r2.mean():+.3f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="q-drift", choices=list(injection.CELLS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[600, 1200, 2400])
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=20250801)
    args = parser.parse_args()

    rows = [one_size(args.cell, n, args.draws, args.seed) for n in args.sizes]
    print(
        format_table(
            [
                "cell",
                "n",
                "draws",
                "mean bias",
                "R2(Q-hat)",
                "R2(Qbar*)",
                "declared",
                "sqrt(n) bias",
                "sqrt(n) R2(Q-hat)",
                "sqrt(n) R2(Qbar*)",
            ],
            rows,
        )
    )
    print(
        "\n`R2(Q-hat)` is the plug-in remainder and must reproduce `declared`, which is what\n"
        "says this comparison is computed correctly. `R2(Qbar*)` is the same expression at the\n"
        "targeted regression, and it is the one the mean bias has to track -- because\n"
        "`psi-hat - psi_0 = (Pn - P0)D* + R2(Qbar*)` and the first term is mean-zero across\n"
        "draws. Where the two columns differ by an order of magnitude, a design that sized a\n"
        "coverage gap off `declared` sized it off the wrong quantity."
    )


if __name__ == "__main__":
    main()
