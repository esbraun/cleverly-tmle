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

**The second table is why the design could be repaired rather than only diagnosed.**  The
pilot left the mechanism as a hypothesis -- ``coverage-study.md`` says its own display is
*"derived from the measurement rather than verified end to end"* and asks for a decomposition
of the existing injection before any new one.  This is that decomposition, and it closes as an
**identity** rather than as an approximate accounting:

.. math::

    b_a = c_a + \tilde\varepsilon_a P_0[u_a S_a],
    \qquad \tilde\varepsilon_a = -\frac{P_0[w_a h_a]}{P_0[w_a S_a]}

with :math:`S_a` the direction the fluctuation's one free parameter per arm moves
:math:`\bar Q_a` in.  So *"how much of* :math:`R_2(\hat Q)` *does the fitted* :math:`\varepsilon
\cdot s` *account for"* has an exact answer per arm, and what is left over is the coefficient
the repaired design declares.

**What it found**, at ``q-drift`` over 24 draws -- the mean bias tracks :math:`R_2(\bar Q^*)`
and not :math:`R_2(\hat Q)`:

===========  =======================  ===================  =====================
``n``        mean bias                :math:`R_2(\hat Q)`  :math:`R_2(\bar Q^*)`
===========  =======================  ===================  =====================
600          ``-0.0036 +/- 0.0203``   ``+0.08082``         ``-0.0039``
1,200        ``-0.0054 +/- 0.0100``   ``+0.06797``         ``+0.0105``
2,400        ``+0.0083 +/- 0.0091``   ``+0.05716``         ``-0.0015``
===========  =======================  ===================  =====================

That reading said *"roughly twentyfold"*, and the decomposition says it was an artefact of the
noise floor: the measured column is consistent with zero at those draw counts, and the
**exact** targeted coefficient is ``b_ATE = 0.00092`` against ``c_ATE = 0.40`` -- a factor of
``436``, not of twenty.  ``g-drift``'s is ``0.0259``, a factor of ``15``.

The consequence is a **design** finding rather than a defect: Tier 1 injects its drift where
the fluctuation's own free parameter can absorb it, so the perturbation never reaches the
estimate and no choice of ``c`` makes that tier produce a coverage gap.  The measured
:math:`R_2(\hat Q)` column reproduces
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


def one_size(cell: str, n: int, draws: int, seed: int) -> tuple[list[str], list[list[str]]]:
    """One bias row, and one decomposition row per arm."""
    dgp = injection.base_law()
    truth = dgp.truth()["ate"]
    scaler = OutcomeScaler(*injection.Q_BOUNDS)
    collected = []
    epsilons = []
    for data_seed in np.random.SeedSequence(seed).generate_state(draws):
        frame, _ = dgp.sample(n, seed=int(data_seed))
        fit = (
            TMLE(**injection.settings(cell, n), random_state=0)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        fluctuation = fit.repeats[0].fluctuations["mean"]
        covariates = np.asarray(fit.data.covariates, dtype=float)
        targeted_arms = fluctuation.targeted.arms
        initial = {
            a: scaler.unscale_level(np.asarray(fit.nuisance.outcome.arms[a], float)) for a in ARMS
        }
        targeted = {a: scaler.unscale_level(np.asarray(targeted_arms[a], float)) for a in ARMS}
        # `names` is the submodel's column order and is not the arm order this module reports
        # in, so the step is read *by name* rather than positionally -- reading it the other
        # way round is a mistake no assertion here would catch.
        by_name = dict(zip(fluctuation.names, np.asarray(fluctuation.epsilon, float), strict=True))
        epsilons.append([by_name[f"h{int(arm)}"] for arm in ARMS])
        collected.append(
            (
                float(fit.estimates["ate"].psi) - truth,
                remainder(cell, n, covariates, initial),
                remainder(cell, n, covariates, targeted),
            )
        )
    bias, initial_r2, targeted_r2 = (np.asarray(column) for column in zip(*collected, strict=True))
    error = float(np.std(bias, ddof=1) / np.sqrt(draws))
    targeted_error = float(np.std(targeted_r2, ddof=1) / np.sqrt(draws))
    root = np.sqrt(n)
    declared_c = injection.drift_coefficients(cell)["c_ate"]
    declared_b = injection.targeted_coefficients(cell)["b_ate"]
    bias_row = [
        cell,
        f"{n:,}",
        str(draws),
        f"{bias.mean():+.5f} +/- {error:.5f}",
        f"{initial_r2.mean():+.5f}",
        f"{targeted_r2.mean():+.5f} +/- {targeted_error:.5f}",
        f"{injection.exact_targeted_remainder(cell, n)['r2_ate']:+.5f}",
        f"{float(n) ** -injection.ALPHA * declared_c:+.5f}",
        f"{float(n) ** -injection.ALPHA * declared_b:+.5f}",
        f"{root * bias.mean():+.3f}",
        f"{root * targeted_r2.mean():+.3f}",
    ]
    return bias_row, _decomposition(cell, n, np.asarray(epsilons))


def _decomposition(cell: str, n: int, epsilons: Any) -> list[list[str]]:
    r"""How much of :math:`c_a` the fluctuation absorbs, and what survives it.

    The accounting is an **identity** -- ``absorbed = b_a - c_a`` by construction, since
    :math:`b_a = P_0[v_a h_a]` and :math:`v_a = 1 - \kappa_a w_a` -- so the column that means
    something is the *fitted* step beside the population one.  Where those agree, the
    population accounting describes the fits; where they do not, the fluctuation is doing
    something at this size that no coefficient predicts.
    """
    dgp = injection.base_law()
    scale = float(n) ** -injection.ALPHA
    rows = []
    for index, arm in enumerate(ARMS):
        c = dgp.expectation(
            lambda w, a=arm: injection.plugin_weight(cell, w, a) * injection.free_shape(cell, w, a)
        )
        b = dgp.expectation(
            lambda w, a=arm: (
                injection.targeted_weight(cell, w, a) * injection.free_shape(cell, w, a)
            )
        )
        fitted = float(np.mean(epsilons[:, index]))
        error = float(np.std(epsilons[:, index], ddof=1) / np.sqrt(epsilons.shape[0]))
        rows.append(
            [
                cell,
                f"{n:,}",
                f"{arm:.0f}",
                f"{fitted:+.5f} +/- {error:.5f}",
                f"{injection.population_epsilon(cell, n, arm):+.5f}",
                f"{scale * c:+.5f}",
                f"{scale * (b - c):+.5f}",
                f"{scale * b:+.5f}",
                f"{1.0 - b / c:.4f}" if c else "-",
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="q-drift", choices=list(injection.CELLS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[600, 1200, 2400])
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=20250801)
    args = parser.parse_args()

    measured = [one_size(args.cell, n, args.draws, args.seed) for n in args.sizes]
    print("What the mean bias tracks")
    print("=========================")
    print(
        format_table(
            [
                "cell",
                "n",
                "draws",
                "mean bias",
                "R2(Q-hat)",
                "R2(Qbar*)",
                "predicted R2(Qbar*)",
                "declared n^-a c",
                "declared n^-a b",
                "sqrt(n) bias",
                "sqrt(n) R2(Qbar*)",
            ],
            [row for row, _ in measured],
        )
    )
    print(
        "\n`R2(Q-hat)` is the plug-in remainder and must reproduce `declared n^-a c`, which is\n"
        "what says this comparison is computed correctly. `R2(Qbar*)` is the same expression at\n"
        "the targeted regression, and it is the one the mean bias has to track -- because\n"
        "`psi-hat - psi_0 = (Pn - P0)D* + R2(Qbar*)` and the first term is mean-zero across\n"
        "draws. It is read against `declared n^-a b`, which is the pre-flight condition\n"
        "`docs/drtmle/validation-plan.md` section 5 requires before any coverage dispatch."
    )

    print("\n\nWhat the fluctuation absorbs, per arm")
    print("=====================================")
    print(
        format_table(
            [
                "cell",
                "n",
                "arm",
                "fitted epsilon",
                "population epsilon",
                "n^-a c",
                "absorbed",
                "n^-a b (survives)",
                "share absorbed",
            ],
            [row for _, block in measured for row in block],
        )
    )
    print(
        "\n`absorbed` is `n^-a (b - c)`, the part of the plug-in remainder the fluctuation's one\n"
        "free parameter per arm removes; `n^-a b` is what survives it and is the estimator's\n"
        "bias. The accounting is an identity, so the column to read is `fitted epsilon` beside\n"
        "`population epsilon`: where they agree the population arithmetic describes the fits."
    )


if __name__ == "__main__":
    main()
