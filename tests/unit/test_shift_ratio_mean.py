"""The shift support report's per-fold mean density ratio.

The density ratio ``h = g(a - delta | w) / g(a | w)`` at the observed dose is the density
of the shifted dose law with respect to the observed one, so its mean under the true law
is 1 while the shifted dose stays inside the support.  A held-out ratio from a poor
density estimate is mis-normalized fold by fold, and the influence-curve standard error
inherits that error (``reviews/notebook-review/investigations/iv-n3.md``, F3).

Witness: on ``make_shift_dose(n=3000, seed=9000)``, a bare booster density cross-fitted
over three folds reads fold means 1.46, 1.20 and 1.87 for the ``+1.0`` shift in the
IV-N3 probe (``probe_ic.csv``, configuration ``xfit_boost``).  Control: the exact law at
320 bins reads 1.03, 0.98 and 1.07 on the same folds (``xfit_oracle320``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    ModelSpec,
    ModifiedTreatmentPolicyEffect,
    PointTreatment,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import make_shift_dose, shift_dgp
from cleverly.interventions import Shift
from cleverly.learners.density import bin_edges
from tests.studies.canonical_shift_policies import OracleShiftDensity

SEED = 9000
SHIFTS = ((0.0, "current practice"), (1.0, "+1.0"))


def _fit(density: str, *, cross_fit: bool):  # type: ignore[no-untyped-def]
    frame, _ = make_shift_dose(
        n=3000, seed=SEED, policies=tuple((delta, None, name) for delta, name in SHIFTS)
    )
    bins = 320 if density == "oracle" else 40
    if density == "oracle":
        dose = np.asarray(frame["A"], dtype=float)
        edges = tuple(float(value) for value in bin_edges(dose, bins))
        learner = OracleShiftDensity(shift_dgp(), edges)
    else:
        learner = HistGradientBoostingClassifier(random_state=SEED)
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
        ),
    ).identify(
        ModifiedTreatmentPolicyEffect(
            tuple(Shift(delta, cap=None, name=name) for delta, name in SHIFTS)
        )
    )
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LinearRegression(),
            treatment_learner=learner,
            density_bins=bins,
        ),
        cross_fitting=CrossFitting(n_folds=3) if cross_fit else CrossFitting(enabled=False),
        targeting=Targeting(q_bounds=(-30.0, 40.0)),
        runtime=Runtime(random_state=SEED, n_jobs=1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(method=method)


def _independent_fold_means(result, column: int) -> list[float]:  # type: ignore[no-untyped-def]
    ratio = np.asarray(result.nuisance.policies.ratio[:, column], dtype=float)
    return [float(ratio[test].mean()) for _, test in result.nuisance.folds]


def test_a_poor_held_out_density_reads_a_fold_mean_far_from_one() -> None:
    result = _fit("boost", cross_fit=True)
    row = result.diagnostics.support()["+1.0"]
    assert len(row.fold_mean_ratio) == 3
    assert row.fold_mean_ratio == pytest.approx(_independent_fold_means(result, 1), abs=1e-12)
    # The IV-N3 probe read 1.46, 1.20 and 1.87 on these folds.
    assert max(row.fold_mean_ratio) > 1.5
    assert row.mean_ratio > 1.3
    assert "per fold" in row.summary()


def test_the_exact_density_reads_fold_means_near_one() -> None:
    result = _fit("oracle", cross_fit=True)
    row = result.diagnostics.support()["+1.0"]
    assert row.fold_mean_ratio == pytest.approx(_independent_fold_means(result, 1), abs=1e-12)
    # A fold of 1000 rows with E[h^2] = 2.77 has a sampling SD of about 0.04.
    assert all(abs(value - 1.0) < 0.1 for value in row.fold_mean_ratio)
    assert abs(row.mean_ratio - 1.0) < 0.05


def test_an_in_sample_fit_reports_one_fold_equal_to_the_overall_mean() -> None:
    result = _fit("oracle", cross_fit=False)
    row = result.diagnostics.support()["+1.0"]
    assert row.fold_mean_ratio == (row.mean_ratio,)
    assert row.mean_ratio == pytest.approx(
        float(np.mean(result.nuisance.policies.ratio[:, 1])), abs=1e-12
    )
    # The natural course has ratio 1 at every row with positive density.
    assert result.diagnostics.support()["current practice"].mean_ratio == pytest.approx(1.0)


def test_the_summary_states_the_reference_mean_for_each_shift_kind() -> None:
    # The reference is supported policy mass for either sign and cap choice.
    # Empirical means need not exactly equal the population expectation.
    from cleverly.interventions.policy import PolicySupport

    row = PolicySupport(
        name="+1.0",
        policy="Shift(delta=1.0, cap=None)",
        min_density=0.1,
        ratio_quantiles={0.5: 1.0},
        max_ratio=2.0,
        effective_sample_size=900.0,
        ess_ratio=0.9,
        moved_fraction=1.0,
        capped_fraction=0.0,
        unsupported=0,
        mean_ratio=0.9,
        fold_mean_ratio=(0.9,),
    )
    line = row.summary().splitlines()[-1]
    assert "P(d(A, W) in the conditional support)" in line
    assert "1 when the policy preserves support" in line
    assert "sampling and density estimation" in line
    assert "do not establish true support or identification" in line
    assert "1 under the true density;" not in line
