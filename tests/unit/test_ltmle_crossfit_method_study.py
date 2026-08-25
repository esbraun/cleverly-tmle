"""Structural gates for the two cross-fitted longitudinal evidence studies."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LongitudinalData
from tests import discrete_law_longitudinal as end_law
from tests import discrete_law_survival as survival_law
from tests.canonical.lmtp_crossfit import audit
from tests.studies import canonical_ltmle_crossfit as end_study
from tests.studies import canonical_ltmle_survival_crossfit as survival_study
from tests.studies import ltmle_crossfit_properties as end_properties
from tests.studies import ltmle_survival_crossfit_properties as survival_properties
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism
from tests.studies.evidence.registry import Margins


@pytest.mark.parametrize(
    ("law", "fit", "names"),
    [
        (end_law, end_properties.fit, tuple(end_properties.CONTRASTS.values())),
        (
            survival_law,
            survival_properties.fit,
            tuple(survival_properties.CONTRASTS.values()),
        ),
    ],
    ids=("end-of-study", "survival"),
)
def test_each_outer_fold_recovers_the_exact_law_and_gateaux_curve(
    law: Any,
    fit: Any,
    names: tuple[str, ...],
) -> None:
    """Every training complement and held-out fold realizes the complete oracle law."""
    base = law.frame()
    frame = pd.concat([base] * 5, ignore_index=True)
    folds = Folds(np.repeat(np.arange(5), len(base)), 5)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=folds):
        result = fit(frame, "both_correct")

    for name in names:
        assert result[name].psi == pytest.approx(float(law.TRUTH[name]), abs=1e-12)
        expected = np.tile(np.repeat(law.eif(name), law.COUNTS), 5)
        np.testing.assert_allclose(result.influence_curves[name], expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "study",
    [end_study, survival_study],
    ids=("end-of-study", "survival"),
)
def test_primary_folds_are_balanced_and_the_r_payload_uses_them(study: Any) -> None:
    """The serialized fold is the fitted fold, not a reconstruction from the same seed.

    That distinction is the whole reason the ``fold`` column is written at all: the paired
    comparison is only paired if R splits the rows the way this fit split them, and comparing
    the column against a *second* fit of the same frame cannot tell the two apart, because a
    reconstruction from ``random_state`` would agree with it as well.  So the fit is given a
    fold assignment no seed would produce, and the column has to carry that one.
    """
    frame, truth = study.draw_scenario(study.SCENARIO, 500, 0)
    assert set(truth) == set(study.ESTIMANDS)

    counts = np.bincount(study.fit_cleverly(frame).folds.assignment, minlength=5)
    assert counts.max() - counts.min() <= 1

    # Contiguous blocks: balanced, so the estimator accepts it, and in an order
    # ``make_folds`` does not generate for any seed.
    planted = Folds(np.repeat(np.arange(5), len(frame) // 5), 5)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=planted):
        sample, _, _ = study._replicate((study.SCENARIO, 0, len(frame)))
    np.testing.assert_array_equal(sample["fold"].to_numpy(), planted.assignment)


#: The two panels :class:`KnownLongitudinalMechanism` is read against, with the column
#: declaration each one is fitted under.  Both present the same mechanism designs; the
#: survival panel merely carries event nodes between them, and that is the claim below.
_PANELS = {
    "end-of-study": (
        make_longitudinal,
        {
            "outcome": "Y",
            "treatment": ["A1", "A2"],
            "baseline": ["W1", "W2"],
            "time_varying": [[], ["L2"]],
            "censoring": ["C1", "C2"],
        },
    ),
    "survival": (
        make_longitudinal_survival,
        {
            "outcome": ["Y1", "Y2"],
            "treatment": ["A1", "A2"],
            "baseline": ["W1", "W2"],
            "time_varying": [[], ["L2"]],
            "censoring": ["C1", "C2"],
        },
    ),
}


@pytest.mark.parametrize("panel", list(_PANELS), ids=list(_PANELS))
def test_the_known_mechanism_reads_the_columns_it_believes_it_reads(panel: str) -> None:
    r"""The guard :class:`KnownLongitudinalMechanism` used to get from the paired comparison.

    The class reads its conditioning columns *by position* and says why that is safe: a width
    it does not recognise raises, and a *reordering* within a width would be caught because
    the agreement with R breaks loudly.  The cross-fitted overfitting cells have no paired
    comparison -- they run against ``make_longitudinal``, not against the discrete law, and no
    R implementation fits that pair -- so the reordering guard has to be written down instead
    of inherited.

    Checked against the generating probabilities rather than against a hard-coded column
    order, because that is the claim the property cells actually rest on: whatever
    :meth:`~cleverly.longitudinal.LongitudinalData.history_design` lays out, the mechanism
    handed to the estimator has to be the one the sampler drew from.  A permutation of two
    columns of equal width changes these numbers and leaves the shape alone, which is exactly
    the failure the width check cannot see.
    """
    generator, columns = _PANELS[panel]
    frame, _ = generator(n=400, seed=7, censoring=True, backend="pandas")
    data = LongitudinalData.from_frame(frame, **columns)

    w1 = frame["W1"].to_numpy(dtype=float)
    w2 = frame["W2"].to_numpy(dtype=float)
    l2 = np.nan_to_num(frame["L2"].to_numpy(dtype=float))
    a1 = np.nan_to_num(frame["A1"].to_numpy(dtype=float))

    treatment = KnownLongitudinalMechanism("treatment").fit(None, None)
    censoring = KnownLongitudinalMechanism("censoring").fit(None, None)
    expected = {
        ("treatment", 1): expit(0.3 * w1 - 0.4 * w2),
        ("treatment", 2): expit(0.5 * l2 + 0.6 * a1 - 0.2 * w2),
        ("censoring", 1): expit(2.2 + 0.3 * w1 - 0.3 * a1),
        ("censoring", 2): expit(2.4 + 0.2 * l2),
    }
    for time in (1, 2):
        design = data.history_design(time)
        np.testing.assert_allclose(
            treatment.predict_proba(design)[:, 1], expected[("treatment", time)], atol=1e-12
        )
        # The censoring model sits after the treatment decision, so it conditions on the
        # current arm as well -- one column wider at the same node.
        censored_design = data.history_design(time, include_current=True)
        np.testing.assert_allclose(
            censoring.predict_proba(censored_design)[:, 1],
            expected[("censoring", time)],
            atol=1e-12,
        )


def test_the_untargeted_plug_in_is_the_cross_fitted_fit_without_its_fluctuation() -> None:
    """The control has to agree with the estimator where agreement is forced.

    The analogue of the ordinary row's gate, on the frame the fold test above builds: five
    copies of the exact law, one per fold, so every outer *training complement* is four whole
    copies and realizes the law exactly.  With the saturated learner the fold-specific initial
    fit already solves every score, the fluctuation is a no-op, and the pooled plug-in and the
    five-fold targeted fit must return the same number.

    That is what says the difference the ``mechanism_correct`` cells report is the targeting
    step rather than a second estimator written differently.  It is an exact-law check, so it
    is blind to anything that vanishes at the truth -- which is the whole reason the finite
    sample bound below exists beside it.
    """
    base = end_law.frame()
    frame = pd.concat([base] * 5, ignore_index=True)
    folds = Folds(np.repeat(np.arange(5), len(base)), 5)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=folds):
        result = end_properties.fit(frame, "both_correct")

    for label in end_properties.REGIMENS:
        plug_in = end_properties.untargeted(frame, label, "both_correct", result.folds)
        assert plug_in == pytest.approx(float(result[f"ey_regimen[{label}]"].psi), abs=1e-9)


def test_the_comparator_audit_table_can_witness_the_failure_it_reports() -> None:
    """The refusal to publish a comparator has to rest on something reproducible.

    ``tests.canonical.lmtp_crossfit.audit`` is how a candidate comparator is examined *before*
    a row is registered against it, outside the evidence gate.  That is not hypothetical: it
    is how the mechanism defect these two rows were nearly published around was found -- an
    ``lmtp`` fitted with its own ``SL.glm`` density ratio covers 0.75 to 0.91 rather than
    0.95, which the paired table showed and no per-implementation gate would have.

    The comparator here is synthetic and deliberately over-confident -- the same estimates
    with the standard errors shrunk -- because the claim being checked is that the table can
    *see* an under-covering comparator.  A schema check alone would pass on a table that
    reported every comparator as fine.
    """
    record = dataclasses.replace(
        audit._record("end"), margins=dataclasses.replace(Margins(), bootstrap_replicates=200)
    )
    published = pd.read_csv(record.artifact("replicates.csv.gz"))
    # The subject's rows only.  The committed file now carries the real comparator beside
    # them, and pairing a synthetic comparator against a file that already has one gives two
    # rows per cell rather than a broken audit -- which reshapes into an error rather than a
    # wrong number, but only by luck.
    subject = published.loc[
        (published["implementation"] == record.implementation) & (published["replicate"] < 60)
    ].copy()

    comparator = subject.copy()
    comparator["implementation"] = audit.REFERENCE
    comparator["std_error"] *= 0.6
    critical = float(norm.ppf(0.975))
    comparator["ci_lower"] = comparator["estimate"] - critical * comparator["std_error"]
    comparator["ci_upper"] = comparator["estimate"] + critical * comparator["std_error"]
    comparator["covered"] = (
        (comparator["ci_lower"] <= comparator["truth"])
        & (comparator["truth"] <= comparator["ci_upper"])
    ).astype(int)

    table = audit.audit_table(pd.concat([subject, comparator], ignore_index=True), record, n_jobs=1)

    assert tuple(table.columns) == audit.AUDIT_COLUMNS
    assert sorted(table["estimand"]) == sorted(record.estimands)
    assert (table["replicates"] == 60).all()
    # The witness: shrinking the interval by 40% has to show up in the two endpoints the
    # committed refusal cites.  The coverage gate is one-sided -- what has to miss the floor
    # is the *lower* bound of the exact interval, which is the same endpoint the published
    # audit reports missing 0.90 on all five estimands.
    assert (table["coverage_ci_lower"] < record.margins.coverage_floor).all(), table
    assert (table["se_ratio"] < record.margins.se_ratio_sanity[0]).all(), table
