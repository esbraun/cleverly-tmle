"""Structural gates for the two competing-risk LTMLE evidence studies."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LongitudinalData
from tests import discrete_law_competing as law
from tests.studies import canonical_ltmle_competing as ordinary
from tests.studies import canonical_ltmle_competing_crossfit as crossfit
from tests.studies import ltmle_competing_crossfit_properties as crossfit_properties
from tests.studies import ltmle_competing_properties as properties


def test_the_primary_studies_report_every_unique_parameter_once() -> None:
    """The dynamic plan's duplicate first horizon stays out of repeated sampling."""
    assert len(ordinary.ESTIMANDS) == 16
    assert len(set(ordinary.ESTIMANDS)) == len(ordinary.ESTIMANDS)
    assert crossfit.ESTIMANDS == ordinary.ESTIMANDS

    for cause in law.CAUSES:
        dynamic = law.TRUTH[f"cif_regimen[continue_if_l2, {cause} @ t=1]"]
        always = law.TRUTH[f"cif_regimen[always, {cause} @ t=1]"]
        assert dynamic == pytest.approx(always, abs=1e-15)
        assert f"cif_regimen[continue_if_l2, {cause} @ t=1]" not in ordinary.ESTIMANDS


def test_the_ordinary_study_reserves_an_independent_resampling_namespace() -> None:
    assert ordinary.STUDY.resampling_seed == 20260828
    assert ordinary.STUDY.resampling_seed != crossfit.STUDY.seed


@pytest.mark.parametrize("study", [ordinary, crossfit], ids=("ordinary", "cross-fitted"))
def test_each_study_declares_the_two_cause_properties(study: Any) -> None:
    assert study.PROPERTY_LABELS == ("relapse_dynamic_t2", "death_static_t2")
    cells = study.STUDY.property_cells
    assert set(cells["competing_risk_recursion_necessity"]) == {
        "relapse_always_t2__all_cause",
        "relapse_always_t2__cause_specific_control",
        "death_always_t2__all_cause",
        "death_always_t2__cause_specific_control",
    }
    assert ("crossfit_overfitting" in cells) is (study is crossfit)


#: The two registered property cells, as (label, plan, cause).  ``never`` is the reference.
NULL_CELLS = (
    ("death_static_t2", "always", "death"),
    ("relapse_dynamic_t2", "continue_if_l2", "relapse"),
)


def baseline_only(probs: np.ndarray, label: str, cause: str, horizon: int) -> float:
    """The cause-specific incidence an analysis that never conditions on ``L2`` reports.

    Written longhand off the support rather than by disabling something in the estimator, for
    the reason every deliberate-mutation control in this suite is: a flag on the code under
    audit makes the control a statement about a branch in it.

    There is one difference from :func:`law.functional`, and it is one line.  The identified
    functional takes the ``L2``-density-weighted average of the *per-stratum* second hazards,
    ``sum_l P(L2 = l | .) events_l / uncensored_l``.  This takes the pooled ratio
    ``(sum_l events_l) / (sum_l uncensored_l)``, which is what a fit that regressed on
    ``(W, A1, A2)`` and never on ``L2`` would estimate.  ``A2`` and ``C2`` both depend on
    ``L2``, so pooling reweights it, and that reweighting is the bias a longitudinal fit
    exists to remove.

    The survival factor stays ``survived / reached``, an all-cause mass over the cells with
    ``j1 = 0``, and never ``1 - hazard1``.  Dropping ``L2`` and reading the survival factor
    cause-specifically are two different mistakes with two different controls, and a witness
    that made both at once could not say which one it had caught.

    The dynamic plan's second arm *is* ``L2``, so the arm cannot be marginalised without
    deleting the plan.  What is marginalised here is the *adjustment*, not the protocol: the
    loop still reads the arm off ``L2``, which is only what an analyst does to decide who
    followed the rule, while the numerator and denominator pool across ``L2`` instead of
    conditioning on it.  That needs no special case -- a static plan takes ``a2`` from a
    scalar and the rule takes it from ``l2``, and the pooling below is identical either way.
    """
    j = law.CAUSES.index(cause) + 1
    node1, node2 = law.REGIMEN_ARMS[label]
    total = law._mass(probs)
    psi = 0.0
    for w in (0, 1):
        a1 = law._arm(node1, w)
        share = law._mass(probs, w=w) / total
        reached = law._mass(probs, w=w, a1=a1, c1=1)
        hazard1 = law._mass(probs, w=w, a1=a1, c1=1, j1=j) / reached
        if horizon == 1:
            psi += share * hazard1
            continue
        survived = law._mass(probs, w=w, a1=a1, c1=1, j1=0)
        uncensored = events = 0.0
        for l2 in (0, 1):
            a2 = law._arm(node2, w, l2)
            uncensored += law._mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1)
            events += law._mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1, j2=j)
        psi += share * (hazard1 + (survived / reached) * (events / uncensored))
    return psi


def crude(probs: np.ndarray, label: str, cause: str, horizon: int) -> float:
    """The same reading with the baseline dropped as well: no ``W``, and no ``L2``.

    The other direction of the same claim.  :func:`baseline_only` says the null needs ``L2``;
    this says it also still needs ``W``, so neither horizon's cell is a null a raw comparison
    of arms would pass.
    """
    j = law.CAUSES.index(cause) + 1
    node1, node2 = law.REGIMEN_ARMS[label]
    a1 = law._arm(node1, 0)
    reached = law._mass(probs, a1=a1, c1=1)
    hazard1 = law._mass(probs, a1=a1, c1=1, j1=j) / reached
    if horizon == 1:
        return hazard1
    survived = law._mass(probs, a1=a1, c1=1, j1=0)
    uncensored = events = 0.0
    for w in (0, 1):
        for l2 in (0, 1):
            a2 = law._arm(node2, w, l2)
            uncensored += law._mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1)
            events += law._mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1, j2=j)
    return hazard1 + (survived / reached) * (events / uncensored)


def _contrast(oracle: Any, probs: np.ndarray, plan: str, cause: str, horizon: int) -> float:
    return oracle(probs, plan, cause, horizon) - oracle(probs, "never", cause, horizon)


def test_the_null_is_exact_and_keeps_both_competing_causes() -> None:
    assert properties.NULL_PROBS.sum() == pytest.approx(1.0, abs=1e-15)
    assert all(value == pytest.approx(0.0, abs=1e-14) for value in properties.NULL_TRUTH.values())
    assert np.all(properties.NULL_H1 > 0.0)
    assert np.all(properties.NULL_H2 > 0.0)
    assert np.all(properties.NULL_H1.sum(axis=0) < 1.0)
    assert np.all(properties.NULL_H2.sum(axis=0) < 1.0)

    # Realised exactly by an N-row sample, like the law itself.  ``law.counts`` refuses a
    # hazard off the quarter grid, and a null off it samples perfectly well while silently
    # ceasing to agree with any exact-law control built on ``law.frame()``.
    counts = law.counts(properties.NULL_H1, properties.NULL_H2)
    assert counts.sum() == law.N
    assert counts.min() >= 1

    # Nonzero witnesses, each the negation of a way this null could have gone flat.  The
    # first hazard is what zeroes the first horizon, so it must not move with the arm; it
    # must still move with the baseline, or nothing confounds the first node.
    assert np.array_equal(properties.NULL_H1[:, :, 0], properties.NULL_H1[:, :, 1])
    assert properties.NULL_H1[0, 0, 0] != properties.NULL_H1[0, 1, 0], "relapse flat in W"
    assert properties.NULL_H1[1, 0, 0] != properties.NULL_H1[1, 1, 0], "death flat in W"

    # And the second hazard carries the longitudinal content: it moves with L2 on the never
    # path at W = 1 for both causes, and with the arm at either node.
    assert properties.NULL_H2[0, 1, 0, 0, 0] != properties.NULL_H2[0, 1, 0, 1, 0]
    assert properties.NULL_H2[1, 1, 0, 0, 0] != properties.NULL_H2[1, 1, 0, 1, 0]
    assert properties.NULL_H2[1, 0, 0, 0, 0] != properties.NULL_H2[1, 0, 1, 0, 0], "flat in A1"
    assert properties.NULL_H2[0, 0, 0, 0, 0] != properties.NULL_H2[0, 0, 0, 0, 1], "flat in A2"


def test_the_horizon_two_null_is_one_a_longitudinal_fit_has_to_work_for() -> None:
    """The deliberate-mutation control: dropping ``L2`` must miss the null.

    The witnesses above say each hazard moves with something.  They do not say an estimator
    has to be longitudinal to find the truth, and that is the claim the type-I cell rests on.
    A null a baseline-only standardisation already recovers cannot tell a sequential-regression
    fit from a pair of cross-sections, and the first version of these constants was one.
    """
    for label, plan, cause in NULL_CELLS:
        naive = _contrast(baseline_only, properties.NULL_PROBS, plan, cause, 2)
        assert abs(naive) > 1e-3, (
            f"an analysis that ignores L2 already recovers the horizon-two {label} null"
        )


def test_the_horizon_one_null_carries_no_longitudinal_content() -> None:
    """And the other direction, pinned so the published limitation cannot drift from it.

    Nothing time varying precedes ``J1``, so at the first horizon the baseline-only analysis
    *is* the identified functional and the cell cannot witness longitudinal adjustment.  It is
    still not a null no estimator has to work for: both the treatment and the censoring
    mechanism depend on ``W``, and the first hazard does too, so a crude comparison of arms is
    biased under it at either horizon.
    """
    for _, plan, cause in NULL_CELLS:
        assert _contrast(baseline_only, properties.NULL_PROBS, plan, cause, 1) == pytest.approx(
            0.0, abs=1e-15
        )
        for horizon in (1, 2):
            assert abs(_contrast(crude, properties.NULL_PROBS, plan, cause, horizon)) > 1e-3


def test_each_power_control_has_a_material_cause_specific_effect() -> None:
    assert properties.POWER_TRUTH["relapse_dynamic_t2"] > 0.10
    assert properties.POWER_TRUTH["death_static_t2"] < -0.10


def test_the_overfitting_panel_has_two_balanced_absorbing_causes() -> None:
    frame, truth = crossfit_properties._balanced_competing_panel(4_000, 21)
    assert abs(truth) > 0.01
    for node in (1, 2):
        both = (frame[f"R{node}"] == 1.0) & (frame[f"D{node}"] == 1.0)
        assert not bool(both.any())
        assert int((frame[f"R{node}"] == 1.0).sum()) > 100
        assert int((frame[f"D{node}"] == 1.0).sum()) > 100
    first_event = (frame["R1"] == 1.0) | (frame["D1"] == 1.0)
    assert frame.loc[first_event, ["L2", "A2", "C2"]].isna().all().all()
    assert (frame.loc[frame["R1"] == 1.0, "R2"] == 1.0).all()
    assert (frame.loc[frame["D1"] == 1.0, "D2"] == 1.0).all()


def test_the_known_mechanisms_read_the_declared_history_design() -> None:
    frame = ordinary.draw_from_seed(ordinary.SCENARIO, 600, 19)[0]
    data = LongitudinalData.from_frame(
        frame,
        outcome=law.outcome_columns(),
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    treatment = ordinary.KnownCompetingMechanism("treatment").fit(None, None)
    censoring = ordinary.KnownCompetingMechanism("censoring").fit(None, None)
    for time in (1, 2):
        history = data.history_design(time)
        index = np.rint(np.nan_to_num(history)).astype(int)
        expected_g = (
            law.G1[index[:, 0]] if time == 1 else law.G2[index[:, 0], index[:, 2], index[:, 1]]
        )
        np.testing.assert_allclose(treatment.predict_proba(history)[:, 1], expected_g)

        current = data.history_design(time, include_current=True)
        index = np.rint(np.nan_to_num(current)).astype(int)
        expected_c = (
            law.C1[index[:, 0], index[:, 1]]
            if time == 1
            else law.C2[index[:, 0], index[:, 2], index[:, 1], index[:, 3]]
        )
        np.testing.assert_allclose(censoring.predict_proba(current)[:, 1], expected_c)


def test_the_all_cause_recursion_agrees_at_the_law_and_the_mutation_does_not() -> None:
    """The control must fail only when the other cause leaves the risk set."""
    frame = law.frame()
    result = properties.fit(frame, "both_correct")
    for cause in law.CAUSES:
        name = f"cif_regimen[always, {cause} @ t=2]"
        correct = properties.untargeted(frame, "always", cause, 2, "both_correct", result.folds)
        wrong = properties.untargeted(
            frame,
            "always",
            cause,
            2,
            "both_correct",
            result.folds,
            cause_specific_survival=True,
        )
        assert correct == pytest.approx(law.TRUTH[name], abs=1e-12)
        assert abs(wrong - law.TRUTH[name]) > 0.02


def test_the_cross_fitted_r_payload_uses_the_realized_outer_folds() -> None:
    frame, truth = crossfit.draw_scenario(crossfit.SCENARIO, 1_000, 0)
    planted = Folds(np.repeat(np.arange(5), len(frame) // 5), 5)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=planted):
        sample, written_truth, rows = crossfit._replicate((crossfit.SCENARIO, 0, len(frame)))

    np.testing.assert_array_equal(sample["fold"].to_numpy(), planted.assignment)
    assert {row["estimand"] for row in written_truth} == set(truth)
    assert {row["estimand"] for row in rows} == set(crossfit.ESTIMANDS)


@pytest.mark.parametrize("study", [ordinary, crossfit], ids=("ordinary", "cross-fitted"))
def test_the_frozen_r_study_matches_beyond_its_acceptance_margin(study: Any) -> None:
    path = study.STUDY.artifact("replicates.csv.gz")
    if not path.exists():
        pytest.skip("the new study artifacts have not been generated yet")
    rows = pd.read_csv(path)
    paired = rows.pivot(
        index=["scenario", "replicate", "estimand"],
        columns="implementation",
        values=["estimate", "std_error"],
    )
    for column, tolerance in {"estimate": 5e-6, "std_error": 5e-7}.items():
        difference = paired[(column, study.STUDY.implementation)] - paired[(column, "lmtp")]
        assert np.max(np.abs(difference)) < tolerance

    movement = (
        rows.assign(targeting_movement=(rows["estimate"] - rows["initial_estimate"]).abs())
        .groupby("implementation")["targeting_movement"]
        .mean()
    )
    assert (movement > 0.005).all(), movement


#: The probe command each fixture README documents, as `(replicates, n)`.
_PROBE = re.compile(r"--replicates (?P<replicates>\d+) --n (?P<n>\d+) --primary-only")


def _followed_to_the_second_node(frame: pd.DataFrame, plan: str) -> pd.Series:
    """Units that reached the second event node still on ``plan``."""
    first, second = law.REGIMEN_ARMS[plan]
    l2 = np.nan_to_num(frame["L2"].to_numpy()).astype(int)
    arm2 = (
        np.full(len(frame), float(second))
        if np.ndim(second) == 0
        else np.asarray(second)[frame["W"].to_numpy().astype(int), l2]
    )
    event_free = (frame["R1"] == 0.0) & (frame["D1"] == 0.0)
    return (
        (frame["A1"] == float(first))
        & (frame["C1"] == 1.0)
        & event_free
        & (frame["A2"].to_numpy() == arm2)
        & (frame["C2"] == 1.0)
    )


@pytest.mark.parametrize("study", [ordinary, crossfit], ids=("ordinary", "cross-fitted"))
def test_the_documented_probe_size_supports_every_cause_and_plan(study: Any) -> None:
    """The README's smoke command has to be one that runs.

    Nothing else checks it.  No test executes a fixture README, and the failure is a per-panel
    refusal rather than a deterministic one: at ``--n 500`` this study refused at replicate 5
    of 8 and fitted the other seven, so running the command once was not enough to find it
    either.  The size is read out of the README rather than restated here, so the two cannot
    drift.

    The precondition is checked rather than the fit run.  ``LongitudinalError`` is raised when
    no unit following a regimen through the second node is observed to leave through the
    cause being estimated, and counting that costs no fit -- where fitting every documented
    replicate for both studies would be the most expensive test in this file.
    """
    readme = study.STUDY.artifact("README.md").read_text(encoding="utf-8")
    probe = _PROBE.search(readme)
    assert probe is not None, "the fixture README no longer documents a probe command"
    replicates, n = int(probe["replicates"]), int(probe["n"])

    thinnest = min(
        int((_followed_to_the_second_node(frame, plan) & (frame[column] == 1.0)).sum())
        for replicate in range(replicates)
        for frame in [study.draw_scenario(study.SCENARIO, n, replicate)[0]]
        for plan in law.REGIMEN_ARMS
        for column in ("R2", "D2")
    )
    # Two, not one.  A probe size that clears the refusal by a single unit clears it only for
    # the replicates drawn today, and the next seed change puts it back under.
    assert thinnest >= 2, (
        f"the probe documented in {study.STUDY.artifacts.name}/README.md draws a panel whose "
        f"thinnest cause and plan cell holds {thinnest} second-node events at n={n}; the fit "
        f"refuses a cell with none, so this command cannot be run as written"
    )
