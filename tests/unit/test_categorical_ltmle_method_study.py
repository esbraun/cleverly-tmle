"""Structural gates for the categorical longitudinal evidence studies."""

from __future__ import annotations

import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import Folds
from tests import discrete_law_longitudinal_multivalue as law
from tests.studies import canonical_categorical_ltmle as ordinary
from tests.studies import canonical_categorical_ltmle_crossfit as crossfit
from tests.studies import categorical_longitudinal_common as common
from tests.studies.evidence.registry import ROOT

#: The R file both registered rows run their comparator from.
RUNNER = ROOT / "tests" / "canonical" / "categorical_ltmle_runner.R"

#: The arm order the runner writes its own mechanism tables in.
RUNNER_ARMS = ("standard", "high", "low")


def test_each_categorical_outer_fold_recovers_the_exact_law_and_gateaux_curve() -> None:
    """Every training complement and held-out fold contains one complete oracle law."""
    base = law.frame()
    frame = pd.concat([base] * common.N_FOLDS, ignore_index=True)
    folds = Folds(np.repeat(np.arange(common.N_FOLDS), len(base)), common.N_FOLDS)
    with patch("cleverly.longitudinal.estimator.make_folds", return_value=folds):
        result = common.fit(frame, cross_fit=True, configuration="both_correct")

    for name in common.CONTRASTS.values():
        assert result[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)
        expected = np.tile(np.repeat(law.eif_at(law.PROBS, name), law.COUNTS), common.N_FOLDS)
        np.testing.assert_allclose(result.influence_curves[name], expected, atol=2e-12, rtol=0.0)


def test_primary_crossfit_training_and_validation_sets_retain_all_three_arms() -> None:
    """No primary fold loses a categorical treatment level at either node."""
    frame, _ = crossfit.draw_scenario(common.SCENARIO, crossfit.PRIMARY_N, 0)
    result = crossfit.fit_cleverly(frame)
    expected = set(law.ARM_LABELS)
    for training, validation in result.folds:
        for column in ("A1", "A2"):
            assert set(frame.iloc[training][column]) == expected
            assert set(frame.iloc[validation][column]) == expected


def _first_follower(frame: pd.DataFrame) -> int:
    """The first row that stays on some declared plan through both treatment nodes.

    Which row this is matters, and it is the whole reason this helper exists.  Three levels
    at each of two nodes give nine arm pairs and the study declares five plans, so most rows
    follow none of them: at ``n=500`` on this law, 235 of them leave every plan at one node
    or the other.  A perturbation to such a row's outcome enters no sequential regression at
    all, and the equality assertions below would then hold because the two fits are bit for
    bit the same fit -- not because a fold kept its held-out rows out of its own training
    complement.  A control that passes when the estimator is right and passes when it is
    wrong is not a control, so the witness in
    :func:`test_a_held_out_outcome_cannot_enter_its_categorical_recursion` requires the
    perturbation to move the estimate before it asks what stayed still.
    """
    followed = np.zeros(len(frame), dtype=bool)
    for label in common.REGIMENS:
        first, second = common._plan_labels(frame, label)
        followed |= (frame["A1"].to_numpy() == first) & (frame["A2"].to_numpy() == second)
    candidates = np.flatnonzero(followed)
    assert candidates.size, "no row follows any declared plan, so no outcome is load bearing"
    return int(candidates[0])


def test_a_held_out_outcome_cannot_enter_its_categorical_recursion() -> None:
    """Each categorical prediction and update uses the row's training complement.

    Two claims, and the family needs both.  The perturbed outcome must reach the estimate,
    which is what makes it evidence about anything; and it must not reach the fitted values
    of the row it belongs to, which is what fold isolation means.
    """
    frame, _ = crossfit.draw_scenario(common.SCENARIO, 500, 1)
    original = crossfit.fit_cleverly(frame)
    changed = frame.copy()
    row = _first_follower(frame)
    changed.loc[row, "Y"] = 1.0 - changed.loc[row, "Y"]
    perturbed = crossfit.fit_cleverly(changed)

    np.testing.assert_array_equal(original.folds.assignment, perturbed.folds.assignment)
    moved = abs(
        float(original[common.DYNAMIC_NAME].psi) - float(perturbed[common.DYNAMIC_NAME].psi)
    )
    assert moved > 1e-6, (
        f"flipping row {row}'s outcome moved the dynamic contrast by {moved:.3e}, so the "
        f"equality checks below would pass on an estimator with no fold isolation at all"
    )

    displaced = 0.0
    for label in common.REGIMENS:
        for left, right in zip(
            original.fits[label].steps, perturbed.fits[label].steps, strict=True
        ):
            assert left.initial[row] == right.initial[row]
            assert left.targeted[row] == right.targeted[row]
            displaced = max(displaced, float(np.max(np.abs(left.initial - right.initial))))
    assert displaced > 1e-6, "no fitted value moved anywhere, so the equalities say nothing"


def test_serialized_folds_are_the_fitted_assignments_for_both_constructions() -> None:
    """The R payload carries the fitted fold instead of reconstructing it from a seed."""
    ordinary_sample, _, _ = common._replicate(ordinary.STUDY, False, common.SCENARIO, 0, 500)
    assert set(ordinary_sample["fold"]) == {0}

    frame = pd.concat([law.frame()] * common.N_FOLDS, ignore_index=True)
    truth = dict(law.TRUTH)
    planted = Folds(np.repeat(np.arange(common.N_FOLDS), len(law.frame())), 5)
    with (
        patch(
            "tests.studies.categorical_longitudinal_common.draw_for", return_value=(frame, truth)
        ),
        patch("cleverly.longitudinal.estimator.make_folds", return_value=planted),
    ):
        sample, truths, _ = common._replicate(crossfit.STUDY, True, common.SCENARIO, 0, len(frame))
    np.testing.assert_array_equal(sample["fold"].to_numpy(), planted.assignment)
    assert {row["estimand"] for row in truths} == set(truth)


def test_scrambled_raw_codes_do_not_match_the_semantic_arm_order() -> None:
    """The study would detect a raw-code or sorted-label substitution."""
    assert tuple(sorted(law.ARM_LABELS)) != law.ARM_LABELS
    assert tuple(law.ARM_LABELS[index] for index in range(3)) == ("standard", "high", "low")
    assert common.LEVELS == ("high", "low", "standard")


def _mechanism_design(
    frame: pd.DataFrame, *, second_node: bool, previous: np.ndarray | None = None
) -> np.ndarray:
    """Rebuild one mechanism design by hand, in the order the estimator hands it over.

    ``previous`` is the earlier arm the indicator block carries.  The estimator conditions on
    the *intervened* history, so in a real fit that is the plan's first-node arm rather than
    the observed one; the default is the observed column, which is what the check against the
    law needs.
    """
    columns = [frame["W"].to_numpy(dtype=float)]
    if "U" in frame:
        columns.append(frame["U"].to_numpy(dtype=float))
    if second_node:
        arm = frame["A1"].to_numpy() if previous is None else previous
        columns.append(frame["L2"].to_numpy(dtype=float))
        columns.extend((arm == level).astype(float) for level in common.LEVELS[1:])
    return np.column_stack(columns)


def _exact_probabilities(frame: pd.DataFrame, *, second_node: bool) -> np.ndarray:
    """The generating probabilities straight off the law, in the estimator's column order."""
    order = [law.ARM_LABELS.index(level) for level in common.LEVELS]
    w = frame["W"].to_numpy(dtype=int)
    if not second_node:
        return law.G1[w][:, order]
    a1 = np.array([law.ARM_LABELS.index(label) for label in frame["A1"]], dtype=int)
    return law.G2[w, a1, frame["L2"].to_numpy(dtype=int)][:, order]


@pytest.mark.parametrize("noise", [False, True], ids=("primary", "with-noise"))
def test_the_known_categorical_mechanism_reads_the_design_it_is_handed(noise: bool) -> None:
    """Positions and encodings, checked against the law rather than against the shape.

    The class recognises a design by its width, so a reordering *within* a width passes that
    check untouched while every probability it returns moves.  The primary cells would catch
    one through the paired R comparison, whose density ratios come off an independently
    written table.  The ``crossfit_overfitting`` cells would not, because no registered
    comparison fits their noise-augmented panel, which is why both widths are pinned here.

    The swap below is the mutation control.  A reordering is detected either because the
    class refuses the design outright or because the probabilities move; both count, and a
    swap that did neither would mean this test cannot witness one.
    """
    frame = law.sample(law.PROBS, 400, 11, noise=noise)
    mechanism = common.KnownCategoricalMechanism().fit(None, None)

    for second_node, width in ((False, 1 + int(noise)), (True, 4 + int(noise))):
        design = _mechanism_design(frame, second_node=second_node)
        assert design.shape[1] == width
        np.testing.assert_allclose(
            mechanism.predict_proba(design),
            _exact_probabilities(frame, second_node=second_node),
            atol=1e-15,
            rtol=0.0,
        )

    design = _mechanism_design(frame, second_node=True)
    exact = _exact_probabilities(frame, second_node=True)
    for left in range(design.shape[1] - 1):
        swapped = design.copy()
        swapped[:, [left, left + 1]] = swapped[:, [left + 1, left]]
        try:
            moved = float(np.max(np.abs(mechanism.predict_proba(swapped) - exact)))
        except (IndexError, ValueError):
            continue
        assert moved >= 0.25, (
            f"swapping second-node design columns {left} and {left + 1} moved the mechanism "
            f"by only {moved:.3f}, so this test cannot witness a reordering"
        )


@pytest.mark.parametrize("noise", [False, True], ids=("primary", "with-noise"))
def test_the_estimator_hands_the_mechanism_the_design_the_study_assumes(noise: bool) -> None:
    """The layout is read off a real fit, not restated from the same assumption twice.

    :func:`_mechanism_design` rebuilds the conditioning set by hand, so on its own it would
    agree with the mechanism class whether or not either matched the estimator.  This is the
    half that fails if ``history_design`` ever reorders its blocks: it records what the
    estimator actually passes and requires it to be that layout, for the width the primary
    cells use and for the width the ``crossfit_overfitting`` cells add a noise column to.

    The second-node block carries the *intervened* first-node arm, which is what the clever
    covariate conditions on.  So the expected designs are one per declared plan, and each one
    must be handed over at least once.
    """
    frame = law.sample(law.PROBS, 300, 3, noise=noise)
    seen: list[np.ndarray] = []
    original = common.KnownCategoricalMechanism.predict_proba

    def record(self: object, X: object) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        if len(design) == len(frame):
            seen.append(design)
        return original(self, X)  # type: ignore[arg-type]

    with patch.object(common.KnownCategoricalMechanism, "predict_proba", record):
        common.fit(frame, cross_fit=False, configuration="mechanism_correct")

    assert {design.shape[1] for design in seen} == {1 + int(noise), 4 + int(noise)}
    first_node = _mechanism_design(frame, second_node=False)
    second_node = {
        label: _mechanism_design(
            frame, second_node=True, previous=common._plan_labels(frame, label)[0]
        )
        for label in common.REGIMENS
    }

    matched: set[str] = set()
    for design in seen:
        if design.shape[1] == first_node.shape[1]:
            np.testing.assert_array_equal(design, first_node)
            continue
        hits = [label for label, plan in second_node.items() if np.array_equal(design, plan)]
        assert hits, "the estimator handed over a second-node design no declared plan builds"
        matched.update(hits)
    assert matched == set(common.REGIMENS), sorted(set(common.REGIMENS) - matched)


def test_the_known_categorical_mechanism_refuses_a_design_it_does_not_recognise() -> None:
    """An unrecognised width raises rather than reading whichever columns are present."""
    mechanism = common.KnownCategoricalMechanism().fit(None, None)
    with pytest.raises(ValueError, match="unexpected categorical mechanism design"):
        mechanism.predict_proba(np.zeros((4, 3)))


def _runner_vector(text: str, name: str) -> list[float]:
    """One ``name = c(...)`` numeric literal from the runner, or a failure that says so."""
    match = re.search(rf"^\s*{name} = c\(([^)]*)\)", text, re.MULTILINE)
    assert match is not None, (
        f"the categorical runner no longer writes {name} as one flat c(...) vector, so this "
        f"gate cannot read it. Re-read the runner rather than deleting the check"
    )
    return [float(value) for value in match.group(1).replace("\n", " ").split(",")]


def test_the_runners_transcribed_mechanism_is_the_python_law() -> None:
    """The comparator's own mechanism tables, checked cell by cell against the law.

    The runner writes the mechanism out by hand rather than importing it, which is what makes
    the comparator's density ratios independent evidence instead of a second reader of one
    array.  Independence is the point and transcription is the cost.  The adapter's screens
    compare the supplied ratios with ``lmtp``'s own on their zero pattern, their correlation
    and their mean, and a permutation of two arm *columns* leaves the zero pattern exactly
    where it was.  This is the check that reads the numbers.
    """
    text = RUNNER.read_text(encoding="utf-8")
    order = [law.ARM_LABELS.index(label) for label in RUNNER_ARMS]

    # The numbers below are read as flat vectors, so the row layout they are read *in* is an
    # assumption.  It is asserted rather than trusted: a reordered table with consistent
    # values would otherwise fail the value checks and read as a transcription error.
    for layout in (
        "W = rep(c(0, 1), each = 3)",
        'arm = rep(c("standard", "high", "low"), 2)',
        "W = rep(c(0, 1), each = 6)",
        'A1 = rep(rep(c("standard", "high", "low"), each = 2), 2)',
        "L2 = rep(c(0, 1), 6)",
    ):
        assert layout in text, f"the runner no longer lays its mechanism tables out as {layout}"

    first = np.asarray(_runner_vector(text, "probability"), dtype=float).reshape(2, 3)
    np.testing.assert_array_equal(first, law.G1[:, order])

    columns = {label: _runner_vector(text, label) for label in RUNNER_ARMS}
    row = 0
    for w in range(2):
        for first_arm in RUNNER_ARMS:
            for l2 in range(2):
                for second_arm in RUNNER_ARMS:
                    expected = law.G2[
                        w,
                        law.ARM_LABELS.index(first_arm),
                        l2,
                        law.ARM_LABELS.index(second_arm),
                    ]
                    assert columns[second_arm][row] == expected, (w, first_arm, l2, second_arm)
                row += 1
    assert row == 12, "the runner's second-node table no longer holds one row per W, A1 and L2"


def test_the_runner_fits_the_declared_plans_with_the_declared_arms() -> None:
    """Every plan the comparator fits assigns the arms the law's regimen assigns."""
    text = RUNNER.read_text(encoding="utf-8")
    plans = re.search(r"^plans <- c\(([^)]*)\)", text, re.MULTILINE)
    assert plans is not None, "the runner no longer declares its plans as one flat c(...) vector"
    assert [value.strip().strip('"') for value in plans.group(1).split(",")] == list(
        common.REGIMENS
    )

    constant = {
        label: (early, late)
        for label, early, late in re.findall(
            r'label == "(\w+)"\)[^\n]*\n?[^\n]*return\(cbind\(rep\("(\w+)", nrow\(frame\)\), '
            r'rep\("(\w+)", nrow\(frame\)\)\)\)',
            text,
        )
    }
    dynamic = {
        label
        for label, specification in law.REGIMEN_SPEC.items()
        if not isinstance(specification, str) and callable(specification[1])
    }
    for label, specification in law.REGIMEN_SPEC.items():
        if label in dynamic:
            continue
        arms = (specification, specification) if isinstance(specification, str) else specification
        assert constant.get(label) == arms, label
    assert dynamic == {"respond"}
    assert 'ifelse(frame$L2 == 1, "high", "low")' in text, (
        "the runner's dynamic plan no longer assigns high when L2 is one and low otherwise"
    )
