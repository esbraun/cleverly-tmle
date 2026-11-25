"""The ``adaptive_nuisance`` support qualifier (X54 part (h)).

A clustered fit on the ``t`` tier whose nuisance learners include a data-adaptive one carries
the qualifier. It is soft: a row that does not declare it still covers the fit. The label moves
in one way only: an in-sample fit is experimental by theory
(:data:`cleverly._support.ADAPTIVE_IN_SAMPLE_REASON`). ``few-cluster-adaptive-learners``
measures the default learner library at 10, 20 and 40 clusters, and every stacked ``TMLE`` and
cross-fitted ``LTMLE`` row passes, so those fits keep their rows' label, and the study is
evidence of ``T11.stacked``, ``T10.unweighted`` and ``G07``. These tests pin each branch with a
witness, the label with its text, and that every plot and every refute, validate and sensitivity
operation answers on an in-sample, stacked and ``LTMLE`` fit with the qualifier as it does
without it, because the qualifier changes no number.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cleverly.plotting as plotting
from cleverly import _support
from cleverly._support import (
    ADAPTIVE_IN_SAMPLE_REASON,
    ADAPTIVE_NUISANCE,
    SOFT_QUALIFIERS,
    adaptive_nuisance,
    classify,
    nuisance_family,
    registered_rows,
)
from cleverly.datasets import make_clustered
from cleverly.estimators import TMLE
from cleverly.learners import GLM, SuperLearner
from tests.conftest import linear_in_sample
from tests.studies import few_cluster_adaptive_learners as study
from tests.studies import support_cases
from tests.unit import _capability_sweep_support as sweep


def _classify(estimator: TMLE, clusters: int | None, **flags: Any) -> Any:
    frame = support_cases.frame(400 if clusters is None else 20 * clusters, clusters=clusters)
    return classify(*support_cases._build(estimator, frame, ("ate",), **flags))


def _tmle(**learners: Any) -> TMLE:
    return TMLE(cross_fit=False, simultaneous=False, **learners)


# ------------------------------------------------------------------------------- detection


def test_the_family_of_each_learner_kind() -> None:
    assert nuisance_family(GLM()) == "parametric"
    assert nuisance_family(LogisticRegression()) == "parametric"
    assert nuisance_family(SuperLearner(library=[GLM(), LogisticRegression()])) == "parametric"
    assert nuisance_family(SuperLearner(library=[("glm", GLM())])) == "parametric"
    # Witnesses of the adaptive branch: a default library, a library with a tree, a tree.
    assert nuisance_family(SuperLearner()) == "adaptive"
    assert nuisance_family(SuperLearner(library=[GLM(), DecisionTreeClassifier()])) == "adaptive"
    assert nuisance_family(DecisionTreeClassifier()) == "adaptive"
    assert nuisance_family(None) == "adaptive"
    assert adaptive_nuisance([GLM(), None]) and not adaptive_nuisance([GLM(), GLM()])


@pytest.mark.parametrize(
    ("learners", "expected"),
    [
        ({}, True),
        ({"outcome_learner": GLM(), "treatment_learner": GLM()}, False),
        ({"outcome_learner": LinearRegression(), "treatment_learner": LogisticRegression()}, False),
        (
            {
                "outcome_learner": SuperLearner(library=[GLM()]),
                "treatment_learner": SuperLearner(library=[GLM()]),
            },
            False,
        ),
        ({"outcome_learner": GLM(), "treatment_learner": DecisionTreeClassifier()}, True),
        ({"outcome_learner": DecisionTreeClassifier(), "treatment_learner": GLM()}, True),
    ],
    ids=["default", "glm", "sklearn_linear", "glm_super_learner", "tree_g", "tree_q"],
)
def test_a_t_tier_point_key_carries_the_qualifier_with_an_adaptive_learner(
    learners: dict[str, Any], expected: bool
) -> None:
    key = _classify(_tmle(**learners), clusters=20).key
    assert key.dependence == "t"
    assert (ADAPTIVE_NUISANCE in key.qualifiers) is expected


def test_a_known_mechanism_leaves_the_treatment_learner_unread() -> None:
    estimator = _tmle(outcome_learner=GLM(), treatment_learner=DecisionTreeClassifier())
    frame = support_cases.frame(400, clusters=20, known=True)
    key = classify(*support_cases._build(estimator, frame, ("ate",))).key
    assert "known_mechanism" in key.qualifiers
    assert ADAPTIVE_NUISANCE not in key.qualifiers
    # Witness: the same learners without the declared mechanism carry it.
    assert ADAPTIVE_NUISANCE in _classify(estimator, clusters=20).key.qualifiers


@pytest.mark.parametrize(
    ("clusters", "dependence", "expected"),
    [(20, "t", True), (40, "clustered", False), (None, "iid", False), (5, "below_floor", False)],
)
def test_the_t_tier_alone_carries_the_qualifier(
    clusters: int | None, dependence: str, expected: bool
) -> None:
    key = _classify(_tmle(), clusters=clusters).key
    assert key.dependence == dependence
    assert (ADAPTIVE_NUISANCE in key.qualifiers) is expected


def test_a_row_without_the_qualifier_still_covers_the_fit() -> None:
    adaptive = _classify(_tmle(), clusters=20)
    parametric = _classify(_tmle(outcome_learner=GLM(), treatment_learner=GLM()), clusters=20)
    assert ADAPTIVE_NUISANCE in adaptive.key.qualifiers
    assert adaptive.row is not None and parametric.row is not None
    assert adaptive.row.id == parametric.row.id == "T11.in_sample"
    assert frozenset({ADAPTIVE_NUISANCE}) == SOFT_QUALIFIERS
    # No registered row declares the qualifier, so it never narrows a row's coverage.
    assert not any(ADAPTIVE_NUISANCE in row.key.qualifiers for row in registered_rows())


def test_a_hard_qualifier_is_still_a_difference() -> None:
    key = _classify(_tmle(), clusters=20).key
    row = next(row for row in registered_rows() if row.id == "T11.in_sample")
    stratified = type(key)(**{**key.__dict__, "qualifiers": key.qualifiers | {"strata"}})
    assert "qualifiers" in stratified.differing_axes(row.key)
    assert "qualifiers" not in key.differing_axes(row.key)


@pytest.mark.parametrize(
    ("fit", "law", "clusters", "expected"),
    [
        ("tmle_in_sample", "equal10", 10, True),
        ("tmle_stacked", "nonlinear10", 20, True),
        ("tmle_in_sample", "equal10", 40, False),
        ("tmle_stacked", "equal10", 40, False),
        ("glm_in_sample", "equal10", 20, False),
        ("glm_stacked", "nonlinear10", 10, False),
        ("interpolating_control", "equal10", 20, True),
        ("ltmle_crossfit", "equal40", 20, True),
        ("ltmle_glm", "equal40", 20, False),
    ],
)
def test_the_study_configurations_carry_the_qualifier_as_declared(
    fit: str, law: str, clusters: int, expected: bool
) -> None:
    """The study's own fits, stopped where the classifier reads the key."""
    frame = study.draw(law, clusters, 0)
    observed = support_cases.capture(lambda: study.fit_one(fit, frame))
    assert (ADAPTIVE_NUISANCE in observed.key.qualifiers) is expected


def test_the_study_is_evidence_of_the_rows_its_gated_fits_reach() -> None:
    """Each case of the study runs its own fit to the classifier and lands on a row naming it.

    The stacked fits reach ``T11.stacked`` below 40 clusters and ``T10.unweighted`` at 40, and
    the cross-fitted ``LTMLE`` reaches ``G07``. The in-sample fits are reported only, so no row
    names the study for them.
    """
    rows = {row.id: row for row in registered_rows()}
    cases = [case for case in support_cases.CASES if case.slug == study.STUDY.slug]
    assert {case.row for case in cases} == {"T11.stacked", "T10.unweighted", "G07"}
    for case in cases:
        assert study.STUDY.slug in rows[case.row].evidence, case.row
    assert study.STUDY.slug not in rows["T11.in_sample"].evidence


# ---------------------------------------------------------------------------------- labels


def _clustered(outcome_learner: Any, clusters: int = 20, *, stacked: bool = False) -> Any:
    frame, _ = make_clustered(n=10 * clusters, cluster_size=10, seed=7, family="binomial")
    settings = linear_in_sample(
        outcome_learner=outcome_learner,
        estimands=("ey1", "ey0", "ate"),
        cross_fit=stacked,
        n_folds=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(**settings).fit(frame, outcome="Y", treatment="A", id="cluster").single()


def _tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)


def test_a_cross_fitted_adaptive_fit_keeps_its_row_label() -> None:
    """The registered run passes every stacked count, so the stacked fit stays supported."""
    estimate = _clustered(_tree(), stacked=True)["ate"]
    assert estimate.support.label == "supported"
    assert estimate.support.row == "T11.stacked"
    assert "few-cluster-adaptive-learners" in estimate.support.evidence
    at_forty = _clustered(_tree(), clusters=40, stacked=True)["ate"]
    assert at_forty.support.label == "supported"
    assert at_forty.support.row == "T10.unweighted"
    assert "few-cluster-adaptive-learners" in at_forty.support.evidence


# In sample, by theory.


def test_an_in_sample_adaptive_fit_is_experimental_by_theory() -> None:
    estimate = _clustered(_tree())["ate"]
    assert estimate.support.label == "experimental"
    assert estimate.support.row == "T11.in_sample"
    assert ADAPTIVE_IN_SAMPLE_REASON.format(row="T11.in_sample") in estimate.support.reason
    assert "F28" in estimate.support.evidence
    assert estimate.inference == "influence_curve"
    assert estimate.reference_df == 18
    assert np.all(np.isfinite(estimate.ci))


def test_an_in_sample_parametric_fit_stays_supported() -> None:
    estimate = _clustered(LogisticRegression(max_iter=1000))["ate"]
    assert estimate.support.label == "supported"
    assert estimate.support.row == "T11.in_sample"


def test_the_theory_reason_stays_off_a_stacked_fit() -> None:
    reason = _clustered(_tree(), stacked=True)["ate"].support.reason
    assert "Donsker" not in reason


def test_mutation_a_detector_that_never_fires_leaves_the_in_sample_fit_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control of the theory label: without the detection the reason is unseen."""
    monkeypatch.setattr(_support, "adaptive_nuisance", lambda learners: False)
    assert _clustered(_tree())["ate"].support.label == "supported"


# ------------------------------------------------------------- plots and assessment rows


def _sensitivity_plots(result: Any) -> list[Any]:
    """The sensitivity plots of a point fit: the truncation curve and the confounding contour."""
    return [
        plotting.sensitivity(result.diagnostics.truncation_curve()),
        plotting.confounding(result.sensitivity.contour()),
    ]


@pytest.fixture
def in_sample() -> Any:
    """The in-sample adaptive fit, experimental by theory (the sweep kind)."""
    result = sweep.fit_clustered_adaptive()
    assert result["ate"].support.label == "experimental"
    return result


@pytest.fixture
def stacked() -> Any:
    """The stacked adaptive fit, supported on ``T11.stacked``."""
    result = _clustered(_tree(), stacked=True)
    assert result["ate"].support.label == "supported"
    return result


@pytest.fixture
def ltmle() -> Any:
    """The cross-fitted clustered LTMLE with an adaptive outcome learner (the sweep kind)."""
    result = sweep.fit_clustered_ltmle_adaptive()
    for name in list(result.keys()):
        assert result[name].support.label == "supported", name
        assert result[name].support.row == "G07"
        assert "few-cluster-adaptive-learners" in result[name].support.evidence
    return result


@pytest.mark.parametrize(
    ("fixture", "kind"),
    [
        ("in_sample", "clustered+adaptive_nuisance"),
        ("stacked", "clustered+adaptive_nuisance"),
        ("ltmle", "ltmle+adaptive_nuisance"),
    ],
)
def test_every_assessment_row_answers_on_the_labelled_fit(
    fixture: str, kind: str, request: pytest.FixtureRequest
) -> None:
    """Refute, validate and sensitivity: each row that runs without the label runs with it."""
    result = request.getfixturevalue(fixture)
    arguments = sweep.sweep_arguments(result)
    assert sweep.problems(result, arguments) == []
    assert sweep.KINDS[kind].must_run <= sweep.ran(result, arguments)


def test_the_summary_names_the_label(in_sample: Any) -> None:
    text = " ".join(str(in_sample.summary(detail="full")).split())
    assert "few-cluster-adaptive-learners" in text


POINT_PLOTS = ("forest", "curve", "influence", "overlap", "balance", "calibration")


@pytest.mark.parametrize("fixture", ["in_sample", "stacked"])
@pytest.mark.parametrize("plot", POINT_PLOTS)
def test_every_plot_draws_the_labelled_point_fit(
    fixture: str, plot: str, request: pytest.FixtureRequest
) -> None:
    result = request.getfixturevalue(fixture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = getattr(plotting, plot)(result)
    plt.close(fig)


def _ltmle_plots() -> list[str]:
    from tests.unit.test_plotting_result_matrix import EXPECTED

    return sorted(
        p for p, outcome in EXPECTED["ltmle_adaptive_nuisance"].items() if outcome == "draws"
    )


@pytest.mark.parametrize("plot", _ltmle_plots())
def test_every_drawing_plot_draws_the_labelled_ltmle_fit(ltmle: Any, plot: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = getattr(plotting, plot)(ltmle)
    plt.close(fig)


@pytest.mark.parametrize("fixture", ["in_sample", "stacked"])
def test_the_sensitivity_plots_draw_and_note_the_label(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    """The truncation plot names the experimental label of the in-sample fit; both draw."""
    result = request.getfixturevalue(fixture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truncation, contour = _sensitivity_plots(result)
    noted = any("experimental" in line for line in plotting.notes(truncation))
    assert noted is (fixture == "in_sample")
    plt.close(truncation)
    plt.close(contour)


def test_the_forest_draws_the_shown_interval_of_the_labelled_fit(in_sample: Any) -> None:
    estimate = in_sample["ate"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plotting.forest(in_sample)
    xs = {
        round(float(x), 9)
        for collection in fig.axes[0].collections
        if collection.get_gid() == "cleverly-interval"
        for path in collection.get_paths()
        for x, _ in path.vertices
    }
    plt.close(fig)
    low, high = estimate.ci
    assert round(float(low), 9) in xs and round(float(high), 9) in xs
