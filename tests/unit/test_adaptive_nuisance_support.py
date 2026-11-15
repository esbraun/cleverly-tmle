"""The ``adaptive_nuisance`` support qualifier (X54 part (h), plan sections 3.5 and 4.5).

A clustered fit that reports a cluster interval (the ``t`` and ``clustered`` tiers) whose
nuisance learners include a data-adaptive one carries the qualifier. It is soft: a row that does
not declare it still covers the fit. The label moves in two ways only: an in-sample fit on the
``t`` tier is experimental by theory (:data:`cleverly._support.ADAPTIVE_IN_SAMPLE_REASON`,
design review M2), and any other fit moves where ``few-cluster-adaptive-learners`` registers a
measured shortfall in :data:`cleverly._support_rows.ADAPTIVE_NUISANCE_SHORTFALLS`, on a
``t``-tier row below a count or, for a red stacked count at 40, on ``T10.unweighted`` at every
count (design review M3). These tests pin each branch with a witness, each label with its text,
and that every plot and every refute, validate and sensitivity operation answers on a labelled
point, stacked and ``LTMLE`` fit as it does without the label, because the qualifier changes no
number (design review M4).
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
from cleverly import _support, _support_rows
from cleverly._support import (
    ADAPTIVE_IN_SAMPLE_REASON,
    ADAPTIVE_NUISANCE,
    SOFT_QUALIFIERS,
    adaptive_nuisance,
    classify,
    nuisance_family,
    registered_rows,
)
from cleverly._support_rows import ADAPTIVE_NUISANCE_SHORTFALLS, AdaptiveNuisanceShortfall
from cleverly.datasets import make_clustered
from cleverly.estimators import TMLE
from cleverly.learners import GLM, SuperLearner
from tests.conftest import linear_in_sample
from tests.studies import few_cluster_adaptive_learners as study
from tests.studies import support_cases
from tests.unit import _capability_sweep_support as sweep

ENTRY = AdaptiveNuisanceShortfall(below=21, counts="10 and 20", splitting="stacked TMLE")
#: The endpoint of a stacked red at 40 clusters with no code defect (design review M3).
AT_EVERY_COUNT = AdaptiveNuisanceShortfall(below=None, counts="40", splitting="stacked TMLE")
LTMLE_ENTRY = AdaptiveNuisanceShortfall(below=21, counts="20", splitting="cross-fitted LTMLE")


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
    [(20, "t", True), (40, "clustered", True), (None, "iid", False), (5, "below_floor", False)],
)
def test_the_interval_tiers_carry_the_qualifier(
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
    assert adaptive.clusters == parametric.clusters == 20
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
        ("tmle_in_sample", "equal10", 40, True),
        ("tmle_stacked", "equal10", 40, True),
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


def test_the_registered_t_tier_evidence_is_parametric() -> None:
    """Each longitudinal t-tier case runs its study's own fit: none carries the qualifier.

    A shortfall registered on one of these rows can then never label the configuration its
    evidence measured. The point cases build their estimators with default learners, so they
    say nothing here; ``T37.few_cluster`` copies a learned-rule study whose outcome learner is a
    ``Pipeline``, which no shortfall of this study can name.
    """
    seen = []
    for case in support_cases.CASES:
        built = case.build()
        if (
            isinstance(built, support_cases.Observed)
            and built.key.kind == "longitudinal"
            and built.key.dependence == "t"
        ):
            seen.append(case.row)
            assert ADAPTIVE_NUISANCE not in built.key.qualifiers, case.row
    assert {"G06", "G07", "G26.clusters_20"} <= set(seen)


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


def _shortfalls(monkeypatch: pytest.MonkeyPatch, entries: dict[str, Any]) -> None:
    monkeypatch.setattr(_support_rows, "ADAPTIVE_NUISANCE_SHORTFALLS", entries)


def test_no_shortfall_is_registered_before_the_run() -> None:
    assert dict(ADAPTIVE_NUISANCE_SHORTFALLS) == {}
    estimate = _clustered(_tree(), stacked=True)["ate"]
    assert estimate.support.label == "supported"
    assert estimate.support.row == "T11.stacked"
    at_forty = _clustered(_tree(), clusters=40, stacked=True)["ate"]
    assert at_forty.support.label == "supported"
    assert at_forty.support.row == "T10.unweighted"


# In sample, by theory (design review M2).


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


# Stacked, by a measured shortfall.


def test_a_declared_shortfall_labels_the_stacked_fit_and_keeps_its_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _shortfalls(monkeypatch, {"T11.stacked": ENTRY})
    estimate = _clustered(_tree(), stacked=True)["ate"]
    assert estimate.support.label == "experimental"
    assert ENTRY.sentence("T11.stacked") in estimate.support.reason
    assert "few-cluster-adaptive-learners" in estimate.support.reason
    assert "F28" in estimate.support.evidence
    assert estimate.inference == "influence_curve"
    assert estimate.reference_df == 18
    assert np.all(np.isfinite(estimate.ci))


def test_a_declared_shortfall_leaves_a_parametric_stacked_fit_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _shortfalls(monkeypatch, {"T11.stacked": ENTRY})
    estimate = _clustered(LogisticRegression(max_iter=1000), stacked=True)["ate"]
    assert estimate.support.label == "supported"


def test_a_shortfall_holds_below_its_count_only(monkeypatch: pytest.MonkeyPatch) -> None:
    below_twenty = AdaptiveNuisanceShortfall(below=20, counts="10", splitting="stacked TMLE")
    _shortfalls(monkeypatch, {"T11.stacked": below_twenty})
    assert _clustered(_tree(), clusters=20, stacked=True)["ate"].support.label == "supported"
    assert _clustered(_tree(), clusters=12, stacked=True)["ate"].support.label == "experimental"


def test_the_shortfall_sentences_name_the_reading_and_the_count() -> None:
    """The limit sentence reads ``below``, and a shared red says the GLM arm reads the same."""
    library = AdaptiveNuisanceShortfall(below=20, counts="10", splitting="stacked TMLE")
    shared = AdaptiveNuisanceShortfall(below=20, counts="10", splitting="stacked TMLE", shared=True)
    assert library.sentence("T11.stacked") == (
        "Row T11.stacked is measured with parametric nuisance learners only below 20 clusters: "
        "the registered few-cluster evidence with the default learner library fails at 10 "
        "clusters for stacked TMLE (few-cluster-adaptive-learners)."
    )
    assert shared.sentence("T11.stacked") == (
        "Row T11.stacked has no passing measurement with the default learner library below 20 "
        "clusters: the default learner library covers below the floor at 10 clusters for "
        "stacked TMLE, as does a GLM fit with an estimated propensity on the same draws; it "
        "passes from 20 clusters (few-cluster-adaptive-learners)."
    )
    assert "fails" not in shared.sentence("T11.stacked")
    assert "below 40" not in library.sentence("T11.stacked")
    with pytest.raises(ValueError, match="needs below"):
        AdaptiveNuisanceShortfall(below=None, counts="40", splitting="stacked TMLE", shared=True)


def test_a_shared_shortfall_labels_the_stacked_fit_with_its_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = AdaptiveNuisanceShortfall(below=20, counts="10", splitting="stacked TMLE", shared=True)
    _shortfalls(monkeypatch, {"T11.stacked": shared})
    estimate = _clustered(_tree(), clusters=12, stacked=True)["ate"]
    assert estimate.support.label == "experimental"
    assert shared.sentence("T11.stacked") in estimate.support.reason


def test_mutation_a_detector_that_never_fires_leaves_the_stacked_fit_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _shortfalls(monkeypatch, {"T11.stacked": ENTRY})
    monkeypatch.setattr(_support, "adaptive_nuisance", lambda learners: False)
    assert _clustered(_tree(), stacked=True)["ate"].support.label == "supported"


def test_a_red_stacked_count_at_40_reaches_the_clustered_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Design review M3: the endpoint labels a 40-cluster stacked fit at every count."""
    _shortfalls(monkeypatch, {"T10.unweighted": AT_EVERY_COUNT})
    estimate = _clustered(_tree(), clusters=40, stacked=True)["ate"]
    assert estimate.support.row == "T10.unweighted"
    assert estimate.support.label == "experimental"
    assert AT_EVERY_COUNT.sentence("T10.unweighted") in estimate.support.reason
    assert "no study measures a larger count" in estimate.support.reason
    assert np.all(np.isfinite(estimate.ci))
    parametric = _clustered(LogisticRegression(max_iter=1000), clusters=40, stacked=True)
    assert parametric["ate"].support.label == "supported"


def test_every_shortfall_names_a_clustered_row() -> None:
    rows = {row.id: row for row in registered_rows()}
    for row_id, entry in ADAPTIVE_NUISANCE_SHORTFALLS.items():
        assert rows[row_id].key.dependence in {"t", "clustered"}
        if rows[row_id].key.dependence == "t":
            assert entry.below is not None and 10 < entry.below <= 40
        else:
            assert entry.below is None


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
def stacked(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The stacked adaptive fit under a declared ``T11.stacked`` shortfall."""
    _shortfalls(monkeypatch, {"T11.stacked": ENTRY})
    result = _clustered(_tree(), stacked=True)
    assert result["ate"].support.label == "experimental"
    return result


@pytest.fixture
def ltmle(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The cross-fitted clustered LTMLE under a declared ``G07`` shortfall (the sweep kind)."""
    _shortfalls(monkeypatch, {"G07": LTMLE_ENTRY})
    result = sweep.fit_clustered_ltmle_adaptive()
    for name in list(result.keys()):
        assert result[name].support.label == "experimental", name
        assert result[name].support.row == "G07"
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


@pytest.mark.parametrize("fixture", ["in_sample", "stacked", "ltmle"])
def test_the_summary_names_the_label(fixture: str, request: pytest.FixtureRequest) -> None:
    result = request.getfixturevalue(fixture)
    text = " ".join(str(result.summary(detail="full")).split())
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
    """The truncation plot names the experimental label in its notes; the contour draws."""
    result = request.getfixturevalue(fixture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truncation, contour = _sensitivity_plots(result)
    assert any("experimental" in line for line in plotting.notes(truncation))
    plt.close(truncation)
    plt.close(contour)


def test_the_forest_draws_the_shown_interval_of_the_labelled_fit(stacked: Any) -> None:
    estimate = stacked["ate"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plotting.forest(stacked)
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
