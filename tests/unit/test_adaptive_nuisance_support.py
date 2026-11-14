"""The ``adaptive_nuisance`` support qualifier (X54 part (h), plan sections 3.5 and 4.5).

A clustered fit on the ``t`` tier whose nuisance learners include a data-adaptive one carries
the qualifier. It is soft: a row that does not declare it still covers the fit, so the label
moves only where ``few-cluster-adaptive-learners`` registers a measured shortfall in
:data:`cleverly._support_rows.ADAPTIVE_NUISANCE_SHORTFALLS`. These tests pin each branch of the
detection with a witness, the label under a declared shortfall with the declared text, and that
every plot and every refute, validate and sensitivity operation answers on such a fit as it does
without the qualifier, because the qualifier changes no number.
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

ENTRY = AdaptiveNuisanceShortfall(below=21, counts="10 and 20", splitting="in-sample TMLE")


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


@pytest.mark.parametrize("clusters", [None, 40, 5])
def test_only_the_t_tier_carries_the_qualifier(clusters: int | None) -> None:
    key = _classify(_tmle(), clusters=clusters).key
    assert key.dependence != "t"
    assert ADAPTIVE_NUISANCE not in key.qualifiers


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
        ("tmle_in_sample", "equal10", 40, False),
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


def _clustered(outcome_learner: Any, clusters: int = 20) -> Any:
    frame, _ = make_clustered(n=10 * clusters, cluster_size=10, seed=7, family="binomial")
    settings = linear_in_sample(outcome_learner=outcome_learner, estimands=("ey1", "ey0", "ate"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(**settings).fit(frame, outcome="Y", treatment="A", id="cluster").single()


def _tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)


def test_no_shortfall_is_registered_before_the_run() -> None:
    assert dict(ADAPTIVE_NUISANCE_SHORTFALLS) == {}
    estimate = _clustered(_tree())["ate"]
    assert estimate.support.label == "supported"
    assert estimate.support.row == "T11.in_sample"


def test_a_declared_shortfall_labels_the_adaptive_fit_and_keeps_its_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_support_rows, "ADAPTIVE_NUISANCE_SHORTFALLS", {"T11.in_sample": ENTRY})
    estimate = _clustered(_tree())["ate"]
    assert estimate.support.label == "experimental"
    assert ENTRY.sentence("T11.in_sample") in estimate.support.reason
    assert "few-cluster-adaptive-learners" in estimate.support.reason
    assert "F28" in estimate.support.evidence
    assert estimate.inference == "influence_curve"
    assert estimate.reference_df == 18
    assert np.all(np.isfinite(estimate.ci))


def test_a_declared_shortfall_leaves_a_parametric_fit_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_support_rows, "ADAPTIVE_NUISANCE_SHORTFALLS", {"T11.in_sample": ENTRY})
    estimate = _clustered(LogisticRegression(max_iter=1000))["ate"]
    assert estimate.support.label == "supported"


def test_a_shortfall_holds_below_its_count_only(monkeypatch: pytest.MonkeyPatch) -> None:
    below_twenty = AdaptiveNuisanceShortfall(below=20, counts="10", splitting="in-sample TMLE")
    monkeypatch.setattr(
        _support_rows, "ADAPTIVE_NUISANCE_SHORTFALLS", {"T11.in_sample": below_twenty}
    )
    assert _clustered(_tree(), clusters=20)["ate"].support.label == "supported"
    assert _clustered(_tree(), clusters=12)["ate"].support.label == "experimental"


def test_mutation_a_detector_that_never_fires_leaves_the_label_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control of the label test: without the detection the declared shortfall is unseen."""
    monkeypatch.setattr(_support_rows, "ADAPTIVE_NUISANCE_SHORTFALLS", {"T11.in_sample": ENTRY})
    monkeypatch.setattr(_support, "adaptive_nuisance", lambda learners: False)
    assert _clustered(_tree())["ate"].support.label == "supported"


def test_every_shortfall_names_a_t_tier_row() -> None:
    rows = {row.id: row for row in registered_rows()}
    for row_id, entry in ADAPTIVE_NUISANCE_SHORTFALLS.items():
        assert rows[row_id].key.dependence == "t"
        assert 10 < entry.below <= 40


# ------------------------------------------------------------- plots and assessment rows


@pytest.fixture
def shortfall(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The adaptive fit under a declared shortfall, as a failed run would leave it."""
    monkeypatch.setattr(_support_rows, "ADAPTIVE_NUISANCE_SHORTFALLS", {"T11.in_sample": ENTRY})
    result = sweep.fit_clustered_adaptive()
    assert result["ate"].support.label == "experimental"
    return result


def test_every_assessment_row_answers_under_the_shortfall(shortfall: Any) -> None:
    """Refute, validate and sensitivity: each row that runs without the shortfall runs with it."""
    arguments = sweep.sweep_arguments(shortfall)
    assert sweep.problems(shortfall, arguments) == []
    assert sweep.KINDS["clustered+adaptive_nuisance"].must_run <= sweep.ran(shortfall, arguments)


def test_the_summary_names_the_shortfall(shortfall: Any) -> None:
    text = " ".join(str(shortfall.summary(detail="full")).split())
    assert "few-cluster-adaptive-learners" in text


@pytest.mark.parametrize("plot", ["forest", "influence", "overlap", "balance", "calibration"])
def test_every_plot_draws_the_labelled_fit(shortfall: Any, plot: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = getattr(plotting, plot)(shortfall)
    plt.close(fig)


def test_the_forest_draws_the_shown_interval_of_the_labelled_fit(shortfall: Any) -> None:
    estimate = shortfall["ate"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plotting.forest(shortfall)
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
