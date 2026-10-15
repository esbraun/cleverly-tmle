"""Assessment and sensitivity of a policy fit at a vector treatment with a dose.

A vector with a continuous component has no arms, as a continuous dose has none.  Each
assessment row reads the policy means, the fitted mechanisms and the stored nuisances, so
every row the scalar dose fit supports runs on the vector, and every row it refuses is
refused with a reason that names the vector.  The witnesses compare the two capability
tables, run each supported row against a value it must reproduce, and check that the
placebo keeps the joint law of the components.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import tests.discrete_law_vector_coarsened as coarse_law
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Shift
from cleverly.validation.refute import _permuted_treatment
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

from . import test_point_vector_treatment as point_tests
from . import test_vector_density_route as density_tests

pytestmark = pytest.mark.xdist_group("point_vector_assessment")


def _capabilities(result: Any) -> dict[str, tuple[bool, str]]:
    rows = (*result.diagnostics.capabilities, *result.sensitivity.capabilities)
    return {row.operation: (row.available, str(row.status)) for row in rows}


def _coarsened_fits(**declared: Any) -> tuple[Any, Any]:
    """A vector fit and a scalar dose fit of the coarsened law, with the same declarations."""
    from sklearn.linear_model import LogisticRegression

    frame = coarse_law.frame()
    learners = {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": LogisticRegression(),
        "missingness_learner": LogisticRegression(),
        "intermediate_learner": LogisticRegression(),
        "cross_fit": False,
        "ratio": "classifier",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector = CausalData.from_frame(
            frame,
            outcome="Y",
            treatment=["F", "D"],
            covariates=["W"],
            continuous_treatment=["D"],
            **declared,
        )
        dose = CausalData.from_frame(
            frame,
            outcome="Y",
            treatment="D",
            covariates=["W", "F"],
            treatment_kind="continuous",
            **declared,
        )
        policies = [point_tests.joint("natural"), point_tests.joint("dose up")]
        fitted = TMLE(policies=policies, **learners).fit(vector)
        scalar = TMLE(policies=[Shift(0.0, cap=None), point_tests.UP], **learners).fit(dose)
    return fitted, scalar


def test_the_capability_table_is_the_scalar_dose_table() -> None:
    """Every row has the availability and the status of the one-column dose fit."""
    vector, scalar = point_tests._sample_fits()
    table = _capabilities(vector)
    assert table == _capabilities(scalar)
    for operation in ("support", "score_equations", "nuisance_models", "truncation_curve"):
        assert table[operation][0], operation


def test_the_capability_table_with_a_missing_outcome_is_the_scalar_dose_table() -> None:
    vector, scalar = _coarsened_fits(delta="Delta")
    assert _capabilities(vector.single()) == _capabilities(scalar.single())


def test_the_nuisance_report_reads_every_fitted_mechanism() -> None:
    """The missingness, intermediate and outcome models each have a row."""
    results, _ = _coarsened_fits(delta="Delta", intermediate="Z")
    for result in results.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = result.diagnostics.nuisance_models().to_frame()
        assert {"missingness", "intermediate", "outcome"} <= set(frame["model"])


def test_a_truncation_curve_row_equals_the_refit_at_its_bound() -> None:
    """The bound reaches the categorical factor of the vector ratio, as a refit's does.

    The law of the M1 witness has ``P(F = 1 | W = 0) = 1 / 4``, so the bound 0.3 binds and
    moves the estimate.  The curve's row at 0.3 equals a fit with ``g_bounds=(0.3, 0.7)``.
    """
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_frame(
            density_tests._m1_frame(),
            outcome="Y",
            treatment=["F", "D"],
            covariates=["W"],
            continuous_treatment=["D"],
        )
        settings = {
            "policies": [density_tests.SET_FLAG_ONLY],
            "outcome_learner": LogisticRegression(),
            "treatment_learner": CellProbabilities(),
            "cross_fit": False,
            "simultaneous": False,
            "ratio": "density",
        }
        unbounded = TMLE(g_bounds=density_tests.NO_TRUNCATION, **settings).fit(data).single()
        bounded = TMLE(g_bounds=(0.3, 0.7), **settings).fit(data).single()
        curve = unbounded.diagnostics.truncation_curve(bounds=[0.3, 0.1])
    name = "ey_policy[set flag]"
    row = curve[(curve["bound"] == 0.3) & (curve["estimand"] == name)]
    assert float(row["psi"].iloc[0]) == pytest.approx(bounded.psi(name), abs=1e-9)
    assert abs(bounded.psi(name) - unbounded.psi(name)) > 1e-3


def test_the_placebo_permutes_whole_component_rows() -> None:
    """A row of the placebo is a row of the data: the joint law of ``(F, D)`` is kept."""
    data = point_tests.vector_data()
    components = np.asarray(data.treatment_components)
    permuted = _permuted_treatment(data, np.random.default_rng(3))
    assert permuted.shape == components.shape
    assert not np.array_equal(permuted, components)
    # The same multiset of rows, so each flag keeps the dose it was recorded with.
    assert sorted(map(tuple, permuted)) == sorted(map(tuple, components))
    replaced = data._with_components(permuted)
    np.testing.assert_array_equal(replaced.covariates, data.covariates)
    np.testing.assert_array_equal(replaced.outcome, data.outcome)
    # The control: a column-wise permutation keeps each marginal and breaks the pairing,
    # and the multiset check sees it, because F and D are dependent in this law.
    rng = np.random.default_rng(4)
    independent = np.column_stack([rng.permutation(components[:, j]) for j in range(2)])
    assert sorted(map(tuple, independent)) != sorted(map(tuple, components))


def test_a_replacement_that_drops_a_level_of_a_categorical_component_is_refused() -> None:
    data = point_tests.vector_data()
    values = np.asarray(data.treatment_components).copy()
    values[:, 0] = 0.0
    with pytest.raises(DataError, match="replacement component 'F' has codes"):
        data._with_components(values)


def test_the_placebo_refit_runs_on_a_policy_contrast() -> None:
    vector, _ = point_tests._sample_fits()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = vector.diagnostics.refute(
            tests=["placebo"], estimand="ate_policy[dose up vs natural]", n_replicates=2
        )
    assert report is not None


def test_the_missingness_tilt_names_the_vector() -> None:
    results, _ = _coarsened_fits(delta="Delta")
    with pytest.raises(CapabilityError, match="a vector treatment with a continuous component"):
        results.single().sensitivity.missingness()
