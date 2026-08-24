"""Structural gates for the two cross-fitted longitudinal evidence studies."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import Folds
from tests import discrete_law_longitudinal as end_law
from tests import discrete_law_survival as survival_law
from tests.studies import canonical_ltmle_crossfit as end_study
from tests.studies import canonical_ltmle_survival_crossfit as survival_study
from tests.studies import ltmle_crossfit_properties as end_properties
from tests.studies import ltmle_survival_crossfit_properties as survival_properties


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
    """The serialized fold is the fitted fold, not a reconstruction from the same seed."""
    frame, truth = study.draw_scenario(study.SCENARIO, 500, 0)
    result = study.fit_cleverly(frame)
    counts = np.bincount(result.folds.assignment, minlength=5)
    assert counts.max() - counts.min() <= 1

    sample, _, _ = study._replicate((study.SCENARIO, 0, 500))
    np.testing.assert_array_equal(sample["fold"].to_numpy(), result.folds.assignment)
    assert set(truth) == set(study.ESTIMANDS)
