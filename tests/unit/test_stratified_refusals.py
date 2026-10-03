"""Two baseline-strata refusals that no other test pins, each before any learner fits.

The other strata refusals each have a pre-fit test already:

=====================================================  =======================================
refusal                                                pinning test
=====================================================  =======================================
``cv_evaluation=True`` or ``targeting_scheme="fold"``  ``test_natural_course_crossfit.py`` and
                                                       ``test_refusals_before_the_nuisance_
                                                       fit.py``
a DR-TMLE stratum with no trainable rows               ``test_refusals_before_the_nuisance_
                                                       fit.py``
an incremental stratum without a treatment arm         ``test_stratified_incremental_exact.py``
a working model singular inside a stratum              ``test_stratified_msm_exact.py``
the stratum alias in a sensitivity analysis            ``test_stratified_targets.py``
=====================================================  =======================================

The two below are the missing-outcome contracts' own strata refusals.  The in-sample
natural-course mean fits strata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly import CapabilityError
from cleverly.estimators import TMLE
from tests import discrete_law_mar as law
from tests.unit._natural_course_support import NeverFit, never_fit_learners


def _frame() -> pd.DataFrame:
    return law.frame().assign(S=(np.arange(law.N) % 2).astype(float))


def _fit(**settings: object) -> None:
    NeverFit.calls = 0
    TMLE(**settings, **never_fit_learners(), random_state=0).fit(
        _frame(),
        outcome="Y",
        treatment="A",
        covariates=("W", "S"),
        delta="Delta",
        strata="S",
    )


def test_the_cross_fitted_arm_indexed_contract_refuses_strata() -> None:
    with pytest.raises(CapabilityError) as caught:
        _fit(cross_fit=True, estimands=("ate",))
    assert str(caught.value).endswith(
        "no audited result covers baseline strata. Drop strata= from fit "
        "(PointTreatment(strata=()))"
    )
    assert NeverFit.calls == 0


def test_the_cross_fitted_natural_course_fits_strata() -> None:
    with pytest.raises(AssertionError, match="before any learner is fitted"):
        _fit(cross_fit=True, estimands=("ey_obs",))
    assert NeverFit.calls > 0


def test_the_in_sample_natural_course_fits_strata() -> None:
    with pytest.raises(AssertionError, match="before any learner is fitted"):
        _fit(cross_fit=False, estimands=("ey_obs",))
    assert NeverFit.calls > 0
