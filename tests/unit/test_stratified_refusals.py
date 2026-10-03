"""Two baseline-strata refusals that no other test pins, each before any learner fits.

The other strata refusals each have a pre-fit test already:

=====================================================  =======================================
refusal                                                pinning test
=====================================================  =======================================
``cv_evaluation=True`` or ``targeting_scheme="fold"``  ``test_natural_course_crossfit.py``,
                                                       ``test_a_complete_outcome_fit_with_
                                                       strata_meets_the_generic_strata_gate``
an incremental target, a non-identity MSM link, a      ``test_refusals_before_the_nuisance_fit.py``
continuous-dose MSM, or a ``DRTMLE`` guard
the stratum alias in a sensitivity analysis            ``test_stratified_targets.py``
=====================================================  =======================================

The two below are the missing-outcome contracts' own strata refusals.
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


@pytest.mark.parametrize("cross_fit", [False, True], ids=("in-sample", "cross-fitted"))
def test_the_natural_course_contract_refuses_strata(cross_fit: bool) -> None:
    with pytest.raises(CapabilityError) as caught:
        _fit(cross_fit=cross_fit, estimands=("ey_obs",))
    assert str(caught.value) == (
        "NaturalCourseMean, PAR and PAF with missing outcomes currently support ordinary TMLE "
        "under their audited implementation contracts; baseline strata need a "
        "stratum-indexed natural-course fluctuation (X8 in docs/roadmap.md)"
    )
    assert NeverFit.calls == 0
