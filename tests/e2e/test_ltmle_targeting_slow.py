"""How much of the paired agreement with R ``ltmle`` the targeting step is responsible for.

The registered comparison hands both implementations the same rows, the same known mechanism
and the same quasibinomial regressions, and they agree to solver precision.  That is a strong
statement about the sequential regression, the follower masks and the influence curve -- and a
weaker one about the fluctuation than the passing counts suggest, because on this law the
update is small next to the standard error.

So the limitation is measured rather than asserted: build the plug-in the study would report
if it never targeted at all, and put it through the study's own two acceptance gates.  Three
of the five estimands survive them; the two that carry a dynamic rule do not.  A test rather
than a paragraph, because a paragraph cannot notice when the number changes.

Slow because it refits several hundred replications of the comparison law twice over, which is
the only way to compare two sampling distributions rather than two numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import (
    CONTRAST_NAMES,
    ESTIMANDS,
    SCENARIO,
    STUDY,
    draw_scenario,
    untargeted_estimands,
)
from tests.studies.evidence.inference import student_interval

pytestmark = pytest.mark.legacy_study

#: Enough to resolve a paired mean difference against a margin of ``0.15`` pooled standard
#: deviations, and far short of the study's own 1,600 -- this compares the untargeted plug-in
#: with committed rows rather than regenerating anything.
REPLICATES = 400

#: The estimands whose agreement with R survives dropping the targeting step entirely.  Named
#: rather than derived, so a change in which ones survive fails here instead of being absorbed.
SURVIVE_WITHOUT_TARGETING = (
    "ey_regimen[never]",
    "ey_regimen[always]",
    "ate_regimen[always vs never]",
)


def _replicate(replicate: int) -> dict[str, float]:
    frame, _ = draw_scenario(SCENARIO, STUDY.n, replicate)
    return untargeted_estimands(frame)


@pytest.fixture(scope="module")
def plug_in() -> pd.DataFrame:
    rows = map_parallel(_replicate, [(index,) for index in range(REPLICATES)], n_jobs=STUDY_JOBS)
    return pd.DataFrame(rows)


def test_the_static_estimands_do_not_witness_the_targeting_step(plug_in: pd.DataFrame) -> None:
    """An implementation that never targeted would pass the paired gate on three of five.

    Which is why the targeting claim rests on ``targeting_necessity`` and on the two dynamic
    estimands, and why the section says so.  The counts in the grid are not wrong; they are
    about agreement with a reference, and agreement is simply not sensitive here.
    """
    committed = pd.read_csv(STUDY.artifact("replicates.csv.gz"))
    reference = committed.loc[
        (committed["implementation"] == str(STUDY.reference))
        & (committed["replicate"] < REPLICATES)
    ]
    surviving, separating = [], []
    for name in ESTIMANDS:
        paired = reference.loc[reference["estimand"] == name].sort_values("replicate")
        assert len(paired) == REPLICATES
        subject = plug_in[name].to_numpy()
        difference = subject - paired["estimate"].to_numpy()
        pooled = math.sqrt(
            0.5 * (np.var(subject, ddof=1) + np.var(paired["estimate"].to_numpy(), ddof=1))
        )
        margin = STUDY.margins.paired_difference * pooled
        interval = student_interval(difference, confidence_level=STUDY.margins.confidence_level)
        (surviving if interval.within(-margin, margin) else separating).append(name)
    assert sorted(surviving) == sorted(SURVIVE_WITHOUT_TARGETING), (
        f"the paired gate now separates a different set: survives={surviving}, "
        f"separates={separating}"
    )
    # Both dynamic estimands separate, which is the half of the statement the section relies on.
    assert set(CONTRAST_NAMES) - set(SURVIVE_WITHOUT_TARGETING) <= set(separating)


def test_the_dynamic_estimands_do_witness_it(plug_in: pd.DataFrame) -> None:
    """The other half of the same statement, against truth rather than against the reference.

    Stated separately because the two gates are different questions -- this one would still
    hold if no comparator existed -- and because a single test asserting both would pass while
    half of it silently stopped being true.
    """
    _, truth = draw_scenario(SCENARIO, 50, 0)
    for name in ESTIMANDS:
        subject = plug_in[name].to_numpy()
        errors = subject - float(truth[name])
        margin = STUDY.margins.standardized_bias * float(np.std(subject, ddof=1))
        interval = student_interval(errors, confidence_level=STUDY.margins.confidence_level)
        inside = interval.within(-margin, margin)
        assert inside is (name in SURVIVE_WITHOUT_TARGETING), (
            f"{name}: untargeted plug-in bias interval {interval} against margin {margin}"
        )
