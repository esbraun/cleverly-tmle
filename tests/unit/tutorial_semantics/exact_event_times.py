"""The reviewed semantic callback for ``docs/examples/exact-event-times``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`, and the slow tier adds :func:`check_refits`.
Each assertion pins a reading the prose makes about this draw.  A relation on one draw certifies
no method.  The true values come from :func:`cleverly.datasets.exact_readmission_truth`, whose
own test (``tests/unit/test_datasets_exact_times.py``) checks them against a Monte Carlo draw of
the potential event times.  The repeated-draw numbers the readings quote come from the page's
probe, ``reviews/notebook-review/probes/exact-event-times``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleverly.datasets import (
    exact_readmission_protocol,
    exact_readmission_truth,
    longitudinal_navigation_protocol,
    make_exact_readmission,
)
from cleverly.survival.result import continuous_bootstrap_support
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refutation",)

NOTEBOOK = EXAMPLES / "exact-event-times.ipynb"

_PROBE = ROOT / "reviews/notebook-review/probes/exact-event-times"
_SUMMARY = "reviews/notebook-review/probes/exact-event-times/summary.log"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.0338": f"mean SE of the month-3 readmission difference, 100 draws: {_SUMMARY}",
    "0.0327": f"SD of the month-3 readmission difference, 100 draws: {_SUMMARY}",
}

#: The lines of the probe summary the readings quote.
_QUOTED = (
    "two ate_regimen[offered vs not offered, readmission @ t=3]: truth -0.2125, bias 0.0007",
    "emp SD 0.0327, mean SE 0.0338, SE/SD 1.04",
    "one risk_regimen[offered @ t=3]: truth 0.2357",
    "coverage 0.960 (MC SE 0.020)",
)

#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it.
NARRATION_SOURCES: dict[str, dict[str, str]] = {}

READMISSION_3 = "ate_regimen[offered vs not offered, readmission @ t=3]"
DEATH_3 = "ate_regimen[offered vs not offered, death @ t=3]"


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    fresh, fresh_truth = make_exact_readmission(n=1_000, seed=2026)
    pd.testing.assert_frame_equal(frame, fresh, check_exact=True)
    assert truth == fresh_truth == exact_readmission_truth((1.5, 3.0), restricted_horizons=(3.0,))
    assert list(frame.columns) == ["time", "event", "offered", "risk_score"]


def _check_association(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    means = frame.groupby("offered")["risk_score"].mean()
    # "Offered patients had a higher risk score".
    assert means[1] > means[0]
    simple = {}
    for code, cause in ((1, "readmission"), (2, "death")):
        recorded = frame["event"].eq(code) & frame["time"].le(3.0)
        shares = recorded.groupby(frame["offered"]).mean()
        simple[cause] = shares[1] - shares[0]
    # "far smaller in size than the true -0.2125" and "larger than the true -0.0586".
    assert truth[READMISSION_3] < simple["readmission"] < 0.0
    assert simple["death"] < truth[DEATH_3] < 0.0


def _check_fit(namespace: dict[str, Any]) -> None:
    protocol, result, truth = namespace["protocol"], namespace["result"], namespace["truth"]
    assert protocol == exact_readmission_protocol()
    assert changed_fields(protocol, longitudinal_navigation_protocol()) == {
        "time_zero",
        "treatment_strategies",
        "treatment_versions",
        "outcome",
        "horizon",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, result)
    table = result.to_frame().set_index("estimand")
    # "Each label reads supported" and "each interval contains its value", for every printed row.
    printed = [name for name in table.index if name.startswith(("cif_regimen", "ate_regimen"))]
    assert len(printed) == 12
    for name in printed:
        assert table.loc[name, "support"] == "supported"
        assert covers(result[name].ci, truth[name])
    # The band "reads experimental".
    assert result.simultaneous.support.label == "experimental"
    # "The data cannot tell whether it changes death by month 3": the interval holds 0 and the
    # true fall alike.
    death = result[DEATH_3]
    assert covers(death.ci, 0.0) and covers(death.ci, truth[DEATH_3]) and truth[DEATH_3] < -0.05
    # The restricted means and the ratio.
    for name in ("gained", "saved", "ratio"):
        estimate = namespace[name]
        assert estimate.support.label == "supported"
        assert covers(estimate.ci, truth[estimate.name])


def _check_assessment(namespace: dict[str, Any]) -> None:
    checks = namespace["checks"].set_index("check")["status"]
    assert checks["score_equations"] == "passed"
    for name in ("omitted_confounding", "marginal_bounds", "missingness"):
        assert checks[name] == "unavailable"
    support = namespace["support"]
    assert support.bounded_share == 0.0


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    evalue = namespace["evalue"]
    assert evalue.point > evalue.limit > 1.0
    assert evalue.limit_descriptive


def _check_deaths(namespace: dict[str, Any]) -> None:
    one, result = namespace["one_event_result"], namespace["result"]
    death_free, truth = namespace["death_free"], namespace["truth"]
    assert death_free == exact_readmission_truth((1.5, 3.0), death_prevented=True)
    for plan in ("offered", "not offered"):
        shortcut = one[f"risk_regimen[{plan} @ t=3]"]
        both = result[f"cif_regimen[{plan}, readmission @ t=3]"]
        # "The shortcut is larger".
        assert shortcut.psi > both.psi
        assert (
            death_free[f"risk_regimen[{plan} @ t=3]"]
            > truth[f"cif_regimen[{plan}, readmission @ t=3]"]
        )
    # "just misses its true 0.2357"; the not-offered interval contains its value.
    assert not covers(
        one["risk_regimen[offered @ t=3]"].ci, death_free["risk_regimen[offered @ t=3]"]
    )
    assert covers(
        one["risk_regimen[not offered @ t=3]"].ci, death_free["risk_regimen[not offered @ t=3]"]
    )


def _check_hook() -> None:
    # "it reads experimental, and its studies have not run yet": when an X6 bootstrap study
    # registers and a label changes, this fails and the page gains its bootstrap lesson.
    for cross_fitted in (False, True):
        support = continuous_bootstrap_support(kind="bootstrap", cross_fitted=cross_fitted)
        assert support.label == "experimental"


def _check_probe() -> None:
    summary = (_PROBE / "summary.log").read_text()
    for line in _QUOTED:
        assert line in summary, line


def check(namespace: dict[str, Any]) -> None:
    _check_data(namespace)
    _check_association(namespace)
    _check_fit(namespace)
    _check_assessment(namespace)
    _check_sensitivity(namespace)
    _check_deaths(namespace)
    _check_hook()
    _check_probe()


def check_refits(namespace: dict[str, Any]) -> None:
    frame = namespace["refutation"].to_frame().set_index("test")
    assert frame.loc["placebo", "verdict"] == "passed"
    assert frame.loc["subset", "verdict"] == "failed"
    # "the refits spread less than the standard error predicts".
    assert frame.loc["subset", "statistic"] < 1.0
