"""The reviewed semantic callback for ``docs/examples/sampling-weights``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from matplotlib.text import Text
from scipy import integrate, stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, Targeting
from cleverly.datasets import biased_sampling_dgp, navigation_protocol, sampling_probability
from cleverly.datasets.longitudinal import WEIGHTED_SELECTION_PROBABILITIES
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    figure_title,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = (
    "sensitivity",
    "refutation",
)

NOTEBOOK = EXAMPLES / "sampling-weights.ipynb"

_PROBE = "reviews/notebook-review/probes/sampling-weights"

#: Each repeated-draw count a reading quotes, keyed by the probe column it counts.
_PROBE_COUNTS = {
    "w_covers": 476,
    "u_covers_all": 0,
    "u_covers_sample": 470,
    "m_covers": 481,
    "lu_always_covers": 55,
    "lu_never_covers": 29,
}

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "two-decisions-heading: 0.3": "the generator's keep chance where W1 > 0, "
    "cleverly.datasets.longitudinal.WEIGHTED_SELECTION_PROBABILITIES",
    "two-decisions-heading: 0.9": "the generator's keep chance elsewhere, "
    "cleverly.datasets.longitudinal.WEIGHTED_SELECTION_PROBABILITIES",
    "0.993": f"spread of the weighted estimates over their mean standard error, 500 draws: "
    f"{_PROBE}/summary.log",
}

#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {
    "plan": {"0.15": "data", "0.90": "data"},
    "failure-reading": {"1.000": "data", "1.487": "data"},
    "assessment-reading": {"0.15": "data"},
}


def _quadrature(rows: Any = None, *, interviewed: bool = False) -> float:
    """An effect of the law by adaptive quadrature over discharge risk, independent of the grid.

    The effect at risk ``w`` is ``1 + 1.2 w``; ``rows`` limits the risk to ``(low, high)``, and
    ``interviewed`` weights each risk by its interview chance.
    """
    low, high = rows if rows is not None else (-np.inf, np.inf)

    def mass(w: float) -> float:
        chance = float(sampling_probability(np.array([w]))[0]) if interviewed else 1.0
        return float(stats.norm.pdf(w)) * chance

    top = integrate.quad(lambda w: (1 + 1.2 * w) * mass(w), low, high, limit=200)[0]
    bottom = integrate.quad(mass, low, high, limit=200)[0]
    return top / bottom


def _probe() -> list[dict[str, str]]:
    with (EXAMPLES.parents[1] / _PROBE / "sweep.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(namespace: dict[str, Any]) -> None:
    """Witness the law, the weighted and unweighted targets, the groups, and the lessons."""
    law = biased_sampling_dgp()
    truth = namespace["truth"]
    frame = namespace["frame"]

    # The law: the effect at a covariate profile is 1 + 1.2 * discharge_risk and does not read
    # age; the interview chance reads discharge risk only.  These are the graph's arrows.
    w = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    effect = law.outcome_mean(w, 1.0, None) - law.outcome_mean(w, 0.0, None)
    assert np.allclose(effect, [1.0, 2.2, 1.0])
    assert float(law.propensity(w)[2]) != float(law.propensity(w)[0])  # age drives the offer
    assert float(law.propensity(w)[1]) != float(law.propensity(w)[0])  # risk drives the offer

    # The notebook's grid gives the exact targets that an independent quadrature gives.
    true_effect, density, interviewed, risk = (
        namespace["true_effect"],
        namespace["density"],
        namespace["interviewed"],
        namespace["risk"],
    )
    assert true_effect(density) == pytest.approx(truth["ate"], abs=1e-6)
    assert true_effect(interviewed) == pytest.approx(_quadrature(interviewed=True), abs=1e-6)
    groups = {
        "high": (risk > 0, (0.0, np.inf)),
        "low": (risk <= 0, (-np.inf, 0.0)),
    }
    # A group's edge at 0 falls on a grid point, which costs about one grid step (1e-4) of the
    # group's mass, far below the printed third decimal.
    for side, bounds in groups.values():
        assert true_effect(density, side) == pytest.approx(_quadrature(bounds), abs=1e-4)
        assert true_effect(interviewed, side) == pytest.approx(
            _quadrature(bounds, interviewed=True), abs=1e-4
        )
    assert f"{true_effect(interviewed):.3f}" == "1.487"

    # The data step: only interviewed rows, with chances in [0.15, 0.90] read off discharge risk.
    assert set(frame.columns) == {
        "function_score",
        "transition_navigation",
        "discharge_risk",
        "age",
        "sampling_prob",
        "sampling_weight",
        "high_risk",
    }
    assert np.allclose(frame["sampling_prob"], sampling_probability(frame["discharge_risk"]))
    assert np.allclose(frame["sampling_weight"] * frame["sampling_prob"], 1.0)
    assert frame["sampling_prob"].min() >= 0.15 and frame["sampling_prob"].max() <= 0.90
    assert abs(frame["sampling_weight"].sum() - 8_000) < 0.02 * 8_000

    # The protocol changes exactly the four fields the heading names.
    program = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, program) == {
        "target_population",
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert all("transition score" not in rule for rule in protocol.intercurrent_event_handling)
    assert "discharge risk" in protocol.assumption_rationale[1]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, namespace["weighted"])

    # The estimate step: the declared models and settings.
    method = namespace["method"]
    assert isinstance(method.models.outcome_learner, LinearRegression)
    assert isinstance(method.models.treatment_learner, LogisticRegression)
    assert method.cross_fitting == CrossFitting(enabled=False)
    assert method.targeting.fluctuation == "linear"
    assert namespace["study"].design.weights == "sampling_weight"
    assert "sampling_prob" not in namespace["study"].design.adjustment

    # The weighted fit covers the population effect and reads supported; the unweighted fit,
    # with the default correction step, covers the interviewed-population effect only.
    weighted = namespace["weighted"]["ate"]
    unweighted = namespace["unweighted"]["ate"]
    assert namespace["plain_method"].targeting == Targeting()
    assert covers(weighted, 1.0) and not covers(weighted, true_effect(interviewed))
    assert covers(unweighted, true_effect(interviewed)) and not covers(unweighted, 1.0)
    assert weighted.support.label == "supported"
    assert unweighted.support.label == "supported"
    # The nonzero witness that the weights enter: the two estimates differ by many combined
    # standard errors, which they would not if the weights were dropped.
    joint = np.hypot(weighted.std_error, unweighted.std_error)
    assert (unweighted.psi - weighted.psi) / joint > 5.0
    assert figure_title(namespace["comparison_fig"]) == "Estimates and intervals"
    # "the two correction steps give the same estimate to three decimals": the unweighted fit
    # with the linear correction step of the weighted fit.
    linear = (
        namespace["CausalStudy"](frame, design=namespace["unweighted_design"])
        .identify(namespace["ATE"](reference=0))
        .estimate(method=replace(namespace["plain_method"], targeting=method.targeting))
    )["ate"]
    assert f"{linear.psi:.3f}" == f"{unweighted.psi:.3f}"

    # The groups: every row is experimental, each interval covers its own fit's target, and the
    # unweighted rows sit above the weighted ones in both groups.
    fits = namespace["fits"]
    for group, (side, _) in zip((1.0, 0.0), groups.values(), strict=True):
        key = f"ate[high_risk={group}]"
        assert covers(fits["weighted"][key], true_effect(density, side))
        assert covers(fits["unweighted"][key], true_effect(interviewed, side))
        assert fits["unweighted"][key].psi > fits["weighted"][key].psi
        for fit in fits.values():
            assert fit[key].support.label == "experimental"
    assert fits["weighted"]["ate"].support.label == "experimental"

    # The overlap check: nothing needs attention, nobody is at the caps.
    assert tuple(namespace["assessment"].attention) == ()
    assert namespace["support"].truncated["fraction"] == 0.0
    assert namespace["weights_report"].effective_n < len(frame)

    # The missing-offer lesson: about a quarter of records hidden, experimental, covers 1.0.
    patchy = namespace["patchy"]
    assert 0.65 < float(patchy["offer_recorded"].mean()) < 0.80
    assert patchy["transition_navigation"].isna().sum() == (patchy["offer_recorded"] == 0).sum()
    missing = namespace["patchy_fit"]["ate"]
    assert covers(missing, 1.0) and missing.support.label == "experimental"

    # The two-decision lesson keeps each patient with chance 0.3 or 0.9, as the heading says.
    assert WEIGHTED_SELECTION_PROBABILITIES == (0.3, 0.9)
    plans_frame = namespace["plans_frame"]
    assert set(np.round(1 / plans_frame["w"], 6)) == {0.3, 0.9}

    # The two-decision lesson: weighted intervals cover both plans, unweighted miss both, and
    # every row is supported.
    rows = namespace["plan_rows"]
    assert [(row["fit"], row["plan"], row["covers"]) for row in rows] == [
        ("weighted", "always", True),
        ("weighted", "never", True),
        ("unweighted", "always", False),
        ("unweighted", "never", False),
    ]
    assert all(row["support"] == "supported" for row in rows)
    assert all(row["estimate"] < row["true"] for row in rows if row["fit"] == "unweighted")

    # The review probe: each quoted count recomputed from its committed rows.
    probe = _probe()
    assert len(probe) == 500
    for column, count in _PROBE_COUNTS.items():
        assert sum(row[column] == "True" for row in probe) == count, column
    summary = (EXAMPLES.parents[1] / _PROBE / "summary.log").read_text(encoding="utf-8")
    assert "SD/SE 0.993" in summary


def check_refits(namespace: dict[str, Any]) -> None:
    """The refit cells: the omitted-variable bound with benchmarks, and the refutation."""
    weighted = namespace["weighted"]["ate"]
    rv = namespace["rv"]
    assert 0.0 < rv["rva"] < rv["rv"] < 1.0
    benchmarks = namespace["benchmarks"]
    risk_record, risk_bound = benchmarks["discharge_risk"]
    _, age_bound = benchmarks["age"]
    # "sits at the top of the score side": the benchmark clips to 1.
    assert risk_record.cf_y == 1.0
    assert risk_bound.lower > 0.0 and age_bound.lower > risk_bound.lower
    # The tipping weight-side strength exceeds the discharge-risk benchmark's, and the range of
    # the estimate (the point bound, not a confidence limit) crosses zero there.
    tipping = namespace["tipping_cf_d"]
    sensitivity = namespace["sensitivity"]
    assert tipping > risk_record.cf_d
    assert sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping - 0.01).lower > 0.0
    assert sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping + 0.01).lower <= 0.0
    assert f"{tipping:.2f}" == "0.15"

    # The reading's figure: the grid reaches both benchmarks and draws the zero line.
    figure = namespace["confounding_fig"]
    gids = {artist.get_gid() for artist in figure.findobj() if hasattr(artist, "get_gid")}
    assert "cleverly-null-contour" in gids
    texts = {text.get_text() for text in figure.findobj(Text)}
    assert {"discharge_risk", "age"} <= texts

    refutation = namespace["refutation"]
    tests = {test.name: test for test in refutation.tests}
    assert set(tests) == {"placebo", "subset", "random_common_cause"}
    for test in tests.values():
        assert test.passed is None
        assert test.original == pytest.approx(weighted.psi)
    # The reading's reasons the verdicts are withheld: no study covers the noise column, and the
    # studies of the other two do not cover an in-sample fit with these learners.
    assert tests["random_common_cause"].evidence == "No study covers this test."
    for name in ("placebo", "subset"):
        assert "no out-of-fold predictions" in tests[name].evidence, tests[name].evidence
    assert abs(float(np.mean(tests["placebo"].values))) < 3 * weighted.std_error
    # "the noise column does not move it"
    noise = tests["random_common_cause"]
    assert abs(float(np.mean(noise.values)) - weighted.psi) < 0.1 * weighted.std_error
    # "the 70% refits spread about as much as the standard error predicts"
    subset = tests["subset"]
    expected = float(subset.expectation.rsplit("~", 1)[1])
    assert 0.5 < float(np.std(subset.values, ddof=1)) / expected < 1.5
