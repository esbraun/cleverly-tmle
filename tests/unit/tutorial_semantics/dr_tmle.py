"""The reviewed semantic callback for ``docs/examples/dr-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

from typing import Any


def check(namespace: dict[str, Any]) -> None:
    """The DR-TMLE tutorial's seeded claims hold at the documented size.

    The page claims an exact reduction to the ordinary estimator, a close estimate and standard
    error on this draw, passing score and correction reports with no active truncation, and a
    reduced-regression table whose g_r1 fits all favor the spline candidate.
    """
    ordinary = namespace["ordinary"]["ate"]
    empty_guard = namespace["empty_guard"]["ate"]
    guarded = namespace["guarded"]["ate"]
    assert ordinary.psi == empty_guard.psi

    # "moves by less than one standard error, and the standard errors are similar". The lower
    # bound witnesses "the guarded estimate moves": about 0.4 SE on this draw.
    shift = abs(guarded.psi - ordinary.psi) / ordinary.std_error
    assert 0.1 < shift < 0.9
    assert 0.8 < guarded.std_error / ordinary.std_error < 1.25

    assessment = namespace["assessment"]
    corrections = assessment.report("corrections")
    assert corrections.passed
    assert corrections.contract == "theorem"  # "no truncation was active"
    assert assessment.report("score_equations").passed
    summary = assessment.summary()
    assert "omitted_confounding" in summary

    nuisance = assessment.report("nuisance_models")
    assert "look reasonable" in nuisance.summary()

    reduced = namespace["reduced"]
    assert set(reduced) == {"qr", "gr1", "gr2"}
    # Fails on a revert to plain linear reducers, which leave these diagnostics empty.
    assert reduced["gr1"] and all(fit.best == "spline" for fit in reduced["gr1"])
