"""The reviewed semantic callback for ``docs/examples/cross-fitting``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

from typing import Any


def check(namespace: dict[str, Any]) -> None:
    """The seeded relations docs/examples/cross-fitting.md narrates, at its documented size."""
    cross_fitted = namespace["cross_fitted"]["ate"]
    in_sample = namespace["in_sample"]["ate"]

    assert "stacked CV-TMLE" in namespace["cross_fitted"].summary()
    # "the two point estimates are close"; "less than a third" (measured ratio about 0.28).
    assert abs(in_sample.psi - cross_fitted.psi) < 0.1
    assert in_sample.std_error < cross_fitted.std_error / 3

    # "nearly identical" points and "about 1.7 times larger" standard error.
    ignoring = namespace["ignoring"]["ate"]
    clustered = namespace["clustered"]["ate"]
    assert abs(ignoring.psi - clustered.psi) < 0.02
    assert 1.5 < clustered.std_error / ignoring.std_error < 1.9
    assert namespace["clustered"].data.n_clusters == 200

    # "warns that the out-of-fold propensity model is poorly calibrated" (slope about 0.49,
    # warning below 0.7); "about 1%" truncated; "about 25%"; in-sample "above 85%".
    diagnostics = namespace["diagnostics"]
    assert diagnostics["nuisance_models"].status.value == "warning"
    nuisance = diagnostics.report("nuisance_models")
    assert nuisance["propensity"].metrics["calibration_slope"] < 0.6
    support = diagnostics.report("support")
    assert 0.005 < support.truncated["fraction"] < 0.02
    ratios = [ess["ratio"] for ess in support.effective_sample_size.values()]
    assert 0.2 < min(ratios) < 0.3
    in_sample_support = namespace["in_sample"].diagnostics.support()
    in_sample_ratios = [ess["ratio"] for ess in in_sample_support.effective_sample_size.values()]
    assert min(in_sample_ratios) > 0.85

    spread = namespace["repeated_nuisance"].repeat_spread
    assert [row.n_repeats for row in spread] == [3]
