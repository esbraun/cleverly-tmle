"""Focused contracts for the weighted point-treatment evidence study."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_weighted_tmle as study
from tests.studies import weighted_point_common as law
from tests.studies import weighted_tmle_properties as properties
from tests.studies.point_study_helpers import initial_estimates


def _stub_result(zero: Any, one: Any, weights: Any) -> Any:
    """The three attributes :func:`initial_estimates` reads, and nothing else."""
    arms = {0.0: np.asarray(zero, dtype=float), 1.0: np.asarray(one, dtype=float)}
    return SimpleNamespace(
        data=SimpleNamespace(weights=np.asarray(weights, dtype=float), arm_codes=(0.0, 1.0)),
        nuisance=SimpleNamespace(
            scaler=SimpleNamespace(unscale_levels=lambda values: values),
            outcome=SimpleNamespace(arms=arms),
        ),
    )


@pytest.mark.parametrize(
    ("zero", "one", "weights"),
    [
        ([0.2, 0.3], [0.4, 1.0], [1.0, 1.0]),
        ([0.0, 0.3], [0.4, 0.5], [1.0, 1.0]),
        ([0.2, np.nan], [0.4, 0.5], [1.0, 1.0]),
        ([0.2, 0.3], [0.4, 0.5], [1.0, 0.0]),
    ],
)
def test_the_ratio_transcription_refuses_what_the_r_adapter_refuses(
    zero: Any, one: Any, weights: Any
) -> None:
    """``tmle_point_adapter.R`` refuses these rows before it forms the same two ratios.

    It requires finite predictions strictly inside the unit interval and finite, strictly
    positive weights.  The Python side divided by ``ey0`` and by ``1 - ey1`` with no guard,
    so the same row that stops the R runner would have published an infinity.
    """
    with pytest.raises(AssertionError):
        initial_estimates(_stub_result(zero, one, weights), ("ey0", "ey1", "ate", "rr", "or"))


def test_the_ratio_guard_does_not_reach_a_study_that_reports_no_ratio() -> None:
    """A continuous outcome has arm means outside the unit interval and reports no ratio.

    The guard, and the two ratios themselves, belong to the estimand list rather than to
    the helper.  The clustered study is exactly this case: it publishes ``ey0``, ``ey1`` and
    ``ate`` on a continuous outcome, and the shared transcription used to divide by
    ``1 - ey1`` for it anyway.
    """
    initials = initial_estimates(_stub_result([2.0, 3.0], [5.0, 7.0], [1.0, 1.0]), ("ate",))
    assert set(initials) == {"ey0", "ey1", "ate"}
    assert initials["ate"] == pytest.approx(3.5)


#: Each property family, and the sample size and replication budget its cells were run at.
#: ``root_n_rate`` is derived from the three ``root_n_and_efficiency`` cells rather than run,
#: so it declares no budget of its own and is absent here.
DECLARED_BUDGETS: dict[str, tuple[int, int]] = {
    "double_robustness": (properties.DOUBLE_ROBUST_N, properties.DOUBLE_ROBUST_REPLICATES),
    "interval_calibration": (properties.CALIBRATION_N, properties.CALIBRATION_REPLICATES),
    "type_i_error": (properties.NULL_N, properties.NULL_REPLICATES),
    "power": (properties.POWER_N, properties.POWER_REPLICATES),
    "targeting_necessity": (properties.NECESSITY_N, properties.NECESSITY_REPLICATES),
    "weight_necessity": (properties.NECESSITY_N, properties.NECESSITY_REPLICATES),
}


def test_the_declared_budgets_are_the_ones_the_committed_cells_were_run_at() -> None:
    """Each budget constant against the artifact it sized, not against its own literal.

    This test used to assert eleven ``CONSTANT == literal`` pairs, which can only fail on
    a deliberate edit and cannot fail on a stale one.  Every margin here is already gated
    against the document's measured-values table by
    ``test_method_evidence.py::TestTheQuantityVocabulary``, and ``replicates`` and ``n``
    are gated against the manifest.  What nothing gated is the replication budget: moving
    ``CALIBRATION_REPLICATES`` without regenerating left the module declaring one run and
    the artifacts recording another, and the eleven literals would have moved with it.
    """
    published = pd.read_csv(study.STUDY.artifact("properties.csv"))
    for family, (n, replicates) in DECLARED_BUDGETS.items():
        cells = published.loc[published["property"] == family]
        assert not cells.empty, f"the committed properties carry no {family} cell"
        assert set(cells["n"]) == {n}, f"{family} was run at {sorted(set(cells['n']))}, not {n}"
        assert set(cells["replicates"]) == {replicates}, (
            f"{family} was run at {sorted(set(cells['replicates']))}, not {replicates}"
        )

    rate = published.loc[published["property"] == "root_n_and_efficiency"]
    assert set(rate["n"]) == set(properties.RATE_SIZES)
    assert set(rate["replicates"]) == {properties.RATE_REPLICATES}


def test_inverse_selection_weights_recover_the_population_law_exactly() -> None:
    tilted = law.SELECTED_P_W * law.OBSERVATION_WEIGHTS
    tilted /= tilted.sum()
    np.testing.assert_allclose(tilted, law.P_W, rtol=0, atol=1e-15)
    population = law.truth_for(law.Q, law.P_W)["ate"]
    selected = law.truth_for(law.Q, law.SELECTED_P_W)["ate"]
    assert population == pytest.approx(0.33)
    assert selected == pytest.approx(0.5222222222222223)
    assert abs(population - selected) > 0.15


def test_the_sampler_draws_the_requested_size_directly_from_the_selected_law() -> None:
    frame = law.sample_selected(law.Q, 731, 19)
    assert len(frame) == 731
    levels = frame["W"].to_numpy(dtype=int)
    np.testing.assert_allclose(frame["obs_weight"], law.OBSERVATION_WEIGHTS[levels])
    assert set(frame.columns) == {"Y", "A", "W", "obs_weight"}


#: The observation weight of each support point, which depends on ``W`` alone.
CELL_WEIGHTS = law.OBSERVATION_WEIGHTS[:, None, None]


def _parameters(probs: Any) -> dict[str, Any]:
    """Every published parameter of a finite point law, written out from cell masses.

    Kept analytic in ``probs`` -- ratios and logarithms of linear functions -- so
    :func:`_weighted_eif` can differentiate through it by a complex step.  Nothing here
    reads :mod:`tests.studies.weighted_point_common`: this is the identification formula
    written a second time, which is what makes the comparison a check rather than an echo.
    """
    p_w = probs.sum(axis=(1, 2))
    q = probs[:, :, 1] / probs.sum(axis=2)
    ey0, ey1 = p_w @ q
    return {
        "ate": ey1 - ey0,
        "logrr": np.log(ey1) - np.log(ey0),
        "logor": np.log(ey1 / (1.0 - ey1)) - np.log(ey0 / (1.0 - ey0)),
    }


def _weighted_eif(name: str, *, step: float = 1e-30) -> np.ndarray:
    r"""The influence curve of :math:`P \mapsto \Psi(P_w)` at every selected-law point.

    The contamination is of the *selected* law, the one the rows are drawn from, and the
    parameter is read off the tilted law :math:`dP_w = w\,dP / E_P[w]`.  That is the whole
    content of the check.  The weights belong to the sampling experiment, so the outer
    density ratio has to fall out of differentiating through the normalisation rather than
    being written in by hand, and an outer ratio of the wrong scale cannot survive it.

    Differentiation is by complex step.  The parameter is a rational function of the
    contamination weight, hence analytic, so the imaginary part carries the derivative to
    full double precision with no subtractive cancellation.
    """
    base = law.selected_probabilities().astype(complex)
    curve = np.empty(len(law.SUPPORT))
    for index, point in enumerate(law.SUPPORT):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        tilted = perturbed * CELL_WEIGHTS
        curve[index] = float(np.imag(_parameters(tilted / tilted.sum())[name]) / step)
    return curve


@pytest.mark.parametrize(
    ("name", "curve", "deviation"),
    [
        ("ate", law.weighted_ate_eif, law.weighted_ate_efficiency_sd),
        ("logrr", law.weighted_logrr_eif, law.weighted_logrr_efficiency_sd),
        ("logor", law.weighted_logor_eif, law.weighted_logor_efficiency_sd),
    ],
)
def test_each_weighted_curve_is_the_gateaux_derivative_of_the_tilted_parameter(
    name: str, curve: Any, deviation: Any
) -> None:
    """The scale witness the two centring checks could not supply.

    A mean-zero check is blind to the outer density ratio's scale: writing
    ``1 / SELECTION[w]`` where the curve needs ``SELECTION_RATE / SELECTION[w]`` multiplies
    every value by 2.469 and leaves the curve mean zero.  Comparing the module's own second
    moment against ``sqrt(p @ curve**2)`` restates its body line for line and cannot fail
    at all.  The derivative below is taken from the identification formula alone, so it
    fixes the scale as well as the shape.

    The two ratio curves get the same treatment, because they are the bounds that decide
    the study's one standard-error disagreement, and a delta-method factor written by hand
    is exactly the kind of term an exact-law check cannot see.
    """
    derived = _weighted_eif(name)
    np.testing.assert_allclose(derived, curve(), rtol=0, atol=1e-12)
    probabilities = law.selected_probabilities().reshape(-1)
    assert float(probabilities @ derived) == pytest.approx(0.0, abs=1e-14)
    assert deviation() == pytest.approx(
        float(np.sqrt(probabilities @ np.square(derived))), rel=1e-12
    )


def test_dropping_the_selection_rate_leaves_the_bound_outside_the_declared_band() -> None:
    """The deliberate mutation the derivative above rules out, priced in the study's band.

    ``SELECTION_RATE`` is what makes the outer ratio a density ratio rather than a raw
    inverse probability.  Omitting it is the "normalize by ``sum(w)`` here and ``n`` there"
    defect, and the cell that would have to catch it is ``interval_calibration``, whose
    empirical efficiency ratio compares the committed sampling spread against the bound.
    """
    published = pd.read_csv(study.STUDY.artifact("properties.csv")).set_index(["property", "cell"])
    cell = published.loc[("interval_calibration", "ate__correctly_specified")]
    empirical_se = float(cell["empirical_se"])
    n = int(cell["n"])

    honest = law.weighted_ate_efficiency_sd() / np.sqrt(n)
    mutated = honest / law.SELECTION_RATE
    low, high = properties.EFFICIENCY_RATIO_BAND
    assert low <= empirical_se / honest <= high
    assert not low <= empirical_se / mutated <= high


#: How far a mean reported standard error may sit from the exact bound, as a share of it.
#: Wide enough for 800 replications of Monte Carlo noise and the finite-sample term, and
#: eight times narrower than the 4% departure the two implementations differ by on ``or``.
BOUND_TOLERANCE = 0.005

#: R ``tmle`` 2.1.1's mean marginal log-odds-ratio standard error, as a share of the exact
#: bound.  Measured from the committed replications, not declared: the package reports 1.8%
#: less than the efficient influence curve of the tilted law carries.
R_LOGOR_BOUND_SHARE = 0.9818
R_LOGOR_BOUND_SHARE_TOLERANCE = 0.0005


def _scaled_standard_errors() -> pd.DataFrame:
    """Each committed mean standard error, multiplied by the square root of the size."""
    summary = pd.read_csv(study.STUDY.artifact("summary.csv"))
    summary["bound_share"] = summary.apply(
        lambda row: (
            row["mean_std_error"]
            * np.sqrt(row["n"])
            / study.STUDY.efficiency_bounds[row["estimand"]]
            if row["estimand"] in study.STUDY.efficiency_bounds
            else np.nan
        ),
        axis=1,
    )
    return summary.set_index(["implementation", "estimand"])


def test_cleverly_reports_the_exact_efficiency_bound_for_all_three_contrasts() -> None:
    """The bound decides the study's one standard-error disagreement.

    The two implementations agree to about 1e-11 on every point estimate and on the ``ate``,
    ``ey0``, ``ey1`` and ``rr`` standard errors.  They disagree on ``or``, and the paired
    comparison cannot say which is right: it asks whether two implementations are similar,
    and both are inside the shared calibration margin.  The exact efficient influence curve
    of the tilted law can say, so this reads it off the committed summary.
    """
    shares = _scaled_standard_errors()
    for estimand in study.STUDY.efficiency_bounds:
        share = float(shares.loc[(study.STUDY.implementation, estimand), "bound_share"])
        assert share == pytest.approx(1.0, abs=BOUND_TOLERANCE), (
            f"{estimand} reports {share:.6f} of its exact bound"
        )


def test_the_reference_under_reports_the_marginal_log_odds_ratio_standard_error() -> None:
    """The departure this study records, and the margin that lets it publish `equivalent`.

    R ``tmle`` 2.1.1 matches the bound on ``ate`` and ``rr`` and sits below it on ``or``.
    The paired verdict stays `equivalent`, because the two SE ratios differ by less than the
    shared `calibration_noninferiority` margin.  That is the margin working as declared, and
    it is also why the disagreement needs recording here rather than in a verdict column.
    """
    shares = _scaled_standard_errors()
    reference = study.STUDY.reference
    assert reference is not None
    for estimand in ("ate", "rr"):
        share = float(shares.loc[(reference, estimand), "bound_share"])
        assert share == pytest.approx(1.0, abs=BOUND_TOLERANCE)

    logor = float(shares.loc[(reference, "or"), "bound_share"])
    assert logor == pytest.approx(R_LOGOR_BOUND_SHARE, abs=R_LOGOR_BOUND_SHARE_TOLERANCE)
    assert logor < 1.0 - BOUND_TOLERANCE

    equivalence = pd.read_csv(study.STUDY.artifact("equivalence.csv")).set_index(
        ["scenario", "estimand"]
    )
    row = equivalence.loc[(study.SCENARIO, "or")]
    assert bool(row["se_comparable"])
    assert row["comparison_conclusion"] == "equivalent"
    assert 0.0 < float(row["se_ratio_difference"]) < study.STUDY.margins.calibration_noninferiority


def test_primary_rows_preserve_native_ratio_inference_and_r_inputs() -> None:
    samples, truths, estimates = study.draw_and_fit(replicates=1, n=600, n_jobs=1)
    assert {"qn0", "qn1", "gn1", "obs_weight"}.issubset(samples)
    assert set(truths["estimand"]) == set(study.ESTIMANDS)
    by_name = estimates.set_index("estimand")
    assert by_name.loc["rr", "inference_scale"] == "log"
    assert by_name.loc["or", "inference_scale"] == "log"
    assert by_name.loc["ate", "inference_scale"] == "identity"
    assert by_name.loc["rr", "inference_estimate"] == pytest.approx(
        np.log(by_name.loc["rr", "estimate"])
    )
    assert by_name.loc["or", "inference_estimate"] == pytest.approx(
        np.log(by_name.loc["or", "estimate"])
    )


def test_the_omitted_weight_control_recovers_only_the_selected_target() -> None:
    weighted, omitted = properties.fit_replication(
        ("weight_necessity", "weighted", 0, 4_000, 1, 641, "both_correct")
    )
    assert weighted["cell"] == "ate__weighted"
    assert omitted["cell"] == "ate__omitted_control"
    assert weighted["truth"] == omitted["truth"] == pytest.approx(properties.TRUTH)
    assert abs(weighted["estimate"] - properties.TRUTH) < 0.08
    assert abs(omitted["estimate"] - properties.SELECTED_TRUTH) < 0.08
    assert abs(omitted["estimate"] - properties.TRUTH) > 0.10


def test_the_targeting_control_removes_only_the_fluctuation() -> None:
    targeted, untargeted = properties.fit_replication(
        ("targeting_necessity", "targeted", 0, 4_000, 1, 877, "treatment_correct")
    )
    assert targeted["cell"] == "ate__targeted"
    assert untargeted["cell"] == "ate__untargeted"
    assert targeted["truth"] == untargeted["truth"] == pytest.approx(properties.TRUTH)
    assert targeted["std_error"] == untargeted["std_error"]
    assert abs(targeted["estimate"] - properties.TRUTH) < 0.08
    assert abs(untargeted["estimate"] - properties.TRUTH) > 0.20
