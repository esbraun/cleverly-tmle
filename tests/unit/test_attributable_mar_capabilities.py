"""Post-fit capabilities of a joint natural-course fit with missing outcomes.

A joint fit reports ``ey_obs``, ``par`` or ``paf`` beside arm means.  It fits the
treatment mechanism, so every reader of ``g`` runs.  Each assessment that a
natural-course target defeats keeps a target-specific refusal: the E-value, the
missingness tilt, the omitted-variable pointer and the simulated-confounding surface.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import CapabilityError, PositivityWarning, load
from cleverly.estimators import TMLE
from cleverly.inference.results import reported_status
from cleverly.targets.population_intervention import (
    NATURAL_COURSE_SUPPORT_REFUSAL,
    natural_course_tilt_refusal,
    reads_natural_course_mean,
)
from tests.unit.test_attributable_mar_stack import _frame

JOINT = ("ey_obs", "ey0", "par", "paf")


def _fit(estimands: tuple[str, ...] = JOINT, *, arms: int = 2, **extra: Any) -> Any:
    frame = extra.pop("frame", None)
    roles = extra.pop("roles", {})
    settings: dict[str, Any] = {"cross_fit": False, "random_state": 1, **extra}
    return (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(),
            estimands=estimands,
            **settings,
        )
        .fit(
            _frame(arms=arms) if frame is None else frame,
            outcome="Y",
            treatment="A",
            covariates=("W1", "W2"),
            delta="Delta",
            **roles,
        )
        .single()
    )


@pytest.fixture(scope="module")
def joint() -> Any:
    return _fit()


class TestTheInferenceStatus:
    @pytest.mark.parametrize(
        ("estimands", "extra"),
        [
            pytest.param(JOINT, {}, id="in-sample"),
            pytest.param(("ey_obs", "ey", "par", "paf"), {"arms": 3}, id="three-arm"),
            pytest.param(JOINT, {"cross_fit": True, "n_folds": 5}, id="stacked"),
            pytest.param(
                ("ey_obs", "ey", "par"),
                {"arms": 3, "cross_fit": True, "n_folds": 5},
                id="stacked-three-arm",
            ),
            pytest.param(
                JOINT,
                {"frame": _frame().assign(w=np.linspace(0.5, 1.5, 400)), "roles": {"weights": "w"}},
                id="weighted",
            ),
            pytest.param(
                JOINT,
                {
                    "frame": _frame().assign(c=np.arange(400) // 8),
                    "roles": {"id": "c"},
                },
                id="clustered",
            ),
        ],
    )
    def test_every_admitted_route_reports_influence_curve_inference(
        self, estimands: tuple[str, ...], extra: dict[str, Any]
    ) -> None:
        result = _fit(estimands, **extra)
        assert reported_status(result.estimates) == "influence_curve"
        assert reads_natural_course_mean(result)


class TestTheReadersOfTheTreatmentMechanism:
    def test_the_support_report_runs_on_a_joint_fit(self, joint: Any) -> None:
        report = joint.diagnostics.support()
        assert report is not None

    def test_the_scalar_three_arm_natural_course_still_refuses_support(self) -> None:
        scalar = _fit(("ey_obs",), arms=3)
        with pytest.raises(CapabilityError) as caught:
            scalar.diagnostics.support()
        assert str(caught.value).endswith(NATURAL_COURSE_SUPPORT_REFUSAL)

    def test_the_scalar_three_arm_readers_run(self) -> None:
        scalar = _fit(("ey_obs",), arms=3)
        models = scalar.diagnostics.nuisance_models()
        assert models is not None
        curve = scalar.diagnostics.truncation_curve(mechanism=True)
        assert set(curve["estimand"]) == {"ey_obs"}

    def test_each_fluctuation_has_a_score_row_near_zero(self, joint: Any) -> None:
        check = joint.diagnostics.score_equations()
        frame = check.to_frame()
        rows = frame[frame["kind"] == "fluctuation"]
        assert set(rows["name"]) == {"natural_course", "mean"}
        assert bool(rows["passed"].all())

    @pytest.mark.parametrize("mechanism", [False, True], ids=["treatment", "response"])
    def test_the_truncation_curve_replays_both_fluctuations(
        self, joint: Any, mechanism: bool
    ) -> None:
        curve = joint.diagnostics.truncation_curve(mechanism=mechanism)
        assert set(curve["estimand"]) == set(JOINT)
        fitted = curve.groupby("estimand")["fitted_psi"].first()
        for name in JOINT:
            assert fitted[name] == joint.psi(name)


def test_the_three_arm_scalar_fit_warns_on_a_low_realised_response() -> None:
    """At K = 3 the natural-course warning reads the realised arm's response."""
    rng = np.random.default_rng(4)
    n = 1200
    a = np.repeat([0.0, 1.0, 2.0], n // 3)
    w = rng.normal(size=n)
    delta = np.where(a == 2.0, np.arange(n) % 400 == 0, rng.random(n) < 0.8).astype(float)
    y = rng.binomial(1, 0.4, size=n).astype(float)
    frame = pd.DataFrame({"Y": np.where(delta == 1.0, y, np.nan), "A": a, "W": w, "Delta": delta})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TMLE(
            outcome_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(C=1e6),
            estimands=("ey_obs",),
            cross_fit=False,
            nuisance_bound=0.01,
        ).fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta")
    messages = [
        str(item.message) for item in caught if issubclass(item.category, PositivityWarning)
    ]
    assert any("natural-course response-residual covariate" in text for text in messages)


class TestTheTargetSpecificRefusals:
    def test_the_e_value_names_the_natural_course_comparison(self, joint: Any) -> None:
        with pytest.raises(CapabilityError) as caught:
            joint.sensitivity.evalue("par")
        assert str(caught.value) == (
            "an E-value is defined for a two-arm risk ratio or contrast, and par compares "
            "the natural course with one reference arm"
        )

    def test_the_e_value_names_the_natural_course_mean(self, joint: Any) -> None:
        with pytest.raises(CapabilityError) as caught:
            joint.sensitivity.evalue("ey_obs")
        assert str(caught.value) == (
            "an E-value is defined for a two-arm risk ratio or contrast, and ey_obs is the "
            "natural-course mean, not a two-arm contrast"
        )

    def test_the_e_value_sentence_is_the_same_on_complete_data(self) -> None:
        frame = _frame().assign(Y=lambda f: f["Y"].fillna(0.0)).drop(columns="Delta")
        complete = (
            TMLE(
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(),
                estimands=("paf",),
                cross_fit=False,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2"))
            .single()
        )
        with pytest.raises(CapabilityError, match="paf compares the natural course"):
            complete.sensitivity.evalue("paf")

    def test_the_omitted_variable_pointer_names_the_arm_means(self, joint: Any) -> None:
        with pytest.raises(CapabilityError) as caught:
            joint.sensitivity.omitted_confounding("ey0")
        message = str(caught.value)
        assert "on this fit's arm means, and not on a target read from the natural-course" in (
            message
        )

    def test_a_par_bound_request_is_not_pointed_at_the_tilt(self, joint: Any) -> None:
        from cleverly.sensitivity.omitted_variable import sensitivity_elements

        with pytest.raises(CapabilityError) as caught:
            sensitivity_elements(joint, "par")
        assert "missingness tilt" not in str(caught.value)
        with pytest.raises(CapabilityError) as arm:
            sensitivity_elements(joint, "ey0")
        assert "missingness tilt remains" in str(arm.value)

    def test_an_explicit_tilt_of_par_refuses(self, joint: Any) -> None:
        with pytest.raises(CapabilityError) as caught:
            joint.sensitivity.missingness(estimands=["par", "ey0"])
        assert str(caught.value) == natural_course_tilt_refusal(["par"])

    def test_the_default_tilt_sweep_skips_the_natural_course_targets(self, joint: Any) -> None:
        frame = joint.sensitivity.missingness()
        assert set(frame["estimand"]) == {"ey0"}
        row = next(row for row in joint.sensitivity.capabilities if row.operation == "missingness")
        assert "the default sweep skips ['ey_obs', 'paf', 'par']" in row.interpretation

    @pytest.mark.parametrize("operation", ["missingness", "tipping_gamma"])
    def test_a_fit_with_no_arm_mean_names_the_natural_course(self, operation: str) -> None:
        """Rule 7, with the natural-course sentence, on the row and on the call."""
        result = _fit(("par", "paf"))
        row = result.sensitivity.capability(operation)
        assert not row.available
        assert "re-mixes the arm-indexed means" in str(row.reason)
        assert str(row.reason).endswith(natural_course_tilt_refusal(["paf", "par"]))
        call = getattr(result.sensitivity, operation)
        with pytest.raises(CapabilityError) as caught:
            call() if operation == "missingness" else call("par")
        assert str(row.reason) in str(caught.value)

    @pytest.mark.parametrize("estimands", [JOINT, ("ey0", "par")], ids=["joint", "par_and_ey0"])
    def test_the_tipping_row_declares_what_the_bare_call_does(
        self, estimands: tuple[str, ...]
    ) -> None:
        """The row defers on the estimand, and the bare call raises the row's own reason."""
        result = _fit(estimands)
        row = result.sensitivity.capability("tipping_gamma")
        assert not row.available
        assert row.status.value == "deferred"
        assert "estimand" in row.requires_arguments
        assert "read the natural-course mean" in str(row.reason)
        assert str(row.reason).endswith("Choose an explicit estimand from ['ey0']")
        with pytest.raises(CapabilityError) as caught:
            result.sensitivity.tipping_gamma()
        assert str(caught.value) == row.reason
        # Naming the arm mean lifts the deferral, and the call runs.
        result.sensitivity.tipping_gamma("ey0")

    def test_simulated_confounding_keeps_the_missing_outcome_refusal(self, joint: Any) -> None:
        with pytest.raises(CapabilityError, match=r"docs/roadmap.md F12 tracks this stop"):
            joint.sensitivity.simulated_confounding("par")


class TestRefitsAndPersistence:
    def test_placebo_and_subset_refutations_run(self, joint: Any) -> None:
        report = joint.diagnostics.refute(
            estimand="par", tests=("placebo", "subset"), n_replicates=2, random_state=1
        )
        assert report is not None

    def test_a_saved_joint_fit_reloads_both_fluctuations(self, joint: Any, tmp_path: Any) -> None:
        path = tmp_path / "joint.joblib"
        joint.save(path)
        back = load(path)
        assert tuple(back.fluctuations) == ("natural_course", "mean")
        for name in JOINT:
            assert back.psi(name) == joint.psi(name)
            np.testing.assert_array_equal(back[name].influence_curve, joint[name].influence_curve)


@pytest.mark.parametrize("estimands", [("ey_obs",), ("par",), ("ey0", "paf")])
def test_a_missing_treatment_keeps_its_refusal_of_the_natural_course(
    estimands: tuple[str, ...],
) -> None:
    """X23's data-layer refusal: a missing treatment leaves P(A | W) unidentified."""
    frame = _frame()
    frame["DeltaA"] = (np.arange(len(frame)) % 7 != 0).astype(float)
    frame.loc[frame["DeltaA"] == 0.0, "A"] = np.nan
    with pytest.raises(CapabilityError, match="read the treatment law of every row"):
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(),
            estimands=estimands,
            cross_fit=False,
        ).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W1", "W2"),
            delta="Delta",
            treatment_delta="DeltaA",
        )


def test_the_ratio_of_the_two_means_is_one_minus_the_paf(joint: Any) -> None:
    """The log-ratio interval of 1 - PAF, the tmle3 form, from the same-row curves."""
    ratio = joint.ratio("ey0", "ey_obs")
    assert ratio.scale == "ratio"
    assert ratio.psi == pytest.approx(1.0 - joint.psi("paf"), abs=1e-14)
    observed = np.asarray(joint["ey_obs"].influence_curve)
    reference = np.asarray(joint["ey0"].influence_curve)
    log_curve = reference / joint.psi("ey0") - observed / joint.psi("ey_obs")
    n = observed.size
    assert ratio.std_error == pytest.approx(np.sqrt(np.var(log_curve, ddof=1) / n), rel=1e-10)
    low, high = ratio.ci
    assert low < ratio.psi < high
