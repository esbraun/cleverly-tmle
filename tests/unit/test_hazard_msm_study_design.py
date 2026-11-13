"""The ``hazard-msm`` declaration: truths, bounds, limits, sizes, pilot floors and seeds.

The truths are checked against a root finder and a fit on each law's exact sample, the
efficiency bounds against the hazard form of the influence function and a large-sample fit,
the robustness limits and the wrong learner against the exact sample, and the replicate counts
against the shared sizing rules.  The pilot readings come from the committed pilot artifact.
Nothing here reads a study run.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root
from scipy.stats import beta, norm
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly.datasets import point_survival_truth
from cleverly.longitudinal import LTMLE
from cleverly.utils.bounds import DEFAULT_LTMLE_G_BOUNDS
from tests.studies import canonical_hazard_msm as study
from tests.studies import hazard_msm_law as law
from tests.studies import hazard_msm_pilot as pilot
from tests.studies import hazard_msm_properties as properties
from tests.studies import mtp_study_tools
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("hazard_msm_study_design")

MARGINS = study.STUDY.margins
MODIFIER = study.MODIFIER_LABEL
#: The ``competing`` death coefficient's plug-in gap under the wrong hazard learner, in
#: efficiency SEs at n = 2,000.
COMPETING_PLUG_IN_GAP = -4.603


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _scale(label: str, n: int) -> float:
    return properties.EFFICIENCY_SD[label] / np.sqrt(n)


def _exact_fit(fit_label: str, configuration: str = "both_correct") -> Any:
    """One fit label's fit on its law's exact sample, each outcome weighted by its mass."""
    scenario, link = properties.FITS[fit_label]
    current = law.LAWS[scenario]
    data = current.container(current.exact_frame(), weights="wt")
    return _quiet(
        LTMLE(
            law.REGIMENS,
            n_folds=1,
            simultaneous=False,
            survival_model="hazard",
            targeting=study.ALGORITHM,
            msm=law.working_model(link),
            tol=study.TOL,
            random_state=0,
            **properties.learners(configuration),
        ).fit,
        data,
    )


@pytest.fixture(scope="module")
def exact_fits() -> dict[tuple[str, str], Any]:
    """The survival logit fit of every robustness configuration, both other fit labels, and
    the competing fit with the wrong hazard learner (the cross-cause robustness cell)."""
    fits = {
        ("survival_logit", configuration): _exact_fit("survival_logit", configuration)
        for configuration in properties.CONFIGURATIONS
    }
    for label in ("survival_identity", "competing_logit"):
        fits[(label, "both_correct")] = _exact_fit(label)
    fits[("competing_logit", "mechanism_correct")] = _exact_fit(
        "competing_logit", "mechanism_correct"
    )
    return fits


# ---------------------------------------------------------------------------- truths


class TestTheTruths:
    @pytest.mark.parametrize("scenario", list(law.LAWS))
    @pytest.mark.parametrize("link", law.LINKS)
    def test_the_projection_solves_its_score_equation(self, scenario: str, link: str) -> None:
        current = law.LAWS[scenario]
        for cause in current.causes:
            risks = law.stratum_risks(current, cause)

            def score(coefficients: np.ndarray, risks: Any = risks) -> np.ndarray:
                out = np.zeros(len(law.TERMS))
                for stratum in law.STRATA:
                    rows = np.array([law.phi(*cell, stratum[0]) for cell in law.cells(current)])
                    eta = rows @ coefficients
                    mean = 1.0 / (1.0 + np.exp(-eta)) if link == "logit" else eta
                    out += law.MASS * rows.T @ (risks[stratum] - mean)
                return out

            solved = root(score, np.zeros(len(law.TERMS)), tol=1e-14)
            assert solved.success
            np.testing.assert_allclose(
                law.exact_beta(current, cause, link), solved.x, rtol=0, atol=1e-10
            )

    def test_the_cell_risks_are_the_cumulative_incidences(self) -> None:
        # Each cause's incidence at the last visit plus the survival there is one, and every
        # curve rises.
        for current in law.LAWS.values():
            for w1, w2 in law.STRATA:
                for a in (0, 1):
                    curve = current.curve(a, w1, w2)
                    survival = 1.0
                    for node in range(1, current.n_times + 1):
                        survival *= 1.0 - current.hazard(node, a, w1, w2).sum()
                    assert curve[-1].sum() + survival == pytest.approx(1.0, abs=1e-14)
                    assert np.all(np.diff(curve, axis=0) > 0.0)

    def test_the_shared_law_is_the_point_survival_law(self) -> None:
        for a in (0, 1):

            def arm(w1: int, w2: int, a: int = a) -> list[float]:
                return [1.0 - a, float(a)]

            expected = point_survival_truth(arm, n_times=5)
            ours = np.mean(
                [law.SHARED_SURVIVAL.curve(a, w1, w2)[:, 0] for w1, w2 in law.STRATA], axis=0
            )
            np.testing.assert_allclose(np.ravel(expected), ours, rtol=0, atol=1e-14)

    def test_the_truths_are_the_declared_projection(self) -> None:
        for label, name in properties.COEFFICIENTS["survival_logit"]:
            assert study.TRUTH["survival"][name] == properties.TRUTHS[label]
        assert study.TRUTH["survival"][properties.ESTIMAND[MODIFIER]] == pytest.approx(
            -0.9202596372, abs=1e-9
        )

    @staticmethod
    def _smallest(current: law.Law) -> float:
        smallest = 1.0
        for w1, w2 in law.STRATA:
            for a in (0, 1):
                retained = np.prod(
                    [current.retained(k, a, w1, w2) for k in range(1, current.n_times + 1)]
                )
                smallest = min(smallest, current.g(w1, w2)[a] * retained)
        return smallest

    def test_the_bounds_never_bind(self) -> None:
        # The smallest product of a treatment and a retention probability through the last
        # visit stays far above zero on the gated laws, so no truncation bound reads.  On the
        # overlap law it is 0.046 (treated at W1 = 0, W2 = 0), still above four times the
        # default cumulative bound.
        for current in study.SCENARIOS.values():
            assert self._smallest(current) > 0.1
        overlap = self._smallest(law.OVERLAP)
        assert overlap == pytest.approx(0.0458, abs=1e-4)
        assert overlap > 4.0 * DEFAULT_LTMLE_G_BOUNDS[0]

    def test_the_overlap_law_shares_the_survival_hazard(self) -> None:
        assert law.OVERLAP.hazards == law.SURVIVAL.hazards
        assert law.OVERLAP.dropout_coefficients == law.SURVIVAL.dropout_coefficients
        for label, name in properties.COEFFICIENTS["overlap_logit"]:
            survival = label.replace("overlap_logit", "survival_logit")
            # The truth is a functional of the hazard and the law of W alone.
            assert properties.TRUTHS[label] == pytest.approx(properties.TRUTHS[survival], abs=1e-14)
            assert properties.ESTIMAND[label] == name


# ---------------------------------------------------------------------------- bounds


def _hazard_form(
    current: law.Law,
    label: str,
    horizon: int,
    cause: int,
    w1: int,
    w2: int,
    a: int,
    time: float,
    event: int,
) -> float:
    """The uncentred curve of one cell in the cause-specific hazard form.

    :math:`F_j(t) + \\sum_{k \\le t} \\frac{1\\{A = a_r\\} R_k}{g G_k} \\sum_l H_{lk}
    (dN_l(k) - \\lambda_l(k))`, with :math:`H_{jk} = 1 - (F_j(t) - F_j(k)) / S(k)` and
    :math:`H_{lk} = -(F_j(t) - F_j(k)) / S(k)` for another cause.  It shares no code with
    :meth:`~tests.studies.hazard_msm_law.Law.cell_curve`.
    """
    arm = law.REGIMENS[label]
    incidence = current.curve(arm, w1, w2)[:, cause - 1]
    total = float(incidence[horizon - 1])
    if a != arm:
        return total
    survival = 1.0
    retained = 1.0
    for node in range(1, min(int(time), horizon) + 1):
        hazards = current.hazard(node, a, w1, w2)
        survival *= 1.0 - hazards.sum()
        retained *= current.retained(node, a, w1, w2)
        remaining = (incidence[horizon - 1] - incidence[node - 1]) / survival
        for code, value in enumerate(hazards, start=1):
            clever = (1.0 if code == cause else 0.0) - remaining
            jump = 1.0 if (event == code and int(time) == node) else 0.0
            total += clever * (jump - value) / (current.g(w1, w2)[a] * retained)
    return total


class TestTheEfficiencyBounds:
    @pytest.mark.parametrize("scenario", list(law.LAWS))
    def test_the_iterated_mean_and_hazard_forms_agree(self, scenario: str) -> None:
        current = law.LAWS[scenario]
        for cause in current.causes:
            for w1, w2 in law.STRATA:
                for a in (0, 1):
                    for _, time, event in current.outcomes(a, w1, w2):
                        for label, horizon in law.cells(current):
                            assert current.cell_curve(
                                label, horizon, cause, w1, w2, a, time, event
                            ) == pytest.approx(
                                _hazard_form(
                                    current, label, horizon, cause, w1, w2, a, time, event
                                ),
                                abs=1e-12,
                            )

    def test_a_cell_risk_bound_is_its_own_variance(self) -> None:
        # A unit coefficient on one cell gives that cell's bound, which splits into the variance
        # of its regression across W and the mean square of its weighted residuals.
        current = law.SURVIVAL
        cells = law.cells(current)
        index = cells.index(("a1", 3))
        unit = np.zeros((1, len(cells)))
        unit[0, index] = 1.0
        psi = float(np.mean([current.curve(1, w1, w2)[2, 0] for w1, w2 in law.STRATA]))
        bound = law.coefficient_sd(current, 1, lambda _: unit, lambda _: np.array([-psi]))[0]
        between = np.var([current.curve(1, w1, w2)[2, 0] for w1, w2 in law.STRATA])
        within = 0.0
        for w1, w2 in law.STRATA:
            q = current.regression(1, w1, w2, 3, 1)[1]
            g = current.g(w1, w2)[1]
            for mass, time, event in current.outcomes(1, w1, w2):
                residual = current.cell_curve("a1", 3, 1, w1, w2, 1, time, event) - q
                within += law.MASS * g * mass * residual**2
        assert bound**2 == pytest.approx(between + within, rel=1e-12)

    def test_the_declared_bounds(self) -> None:
        np.testing.assert_allclose(
            [properties.EFFICIENCY_SD[label] for label in study.SURVIVAL_LABELS],
            [3.191, 5.010, 0.739, 6.686],
            atol=5e-4,
        )
        np.testing.assert_allclose(
            [properties.EFFICIENCY_SD[label] for label in study.OVERLAP_LABELS],
            [3.656, 6.479, 0.898, 9.484],
            atol=5e-4,
        )

    def test_a_large_fit_reports_the_bound(self) -> None:
        frame = law.SURVIVAL.draw(40_000, 20261113)
        result = _quiet(study.fit, "survival", frame)
        for label, name in properties.COEFFICIENTS["survival_logit"]:
            curve = np.asarray(result[name].influence_curve)
            assert np.std(curve) == pytest.approx(properties.EFFICIENCY_SD[label], rel=0.05), label


# ---------------------------------------------------------------------------- robustness


class TestTheRobustnessCells:
    def test_every_positive_limit_is_the_truth(self, exact_fits: dict[Any, Any]) -> None:
        for (label, configuration), result in exact_fits.items():
            if configuration == "both_wrong":
                continue
            for cell_label, name in properties.COEFFICIENTS[label]:
                assert result[name].psi == pytest.approx(properties.TRUTHS[cell_label], abs=1e-8), (
                    label,
                    configuration,
                    cell_label,
                )

    def test_the_both_wrong_control_is_discriminated(self, exact_fits: dict[Any, Any]) -> None:
        result = exact_fits[("survival_logit", "both_wrong")]
        gap = result[properties.ESTIMAND[MODIFIER]].psi - properties.TRUTHS[MODIFIER]
        n = properties.DOUBLE_ROBUST_N
        # Read at the efficient spread and again at 1.5 times it, since a wrong fit's own
        # spread can exceed the bound.  The two gap-read controls are this one and the
        # untargeted arm.
        assert gap / _scale(MODIFIER, n) == pytest.approx(-2.097, abs=1e-3)
        for spread in (_scale(MODIFIER, n), 1.5 * _scale(MODIFIER, n)):
            probability = sizing.bias_discriminated_probability(
                MARGINS, gap / spread, properties.DOUBLE_ROBUST_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(2)

    def test_the_cross_cause_cell_needs_its_clever_covariate(
        self, exact_fits: dict[Any, Any]
    ) -> None:
        """On ``competing`` the wrong hazard learner's plug-in death coefficient is far off.

        The targeted limit is the truth (:meth:`test_every_positive_limit_is_the_truth`), so the
        positive cell reads the clever covariate that carries the relapse hazard into the death
        update.  The estimator's plug-in matches the exact limit of the learner.
        """
        label = properties.COMPETING_DR_LABEL
        index = law.TERMS.index("a_W1")
        death = next(cause for cause in law.COMPETING.causes if law.CAUSES[cause] == "death")
        limit = law.wrong_hazard_limit(law.COMPETING, cause=death)
        gap = (limit[index] - properties.TRUTHS[label]) / _scale(label, properties.DOUBLE_ROBUST_N)
        assert abs(gap) >= 3.0
        assert gap == pytest.approx(COMPETING_PLUG_IN_GAP, abs=1e-3)
        result = exact_fits[("competing_logit", "mechanism_correct")]
        assert properties.plug_in(result, properties.ESTIMAND[label]) == pytest.approx(
            limit[index], abs=1e-7
        )
        assert result[properties.ESTIMAND[label]].psi == pytest.approx(
            properties.TRUTHS[label], abs=1e-8
        )


# ---------------------------------------------------------------------------- wrong learner


PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()
FITS_DIFFER = PILOT.loc[PILOT["label"] == "fits_differ"] if not PILOT.empty else PILOT
WRONG_LEARNER = PILOT.loc[PILOT["label"] == "wrong_learner"] if not PILOT.empty else PILOT
MECHANISM_DIFFER = (
    PILOT.loc[PILOT["label"] == "fits_differ_mechanism_correct"] if not PILOT.empty else PILOT
)
CALIBRATION_PILOTED = not PILOT.empty and PILOT["label"].str.startswith("calibration_").any()

#: The pass probability each positive bias arm needs (design review d).
POSITIVE_PASS = 0.99
#: Every positive arm whose verdict reads a bias interval, with its replicates: the
#: ``double_robustness`` positives, the targeted necessity arm and the claimed root-n rungs.
POSITIVE_ARMS: dict[str, int] = {
    cell.cell: cell.replicates
    for cell in properties.declared_cells()
    if cell.role == "positive"
    and (
        cell.property in ("double_robustness", "targeting_necessity")
        or (cell.property == "root_n_and_efficiency" and cell.n != min(properties.RATE_SIZES))
    )
}


def _tolerated_bias(replicates: int) -> float:
    """The largest |standardized bias| at which a positive arm passes with ``POSITIVE_PASS``."""
    z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
    return MARGINS.standardized_bias - (z + float(norm.ppf(POSITIVE_PASS))) / np.sqrt(replicates)


class TestTheWrongHazardLearner:
    def test_the_wrong_hazard_learner_rule(self) -> None:
        """The plug-in ``a_W1`` limit lies at least 3 efficiency SEs from the truth at n = 2,000."""
        limit = law.wrong_hazard_limit(law.SURVIVAL)
        index = law.TERMS.index("a_W1")
        bias = (limit[index] - properties.TRUTHS[MODIFIER]) / _scale(MODIFIER, 2_000)
        assert bias >= 3.0
        assert bias == pytest.approx(5.897, abs=1e-3)
        assert abs(limit[index]) > 0.01  # the untargeted coefficient is not a constant zero

    def test_the_estimator_plug_in_is_the_limit(self, exact_fits: dict[Any, Any]) -> None:
        result = exact_fits[("survival_logit", "mechanism_correct")]
        limit = law.wrong_hazard_limit(law.SURVIVAL)
        for index, (_, name) in enumerate(properties.COEFFICIENTS["survival_logit"]):
            assert properties.plug_in(result, name) == pytest.approx(limit[index], abs=1e-7)

    def test_the_shared_law_fails_the_rule(self) -> None:
        index = law.TERMS.index("a_W1")
        shared = law.SHARED_SURVIVAL
        gap = law.wrong_hazard_limit(shared)[index] - law.exact_beta(shared)[index]
        bias = gap / (law.efficiency_sd(shared)[index] / np.sqrt(2_000))
        assert abs(bias) < 0.1

    @pytest.mark.skipif(WRONG_LEARNER.empty, reason="the pilot has not been committed")
    def test_the_pilot_plug_in_varies(self) -> None:
        rows = WRONG_LEARNER.loc[WRONG_LEARNER["estimand"] == properties.ESTIMAND[MODIFIER]]
        assert rows["refusal"].fillna("").eq("").all()
        assert len(rows) == pilot.WRONG_LEARNER_DRAWS
        values = rows["wrong_plug_in"].to_numpy(dtype=float)
        assert np.var(values, ddof=1) > 0.0
        # The pilot mean sits within three of its standard errors of the exact limit.
        limit = law.wrong_hazard_limit(law.SURVIVAL)[law.TERMS.index("a_W1")]
        error = np.std(values, ddof=1) / np.sqrt(values.size)
        assert abs(values.mean() - limit) < 3.0 * error


class SaturatedShare(BaseEstimator, ClassifierMixin):
    """The empirical share of ``y`` in each distinct row of the design."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SaturatedShare:
        keys = [tuple(row) for row in np.asarray(X, dtype=float)]
        frame = pd.DataFrame({"key": keys, "y": np.asarray(y, dtype=float)})
        self.shares_ = frame.groupby("key")["y"].mean().to_dict()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.array([self.shares_[tuple(row)] for row in np.asarray(X, dtype=float)])
        return np.column_stack([1.0 - p, p])


def test_a_saturated_mechanism_hides_the_hazard_learner() -> None:
    """Why the mechanisms are parametric and not saturated.

    With the empirical treatment and retention shares of every stratum, the targeted
    coefficients of the right and the wrong hazard learner coincide, so no cell could see the
    hazard fit.  The declared parametric mechanisms separate them.
    """
    frame = law.SURVIVAL.draw(2_000, 20261114)

    def fit(hazard: Any, saturated: bool) -> Any:
        mechanism = (
            {"treatment_learner": SaturatedShare(), "censoring_learner": SaturatedShare()}
            if saturated
            else {
                "treatment_learner": law.TreatmentLogistic(),
                "censoring_learner": law.CensoringLogistic(),
            }
        )
        return _quiet(
            study.fit, "survival", frame, learner_set={"outcome_learner": hazard, **mechanism}
        )

    name = properties.ESTIMAND[MODIFIER]
    right, wrong = fit(law.InteractionLogistic(), True), fit(law.NoW2Logistic(), True)
    assert right[name].psi == pytest.approx(wrong[name].psi, abs=1e-6)
    right, wrong = fit(law.InteractionLogistic(), False), fit(law.NoW2Logistic(), False)
    assert abs(right[name].psi - wrong[name].psi) > 0.01


# ---------------------------------------------------------------------------- sizes


class TestTheSizes:
    def test_the_power_cell_has_power(self) -> None:
        power = design_power(
            properties.TRUTHS[MODIFIER], properties.EFFICIENCY_SD[MODIFIER], properties.POWER_N
        )
        assert power >= 0.80
        assert power_cell_pass_probability(power, properties.POWER_REPLICATES) >= 0.99

    def test_the_derived_controls_leave_the_band(self) -> None:
        """The shrunken-SE arm leaves the SE-ratio band and the noise arm the efficiency band."""
        replicates = properties.CALIBRATION_REPLICATES
        assert (
            sizing.calibration_control_probability(
                MARGINS, replicates, properties.SHRUNKEN_SE_FACTOR
            )
            > 0.999
        )
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        noise = np.sqrt(2.0)
        spread = noise / np.sqrt(2.0 * (replicates - 1))
        upper = properties.EFFICIENCY_RATIO_BAND[1]
        assert norm.cdf((noise - z * spread - upper) / spread) > 0.999

    @pytest.mark.skipif(CALIBRATION_PILOTED, reason="the committed calibration pilot sizes R")
    def test_the_calibration_replicates_at_a_calibrated_pilot(self) -> None:
        """The arm-pooled rule at a pilot of the declared size that reads a ratio of one.

        Every calibration positive arm is one arm.  The committed pilot replaces this reading
        once it exists (:meth:`test_the_calibration_replicates_follow_the_pilot`), and this
        test then skips.
        """
        arms = sum(len(properties.COEFFICIENTS[label]) for label in properties.CALIBRATION_FITS)
        assert arms == 13
        replicates, cell, family = mtp_study_tools.smallest_replicates(
            1.0, sizing.CALIBRATION_PILOT_DRAWS, arms
        )
        assert replicates == properties.CALIBRATION_REPLICATES
        assert cell >= mtp_study_tools.CELL_TARGET and family >= mtp_study_tools.FAMILY_TARGET

    @staticmethod
    def _calibration_rows(fit_label: str) -> pd.DataFrame:
        rows = PILOT.loc[PILOT["label"] == f"calibration_{fit_label}"]
        if rows.empty:
            pytest.skip(f"the calibration pilot of {fit_label} has not been committed")
        return rows

    def test_the_calibration_replicates_follow_the_pilot(self) -> None:
        ratios = []
        for fit_label in properties.CALIBRATION_FITS:
            rows = self._calibration_rows(fit_label)
            assert rows["refusal"].fillna("").eq("").all()
            for _, name in properties.COEFFICIENTS[fit_label]:
                arm = rows.loc[rows["estimand"] == name]
                assert len(arm) == sizing.CALIBRATION_PILOT_DRAWS
                ratios.append(float(arm["std_error"].mean() / arm["estimate"].std(ddof=1)))
        reading = mtp_study_tools.replicates_from_pilot(ratios, sizing.CALIBRATION_PILOT_DRAWS)
        assert not reading["suspect"]
        assert all(reading["screened"])
        assert reading["R"] == properties.CALIBRATION_REPLICATES

    def test_every_positive_arm_passes_at_its_exact_limit(self) -> None:
        """Each positive bias arm passes with probability at least 0.99 at standardized bias 0.

        Every positive limit is the truth (:class:`TestTheRobustnessCells`); the finite-sample
        bias is read from the pilots below where one exists.  The largest |bias| each count
        tolerates at 0.99 is :func:`_tolerated_bias`.
        """
        for cell, replicates in POSITIVE_ARMS.items():
            probability = sizing.bias_equivalent_probability(MARGINS, 0.0, replicates)
            assert probability >= POSITIVE_PASS, (cell, probability)
        assert _tolerated_bias(properties.DOUBLE_ROBUST_REPLICATES) == pytest.approx(
            0.1085, abs=1e-3
        )
        assert _tolerated_bias(properties.RATE_REPLICATES) == pytest.approx(0.0767, abs=1e-3)

    @pytest.mark.skipif(WRONG_LEARNER.empty, reason="the pilot has not been committed")
    def test_the_wrong_learner_arms_pass_at_the_pilot_bias(self) -> None:
        """The targeted ``a_W1`` arm of the wrong hazard learner, read at the pilot.

        The ``wrong_learner`` pilot fits the ``double_robustness/mechanism_correct`` and
        ``targeting_necessity/targeted`` configuration.  The arm is read at the pilot's
        standardized bias plus one of its standard errors.
        """
        rows = WRONG_LEARNER.loc[WRONG_LEARNER["estimand"] == properties.ESTIMAND[MODIFIER]]
        error = rows["wrong_targeted"].to_numpy(dtype=float) - properties.TRUTHS[MODIFIER]
        standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
        assert standardized < 0.1
        for replicates in (properties.DOUBLE_ROBUST_REPLICATES, properties.NECESSITY_REPLICATES):
            probability = sizing.bias_equivalent_probability(MARGINS, standardized, replicates)
            assert probability >= POSITIVE_PASS, probability

    def test_the_positive_arms_follow_the_calibration_pilot(self) -> None:
        """The ``both_correct`` arm and the n = 2,000 rungs at the calibration pilot's biases."""
        rows = self._calibration_rows("survival_logit")
        for cell_label, name in properties.COEFFICIENTS["survival_logit"]:
            arm = rows.loc[rows["estimand"] == name]
            error = (arm["estimate"] - arm["truth"]).to_numpy(dtype=float)
            standardized = abs(error.mean()) / error.std(ddof=1)
            for cell, replicates in POSITIVE_ARMS.items():
                if cell.startswith(f"{cell_label}__") and (
                    cell.endswith("__both_correct") or cell.endswith("__n_2000")
                ):
                    probability = sizing.bias_equivalent_probability(
                        MARGINS, standardized, replicates
                    )
                    assert probability >= POSITIVE_PASS, (cell, probability)

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` on the calibration pilot.

        The ``survival_logit`` and ``competing_logit`` calibration pilots draw the primary
        scenarios at the primary n with the primary fits.  One implementation, clauses (a) to
        (c) on their covered counts and clause (d) on their standardized biases, one group per
        scenario; both shares are :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`.
        """
        covered: dict[str, int] = {}
        groups = []
        for scenario, fit_label in (
            ("survival", "survival_logit"),
            ("competing", "competing_logit"),
        ):
            rows = self._calibration_rows(fit_label)
            assert int(rows["n"].iloc[0]) == study.PRIMARY_N
            names = list(study.ESTIMANDS[scenario])
            hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
            covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
            error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
                index="replicate", columns="estimand", values="error"
            )[names]
            standardized = error / error.std(ddof=1)
            groups.append(
                sizing.PilotGroup(
                    standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
                )
            )
        share = sizing.PRIMARY_CLAUSE_SHARE
        assert (
            sizing.primary_replicates(
                MARGINS,
                covered,
                sizing.CALIBRATION_PILOT_DRAWS,
                implementations=1,
                coverage_pass=share,
                bias_groups=groups,
                bias_pass=share,
            )
            == study.PRIMARY_REPLICATES
        )


# ---------------------------------------------------------------------------- pilot floors


def _decreasing() -> tuple[int, int, float, float]:
    rows = FITS_DIFFER.loc[FITS_DIFFER["estimand"] == "largest_decrease"]
    count, total = int((rows["estimate"] > 0.0).sum()), len(rows)
    tail = (1.0 - MARGINS.confidence_level) / 2.0
    low = float(beta.ppf(tail, count, total - count + 1)) if count else 0.0
    high = float(beta.ppf(1.0 - tail, count + 1, total - count)) if count < total else 1.0
    return count, total, low, high


def _paired_ratios(
    rows: pd.DataFrame | None = None, fit_label: str = properties.DIFFERENCE_FIT
) -> dict[str, float]:
    """The SD of hazard minus iterated-mean over the efficiency-bound SE, by coefficient."""
    rows = FITS_DIFFER if rows is None else rows
    ratios = {}
    for label, name in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == name]
        difference = arm["estimate"] - arm["iterated_estimate"]
        ratios[label] = float(difference.std(ddof=1)) / _scale(label, pilot.PILOT_N)
    return ratios


def _side_by_side() -> dict[str, dict[str, float]]:
    """Each fit's coverage and SE ratio (mean SE over empirical SD), by coefficient."""
    out = {}
    for label, name in properties.COEFFICIENTS[properties.DIFFERENCE_FIT]:
        rows = FITS_DIFFER.loc[FITS_DIFFER["estimand"] == name]
        out[label] = {
            "hazard_coverage": float(rows["covered"].mean()),
            "hazard_se_ratio": float(rows["std_error"].mean() / rows["estimate"].std(ddof=1)),
            "iterated_coverage": float(rows["iterated_covered"].mean()),
            "iterated_se_ratio": float(
                rows["iterated_std_error"].mean() / rows["iterated_estimate"].std(ddof=1)
            ),
        }
    return out


@pytest.mark.skipif(FITS_DIFFER.empty, reason="the pilot has not been committed")
class TestTheFitsDifferPilot:
    """The ``iterated_mean_difference`` readings of plan X70 rev 2.4, on the ``overlap`` law.

    The paired-difference floor (0.05) is read on the largest coefficient ratio and on the
    gated coefficient ``a_W1`` (design review d), and every ratio is reported: a coefficient whose information comes from well-overlapped strata need
    not separate the fits.  Each fit's coverage and SE ratio are reported side by side and are
    not gated.  The share of decreasing iterated-mean curves is reported with its 99%
    Clopper-Pearson interval and carries no floor.
    """

    def test_the_pilot_reads_the_overlap_law(self) -> None:
        assert pilot.PILOTS["fits_differ"][0] == properties.DIFFERENCE_FIT
        assert properties.FITS[properties.DIFFERENCE_FIT] == ("overlap", "logit")
        truths = FITS_DIFFER.dropna(subset=["iterated_estimate"])
        for label, name in properties.COEFFICIENTS[properties.DIFFERENCE_FIT]:
            truth = truths.loc[truths["estimand"] == name, "truth"]
            assert np.allclose(truth, properties.TRUTHS[label])

    def test_the_pilot_lost_no_fit(self) -> None:
        assert FITS_DIFFER["refusal"].fillna("").eq("").all()
        assert FITS_DIFFER["replicate"].nunique() == pilot.FITS_DIFFER_DRAWS

    def test_the_two_fits_differ_enough(self) -> None:
        ratios = _paired_ratios()
        assert max(ratios.values()) >= properties.PAIRED_DIFFERENCE_FLOOR, ratios
        gated = f"{properties.DIFFERENCE_FIT}_a_W1"
        assert ratios[gated] >= properties.PAIRED_DIFFERENCE_FLOOR, ratios

    def test_both_fits_report_coverage_and_se_ratio(self) -> None:
        for label, reading in _side_by_side().items():
            assert all(np.isfinite(value) for value in reading.values()), label

    def test_the_decreasing_share_is_reported(self) -> None:
        count, total, low, high = _decreasing()
        assert total == pilot.FITS_DIFFER_DRAWS
        assert 0.0 <= low <= count / total <= high <= 1.0


@pytest.mark.skipif(MECHANISM_DIFFER.empty, reason="the pilot has not been committed")
class TestTheMechanismCorrectDifferencePilot:
    """The paired difference on the gated law's ``mechanism_correct`` configuration.

    Both fits are consistent through the correct mechanism and both regressions omit ``W2``.
    The ratios are reported with no floor: the 200-draw pilot reads 0.04 to 0.07, so the two
    fits do not separate at first order here either.
    """

    def test_the_pilot_reads_the_gated_law(self) -> None:
        assert pilot.PILOTS["fits_differ_mechanism_correct"][0] == "survival_logit"
        for label, name in properties.COEFFICIENTS["survival_logit"]:
            truth = MECHANISM_DIFFER.loc[MECHANISM_DIFFER["estimand"] == name, "truth"]
            assert np.allclose(truth, properties.TRUTHS[label])

    def test_the_pilot_lost_no_fit(self) -> None:
        assert MECHANISM_DIFFER["refusal"].fillna("").eq("").all()
        assert MECHANISM_DIFFER["replicate"].nunique() == pilot.MECHANISM_DIFFER_DRAWS

    def test_the_ratios_are_reported(self) -> None:
        ratios = _paired_ratios(MECHANISM_DIFFER, "survival_logit")
        assert all(np.isfinite(value) and value > 0.0 for value in ratios.values()), ratios


# ---------------------------------------------------------------------------- cells and seeds


class TestTheCells:
    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_declares_every_cell_the_module_writes(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.STUDY.property_cells.items()
            for cell in cells
        }
        written = {f"{cell.property}/{cell.cell}" for cell in properties.declared_cells()}
        filled = {
            name
            for key in properties._payloads(properties.FIT_SETS, 1)
            for name in properties.cells_of(key)
        }
        rate = {name for name in declared if name.startswith("root_n_rate/")}
        assert written | rate == declared
        assert filled | rate == declared

    def test_the_study_is_not_yet_registered(self) -> None:
        # Registration comes with the artifacts, the evidence page and the grid row.
        assert study.STUDY.slug not in {record.slug for record in registered()}

    def test_the_seeds_are_distinct(self) -> None:
        others = registered()
        assert study.SEED not in {record.seed for record in others}
        assert study.RESAMPLING_SEED not in {record.resampling_seed for record in others}
        seeds = [
            replicate_seed(study.STUDY, scenario, r)
            for scenario in study.SCENARIOS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, stream, r)
            for family, stream, *_ in properties.FIT_SETS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(50)
        }
        assert not foreign & set(seeds)
