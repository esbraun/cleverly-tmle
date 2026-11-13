"""The ``hazard-multinode`` declaration: truths, bounds, limits, sizes, pilot readings and seeds.

Each truth is checked by an independent route (inverse weighting over the support of the
observed law, written here) and by a fit on the law's exact sample; each efficiency bound
against the influence curve that exact-sample fit reports; the robustness limits and the wrong
learner against the exact sample; and the replicate counts against the shared sizing rules.
The pilot readings come from the committed pilot artifact, and the tests that read it skip
until it is committed.  Nothing here reads a study run.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cleverly.longitudinal import LTMLE
from cleverly.utils.bounds import DEFAULT_LTMLE_G_BOUNDS
from tests.studies import canonical_hazard_multinode as study
from tests.studies import hazard_multinode_law as law
from tests.studies import hazard_multinode_pilot as pilot
from tests.studies import hazard_multinode_properties as properties
from tests.studies import mtp_study_tools
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins
CONTRAST = study.CONTRAST_LABEL
#: How close an exact-sample fit comes to the truth.  The walk stops at ``tol=1e-10`` on the
#: relative score and the logistic mechanisms are solved to ``1e-10``, which leaves about
#: ``1e-8`` on the risks.
EXACT = 1e-7


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _scale(label: str, n: int) -> float:
    return properties.EFFICIENCY_SD[label] / np.sqrt(n)


def _exact_fit(fit_label: str, configuration: str = "both_correct") -> Any:
    """The fit of one fit label on its law's exact sample, each outcome weighted by its mass."""
    current = study.law_of(fit_label)
    _, _, msm = study.FITS[fit_label]
    target: dict[str, Any] = {"msm": law.working_model()} if msm else {"reference": "never"}
    return _quiet(
        LTMLE(
            study.regimens(fit_label),
            n_folds=1,
            survival_model="hazard",
            targeting=study.algorithm(fit_label),
            tol=study.TOL,
            g_bounds=study.G_BOUNDS,
            random_state=0,
            **target,
            **study.learners(configuration),
        ).fit,
        current.exact_frame(),
        **current.fit_columns(weights="wt"),
    )


@pytest.fixture(scope="module")
def exact_fits() -> dict[tuple[str, str], Any]:
    """The exact-sample fit of every fit label and robustness configuration."""
    return {
        (fit, configuration): _exact_fit(fit, configuration)
        for fit in study.FITS
        for configuration in properties.CONFIGURATIONS
    }


# ---------------------------------------------------------------------------- independent routes


def _weighted_risk(current: law.Law, label: str, horizon: int) -> float:
    """:math:`\\psi_r(t)` by inverse weighting over the support of the observed law.

    Each observed outcome with an event at node :math:`k \\le t` contributes its mass times
    :math:`\\prod_{s \\le k} q_{r,s} / (g_s c_s)` at its own arms.  This reads the observed
    law, not the hazards, so it shares no step with :meth:`~law.Law.risk`.
    """
    total = 0.0
    for row, mass in current.support():
        w1, w2 = int(row[0]), int(row[1])
        weight, previous = 1.0, 0
        for node in range(1, horizon + 1):
            base = 2 + 3 * (node - 1)
            if np.isnan(row[base]) or row[base + 1] == 0.0:
                break
            a = int(row[base])
            q = law.treat_probability(label, node, w1, w2, previous)
            weight *= (q if a == 1 else 1.0 - q) / current.g(node, a, w1, w2, previous)
            weight /= current.retained(w1, w2, a, previous)
            if row[base + 2] == 1.0:
                total += mass * weight
                break
            previous = a
    return total


class TestTheTruths:
    @pytest.mark.parametrize("law_name", list(law.LAWS))
    def test_the_enumerated_risk_is_the_weighted_risk(self, law_name: str) -> None:
        current = law.LAWS[law_name]
        for label in law.LABELS:
            for horizon in range(1, current.n_times + 1):
                assert current.risk(label, horizon) == pytest.approx(
                    _weighted_risk(current, label, horizon), abs=1e-10
                ), (label, horizon)

    def test_the_static_path_risks_are_the_weighted_risks(self) -> None:
        for label, horizon in law.msm_cells(law.BASE):
            assert law.BASE.risk(label, horizon) == pytest.approx(
                _weighted_risk(law.BASE, label, horizon), abs=1e-10
            )

    def test_the_exact_sample_fit_returns_every_truth(
        self, exact_fits: dict[tuple[str, str], Any]
    ) -> None:
        for fit in study.FITS:
            result = exact_fits[(fit, "both_correct")]
            for name, truth in study.TRUTH[fit].items():
                assert result[name].psi == pytest.approx(truth, abs=EXACT), (fit, name)

    def test_the_declared_truths(self) -> None:
        expected = {
            "base_contrast": -0.218028,
            "base_rule": 0.373959,
            "base_policy": 0.398387,
            "base_stay": 0.408532,
            "five_contrast": -0.275896,
            "five_policy": 0.570281,
            "msm_dose": -0.500912,
            "msm_dose_W1": 0.253667,
        }
        for label, value in expected.items():
            assert properties.TRUTHS[label] == pytest.approx(value, abs=1e-6), label

    def test_the_policies_differ_from_the_static_arms(self) -> None:
        """The rule, the policy and "stay" each lie away from always and never."""
        for current in law.LAWS.values():
            last = current.n_times
            static = (current.risk("always", last), current.risk("never", last))
            for label in ("rule", "policy", "stay"):
                assert min(abs(current.risk(label, last) - value) for value in static) > 0.02

    @pytest.mark.parametrize(("law_name", "expected"), [("base", 0.0196), ("five", 0.0110)])
    def test_the_bounds_never_bind(self, law_name: str, expected: float) -> None:
        """Every path keeps its cumulative treatment and retention probability ten times above
        the declared bound, so no clipping touches a fit.  The shipped default would sit
        within a factor of two of the smallest, which is why the study declares its own."""
        current = law.LAWS[law_name]
        smallest = 1.0
        for (w1, w2), path in itertools.product(
            law.STRATA, itertools.product((0, 1), repeat=current.n_times)
        ):
            product, previous = 1.0, 0
            for node, a in enumerate(path, start=1):
                product *= current.g(node, a, w1, w2, previous)
                product *= current.retained(w1, w2, a, previous)
                previous = a
            smallest = min(smallest, product)
        assert smallest == pytest.approx(expected, abs=1e-4)
        assert smallest >= 10.0 * study.G_BOUNDS
        assert smallest < 2.0 * DEFAULT_LTMLE_G_BOUNDS[0]


class TestTheBounds:
    @pytest.mark.parametrize("fit", ["base", "five"])
    def test_the_reported_curve_is_the_iterated_mean_curve(
        self, exact_fits: dict[tuple[str, str], Any], fit: str
    ) -> None:
        """On the exact sample every fitted factor is the truth, so the reported influence curve
        is the efficient one; the fit reports it per row as ``n w`` times the unit's curve."""
        current = study.law_of(fit)
        result = exact_fits[(fit, "both_correct")]
        frame = current.exact_frame()
        weights = frame["wt"].to_numpy()
        rows = [tuple(row) for row in frame[current.columns()].to_numpy()]
        last = current.n_times
        for label in study.FITS[fit][1]:
            name = study.risk_name(label, last)
            reported = np.asarray(result.influence_curves[name]) / (len(rows) * weights)
            expected = np.array([current.curve(label, last, row) for row in rows])
            expected -= current.risk(label, last)
            np.testing.assert_allclose(reported, expected, rtol=0.0, atol=1e-5)

    def test_the_bounds_are_the_reported_curves(
        self, exact_fits: dict[tuple[str, str], Any]
    ) -> None:
        for fit in study.FITS:
            result = exact_fits[(fit, "both_correct")]
            weights = study.law_of(fit).exact_frame()["wt"].to_numpy()
            for label, name in properties.COEFFICIENTS[fit]:
                curve = np.asarray(result.influence_curves[name]) / (len(weights) * weights)
                spread = float(np.sqrt(np.sum(weights * curve**2)))
                assert spread == pytest.approx(properties.EFFICIENCY_SD[label], rel=1e-6), label

    def test_the_declared_bounds(self) -> None:
        expected = {
            "base_contrast": 1.441826,
            "base_rule": 0.789002,
            "base_policy": 0.559790,
            "base_stay": 0.749446,
            "five_contrast": 2.528426,
            "five_policy": 0.621779,
            "msm_dose": 3.128782,
            "msm_dose_W1": 3.248549,
        }
        for label, value in expected.items():
            assert properties.EFFICIENCY_SD[label] == pytest.approx(value, abs=1e-6), label


# ---------------------------------------------------------------------------- learners


class _Recorder(law.HistoryLogistic):
    """The correct learner, recording the width and last column of each design it fits."""

    seen: list[tuple[int, np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Recorder:
        matrix = np.asarray(X, dtype=float)
        _Recorder.seen.append((matrix.shape[1], matrix[:, -1].copy()))
        return super().fit(X, y, sample_weight)


def test_the_hazard_design_ends_with_the_node_arm() -> None:
    """The wrong learner reads the last column: at node ``k`` the design is ``[W, A_1..A_k]``."""
    current = law.BASE
    frame = current.draw(400, 0)
    _Recorder.seen = []
    _quiet(
        LTMLE(
            {"always": (1, 1, 1), "never": (0, 0, 0)},
            n_folds=1,
            survival_model="hazard",
            random_state=0,
            outcome_learner=_Recorder(),
            treatment_learner=law.HistoryLogistic(),
            censoring_learner=law.HistoryLogistic(),
        ).fit,
        frame,
        **current.fit_columns(),
    )
    widths = sorted({width for width, _ in _Recorder.seen})
    assert widths == [3, 4, 5]
    for width, column in _Recorder.seen:
        node = width - 2
        at_risk = frame.loc[frame[f"C{node}"] == 1.0, f"A{node}"].to_numpy()
        assert set(np.unique(column)) <= {0.0, 1.0}
        assert column.size == at_risk.size and np.array_equal(np.sort(column), np.sort(at_risk))


@pytest.mark.parametrize(
    ("labels", "widths"),
    [(("always", "never", "rule"), {2}), (("policy",), {3, 4, 5}), (("stay",), {3})],
)
def test_the_iterated_mean_design_starts_with_w(labels: tuple[str, ...], widths: set) -> None:
    """:class:`~properties.NoWQuasiBinomial` drops the first two columns, which are ``W``.

    A static path and the rule regress on ``[W1, W2]`` among the units that follow them; the
    policy and "stay" add binary arm columns after ``W``.
    """
    from tests.studies.fractional_glm import QuasiBinomialGLM

    seen: list[np.ndarray] = []

    class Recorder(QuasiBinomialGLM):
        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Recorder:
            seen.append(np.asarray(X, dtype=float))
            return super().fit(X, y, sample_weight=sample_weight)

    current = law.BASE
    frame = current.draw(400, 0)
    _quiet(
        LTMLE(
            law.regimens(labels, current.n_times),
            n_folds=1,
            survival_model="iterated_mean",
            random_state=0,
            outcome_learner=Recorder(),
            pseudo_learner=Recorder(),
            treatment_learner=law.HistoryLogistic(),
            censoring_learner=law.HistoryLogistic(),
        ).fit,
        frame,
        **current.fit_columns(),
    )
    assert {matrix.shape[1] for matrix in seen} == widths
    for matrix in seen:
        assert set(np.unique(matrix[:, 0])) == {0.0, 1.0}
        assert set(np.unique(matrix[:, 1])) == {0.0, 1.0, 2.0, 3.0}
        assert set(np.unique(matrix[:, 2:])) <= {0.0, 1.0}


# ---------------------------------------------------------------------------- robustness


class TestTheRobustnessCells:
    def test_every_positive_limit_is_the_truth(
        self, exact_fits: dict[tuple[str, str], Any]
    ) -> None:
        for fit in study.FITS:
            for configuration in ("outcome_correct", "mechanism_correct"):
                result = exact_fits[(fit, configuration)]
                for label, name in properties.COEFFICIENTS[fit]:
                    assert result[name].psi == pytest.approx(
                        properties.TRUTHS[label], abs=EXACT
                    ), (fit, configuration, label)

    def test_the_both_wrong_control_is_discriminated(
        self, exact_fits: dict[tuple[str, str], Any]
    ) -> None:
        result = exact_fits[("base", "both_wrong")]
        gap = result[properties.ESTIMAND[CONTRAST]].psi - properties.TRUTHS[CONTRAST]
        n = properties.DOUBLE_ROBUST_N
        assert gap / _scale(CONTRAST, n) == pytest.approx(3.070, abs=1e-3)
        for spread in (_scale(CONTRAST, n), 1.5 * _scale(CONTRAST, n)):
            probability = sizing.bias_discriminated_probability(
                MARGINS, gap / spread, properties.DOUBLE_ROBUST_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(2)


# ---------------------------------------------------------------------------- wrong learner


PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()


def _pilot(label: str) -> pd.DataFrame:
    return PILOT.loc[PILOT["label"] == label] if not PILOT.empty else PILOT


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
#: The wrong hazard learner's plug-in gap, in efficiency SEs at n = 2,000, by cell label.
WRONG_GAPS = {
    "base_contrast": 3.541,
    "base_rule": 0.457,
    "base_policy": -1.012,
    "base_stay": -0.962,
    "five_contrast": 1.822,
    "five_policy": -0.701,
    "msm_dose": 4.643,
    "msm_dose_W1": -3.492,
}


def _tolerated_bias(replicates: int) -> float:
    """The largest |standardized bias| at which a positive arm passes with ``POSITIVE_PASS``."""
    z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
    return MARGINS.standardized_bias - (z + float(norm.ppf(POSITIVE_PASS))) / np.sqrt(replicates)


def _wrong_learner_errors(label: str) -> np.ndarray:
    """The pilot's targeted errors of one cell label under the wrong hazard learner.

    On ``base`` they pool ``fits_differ_mechanism_correct``, whose hazard fit is the same
    wrong-learner fit on its own seeds (as plan X70 rev 2.7 pooled them for ``hazard-mtp``).
    """
    fit = properties.FIT_OF[label]
    name = properties.ESTIMAND[label]
    rows = _pilot(f"wrong_learner_{fit}")
    if rows.empty:
        pytest.skip("the wrong-learner pilot has not been committed")
    rows = rows.loc[rows["estimand"] == name]
    assert rows["refusal"].fillna("").eq("").all()
    assert len(rows) == pilot.WRONG_LEARNER_DRAWS
    values = rows["wrong_targeted"].to_numpy(dtype=float)
    if fit == "base":
        paired = _pilot("fits_differ_mechanism_correct")
        paired = paired.loc[paired["estimand"] == name]
        assert len(paired) == pilot.MECHANISM_DIFFER_DRAWS
        values = np.concatenate([values, paired["estimate"].to_numpy(dtype=float)])
    return values - properties.TRUTHS[label]


def _needed(label: str) -> int:
    """The plan X70 rev 2.7 count of one wrong-learner arm, from the pilot."""
    error = _wrong_learner_errors(label)
    standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
    needed = int(properties.DOUBLE_ROBUST_REPLICATES)
    while sizing.bias_equivalent_probability(MARGINS, standardized, needed) < POSITIVE_PASS:
        needed += 400
    assert needed <= 4_000, (label, standardized, needed)
    return needed


class TestTheWrongHazardLearner:
    def test_the_wrong_hazard_learner_rule(self) -> None:
        """The plug-in limit of the necessity contrast lies at least 3 efficiency SEs away."""
        for label, gap in WRONG_GAPS.items():
            bias = (properties.wrong_limit(label) - properties.TRUTHS[label]) / _scale(label, 2_000)
            assert bias == pytest.approx(gap, abs=1e-3), label
        assert WRONG_GAPS[CONTRAST] >= 3.0

    def test_the_estimator_plug_in_is_the_limit(
        self, exact_fits: dict[tuple[str, str], Any]
    ) -> None:
        for fit in study.FITS:
            result = exact_fits[(fit, "mechanism_correct")]
            for label, name in properties.COEFFICIENTS[fit]:
                assert properties.plug_in(result, name) == pytest.approx(
                    properties.wrong_limit(label), abs=1e-7
                ), label

    @pytest.mark.parametrize("fit", pilot.WRONG_LEARNER_FITS)
    def test_the_pilot_plug_in_varies(self, fit: str) -> None:
        rows = _pilot(f"wrong_learner_{fit}")
        if rows.empty:
            pytest.skip("the wrong-learner pilot has not been committed")
        for label, name in properties.COEFFICIENTS[fit]:
            arm = rows.loc[rows["estimand"] == name]
            assert arm["refusal"].fillna("").eq("").all()
            assert len(arm) == pilot.WRONG_LEARNER_DRAWS
            values = arm["wrong_plug_in"].to_numpy(dtype=float)
            assert np.var(values, ddof=1) > 0.0
            error = np.std(values, ddof=1) / np.sqrt(values.size)
            assert abs(values.mean() - properties.wrong_limit(label)) < 3.0 * error, label


# ---------------------------------------------------------------------------- sizes


class TestTheSizes:
    def test_the_power_cell_has_power(self) -> None:
        power = design_power(
            properties.TRUTHS[CONTRAST], properties.EFFICIENCY_SD[CONTRAST], properties.POWER_N
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

    def test_the_untargeted_control_is_discriminated(self) -> None:
        """The untargeted arm misses by the wrong learner's plug-in gap, at 1.0 and 1.5 spreads."""
        gap = WRONG_GAPS[CONTRAST]
        for spread in (1.0, 1.5):
            probability = sizing.bias_discriminated_probability(
                MARGINS, gap / spread, properties.NECESSITY_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(2)

    @pytest.mark.skipif(CALIBRATION_PILOTED, reason="the committed calibration pilot sizes R")
    def test_the_calibration_replicates_at_a_calibrated_pilot(self) -> None:
        """The arm-pooled rule at a pilot of the declared size that reads a ratio of one.

        The committed pilot replaces this reading once it exists, and this test then skips.
        """
        arms = sum(len(properties.COEFFICIENTS[fit]) for fit in properties.CALIBRATION_FITS)
        assert arms == 8
        replicates, cell, family = mtp_study_tools.smallest_replicates(
            1.0, sizing.CALIBRATION_PILOT_DRAWS, arms
        )
        assert replicates == properties.CALIBRATION_REPLICATES
        assert cell >= mtp_study_tools.CELL_TARGET and family >= mtp_study_tools.FAMILY_TARGET

    @staticmethod
    def _calibration_rows(fit: str) -> pd.DataFrame:
        rows = _pilot(f"calibration_{fit}")
        if rows.empty:
            pytest.skip(f"the calibration pilot of {fit} has not been committed")
        return rows

    def test_the_calibration_replicates_follow_the_pilot(self) -> None:
        ratios = []
        for fit in properties.CALIBRATION_FITS:
            rows = self._calibration_rows(fit)
            assert rows["refusal"].fillna("").eq("").all()
            for _, name in properties.COEFFICIENTS[fit]:
                arm = rows.loc[rows["estimand"] == name]
                assert len(arm) == sizing.CALIBRATION_PILOT_DRAWS
                ratios.append(float(arm["std_error"].mean() / arm["estimate"].std(ddof=1)))
        reading = mtp_study_tools.replicates_from_pilot(ratios, sizing.CALIBRATION_PILOT_DRAWS)
        assert not reading["suspect"]
        assert all(reading["screened"])
        assert reading["R"] == properties.CALIBRATION_REPLICATES

    def test_every_positive_arm_passes_at_its_exact_limit(self) -> None:
        """Each positive bias arm passes with probability at least 0.99 at standardized bias 0.

        Every positive limit is the truth on the exact sample (:class:`TestTheRobustnessCells`);
        the finite-sample bias of the wrong-learner arms is read from their pilot.
        """
        for cell, replicates in POSITIVE_ARMS.items():
            probability = sizing.bias_equivalent_probability(MARGINS, 0.0, replicates)
            assert probability >= POSITIVE_PASS, (cell, probability)
        assert _tolerated_bias(properties.DOUBLE_ROBUST_REPLICATES) == pytest.approx(
            0.1085, abs=1e-3
        )
        assert _tolerated_bias(properties.RATE_REPLICATES) == pytest.approx(0.0767, abs=1e-3)

    @pytest.mark.parametrize("fit", pilot.WRONG_LEARNER_FITS)
    def test_the_wrong_learner_arms_are_sized_by_the_pilot_rule(self, fit: str) -> None:
        """Each fit set that fits the wrong hazard learner takes the pre-committed size.

        The rule of plan X70 rev 2.7, fixed before any reading: an arm needs the larger of
        :data:`~tests.studies.hazard_multinode_properties.DOUBLE_ROBUST_REPLICATES` and the
        smallest multiple of 400 that passes with :data:`POSITIVE_PASS` at the pilot's
        |standardized bias| plus one of its standard errors, with no cap; a reading that needs
        more than 4,000 stops the declaration for a diagnosis.  A ``mechanism_correct`` fit set
        fills every arm of its fit label, so it takes the largest need of those arms; the
        necessity fit set takes the need of ``base_contrast``.
        """
        needs = {label: _needed(label) for label, _ in properties.COEFFICIENTS[fit]}
        assert properties.MECHANISM_REPLICATES[fit] == max(needs.values()), needs
        for label, _ in properties.COEFFICIENTS[fit]:
            assert POSITIVE_ARMS[f"{label}__mechanism_correct"] == max(needs.values())
        if fit == "base":
            assert properties.NECESSITY_REPLICATES == needs[CONTRAST]
            assert POSITIVE_ARMS[f"{CONTRAST}__targeted"] == needs[CONTRAST]

    def test_the_positive_arms_follow_the_calibration_pilot(self) -> None:
        """The ``both_correct`` and ``outcome_correct`` arms and the claimed rungs, at the pilot."""
        rows = self._calibration_rows("base")
        for cell_label, name in properties.COEFFICIENTS["base"]:
            arm = rows.loc[rows["estimand"] == name]
            error = (arm["estimate"] - arm["truth"]).to_numpy(dtype=float)
            standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
            for cell, replicates in POSITIVE_ARMS.items():
                if cell.startswith(f"{cell_label}__") and cell.endswith(
                    ("__both_correct", "__outcome_correct", "__n_2000", "__n_8000")
                ):
                    probability = sizing.bias_equivalent_probability(
                        MARGINS, standardized, replicates
                    )
                    assert probability >= POSITIVE_PASS, (cell, probability)

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` on the calibration pilots.

        The calibration pilot of each fit label draws its scenario at the primary n with the
        primary fit.  One implementation, clauses (a) to (c) on the covered counts and clause
        (d) on the standardized biases, one group per scenario.
        """
        covered: dict[str, int] = {}
        groups = []
        for scenario in study.SCENARIOS:
            rows = self._calibration_rows(scenario)
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


# ---------------------------------------------------------------------------- pilot readings


def _paired_ratios(
    rows: pd.DataFrame, fit: str, column: str = "iterated_estimate"
) -> dict[str, float]:
    """The SD of hazard minus iterated-mean over the efficiency-bound SE, by cell label."""
    ratios = {}
    for label, name in properties.COEFFICIENTS[fit]:
        arm = rows.loc[rows["estimand"] == name]
        difference = arm["estimate"] - arm[column]
        ratios[label] = float(difference.std(ddof=1)) / _scale(label, pilot.PILOT_N)
    return ratios


@pytest.mark.parametrize("fit", study.DIFFERENCE_FITS)
class TestTheFitsDifferPilot:
    """The ``iterated_mean_difference`` readings, reported with no floor.

    :func:`~tests.studies.hazard_multinode_properties.fits_differ_reading` states the paired
    ratios and the decreasing share with its interval.  X70 reports the agreement of the two
    fits instead of requiring a gap (the owner's decision of 2026-11-13).
    """

    @staticmethod
    def _rows(fit: str) -> pd.DataFrame:
        rows = _pilot(f"fits_differ_{fit}")
        if rows.empty:
            pytest.skip("the fits-differ pilot has not been committed")
        return rows

    def test_the_pilot_lost_no_fit(self, fit: str) -> None:
        rows = self._rows(fit)
        assert rows["refusal"].fillna("").eq("").all()
        assert rows["replicate"].nunique() == pilot.FITS_DIFFER_DRAWS

    def test_the_reading_is_complete_and_finite(self, fit: str) -> None:
        rows = self._rows(fit)
        decreases = rows.loc[rows["estimand"] == "largest_decrease", "estimate"]
        assert len(decreases) == pilot.FITS_DIFFER_DRAWS
        assert np.isfinite(decreases.to_numpy(dtype=float)).all()
        for column in ("estimate", "iterated_estimate", "isotonic_estimate"):
            values = rows.loc[rows["estimand"] != "largest_decrease", column]
            assert np.isfinite(values.to_numpy(dtype=float)).all(), column
        ratios = _paired_ratios(rows, fit)
        ratios |= {
            f"{label} (isotonic)": value
            for label, value in _paired_ratios(rows, fit, "isotonic_estimate").items()
        }
        reading = properties.fits_differ_reading(
            ratios, int((decreases > 0.0).sum()), len(decreases)
        )
        assert all(np.isfinite(value) and value > 0.0 for value in reading["ratios"].values())
        low, high = reading["share_interval"]
        assert 0.0 <= low <= reading["share"] <= high <= 1.0
        print(reading["text"])


@pytest.mark.skipif(
    _pilot("fits_differ_mechanism_correct").empty, reason="the pilot has not been committed"
)
def test_the_mechanism_correct_ratios_are_reported() -> None:
    rows = _pilot("fits_differ_mechanism_correct")
    assert rows["refusal"].fillna("").eq("").all()
    ratios = _paired_ratios(rows, "base")
    assert all(np.isfinite(value) and value > 0.0 for value in ratios.values()), ratios


def test_the_agreement_reading_carries_no_floor() -> None:
    """The reading states the ratios and the decreasing share with its interval, and no floor."""
    reading = properties.fits_differ_reading({"base_contrast": 0.03, "base_policy": 0.02}, 0, 200)
    assert not {key for key in reading if key.endswith("passes")}
    assert reading["largest_ratio"] == 0.03 and reading["contrast_ratio"] == 0.03
    assert reading["share_interval"] == (0.0, pytest.approx(0.0261, abs=1e-4))
    assert "floor" in reading["text"] and "0.05" not in reading["text"]


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
        from tests.studies import canonical_hazard_msm, canonical_hazard_mtp

        # The other X70 studies are named whether or not they are registered yet: the union of
        # the records by slug holds each once either way.
        records = {record.slug: record for record in registered()}
        for other in (canonical_hazard_msm.STUDY, canonical_hazard_mtp.STUDY):
            records.setdefault(other.slug, other)
        others = [record for slug, record in records.items() if slug != study.STUDY.slug]
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
