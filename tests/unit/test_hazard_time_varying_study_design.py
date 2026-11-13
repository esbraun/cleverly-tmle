"""The ``hazard-time-varying`` declaration: truths, bounds, limits, sizes, pilot readings, seeds.

Each truth is checked by an independent route: a forward sum over the covariate paths written
here, and a fit on each law's exact sample.  Each efficiency bound is checked against the
influence curve of that exact-sample fit, which at the true nuisances is the efficient
influence function.  The robustness limits and the wrong learners are checked on the exact sample
and against a limit computed from the law alone, and the replicate counts against the shared
sizing rules.  The pilot readings come from the committed pilot artifact, and the tests that read
it skip until it is committed.  Nothing here reads a study run.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

import cleverly.longitudinal.estimator as estimator_module
from cleverly import _support
from cleverly._support_rows import LONGITUDINAL_ROWS
from cleverly.utils.bounds import DEFAULT_LTMLE_G_BOUNDS
from tests.studies import canonical_hazard_time_varying as study
from tests.studies import hazard_time_varying_law as law
from tests.studies import hazard_time_varying_pilot as pilot
from tests.studies import hazard_time_varying_properties as properties
from tests.studies import mtp_study_tools
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins
CONTRAST = study.CONTRAST_LABEL
#: How close an exact-sample fit with a correct nuisance comes to the truth.  The walks stop at
#: ``tol=1e-10`` on the relative score and the logistic fits are solved to ``1e-10``.
EXACT = 1e-7
#: The exact-law gaps, in efficiency SEs at n = 2,000, of the plug-in whose hazard and
#: continuation omit ``L`` and of the both-wrong targeted fit, on ``one_cause_contrast``.  The
#: two read one bias: with every factor a function of ``(W, A)`` the walk barely moves.
WRONG_GAP = 4.225
BOTH_WRONG_GAP = 4.216
#: The untargeted plug-in gap of every wrong-learner robustness arm's contrast, in efficiency
#: SEs at n = 2,000 on the exact sample.  ``continuation_wrong`` keeps the hazard correct, so
#: its targeted fit reaches the truth only through the covariate term of the walk.
PLUG_IN_GAPS = {
    ("one_cause", "mechanism_correct"): 4.225,
    ("one_cause", "continuation_wrong"): 1.369,
    ("competing", "mechanism_correct"): 3.203,
    ("three_arm", "mechanism_correct"): 3.626,
}


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _scale(label: str, n: int) -> float:
    return properties.EFFICIENCY_SD[label] / np.sqrt(n)


def _exact_fit(fit_label: str, configuration: str = "both_correct") -> Any:
    """The fit of one fit label's law on its exact sample, each outcome weighted by its mass."""
    frame = study.law_of(fit_label).exact_frame()
    return _quiet(study.fit, fit_label, frame, configuration=configuration, weights="wt")


@pytest.fixture(scope="module")
def exact_fits() -> dict[tuple[str, str], Any]:
    """The exact-sample fit of every robustness arm, by ``(fit label, configuration)``."""
    return {arm: _exact_fit(*arm) for arm in properties.DOUBLE_ROBUST_ARMS}


# ---------------------------------------------------------------------------- independent routes


def _forward_truth(current: law.Law, regimen: Any, horizon: int, cause: int) -> float:
    """``E Q_1(W, d(A))`` by a forward sum over the covariate paths, apart from the law module.

    Each path of ``L_2, ..., L_t`` carries its probability given ``(W1, a)``, and along it the
    cause's incidence accumulates as survival times the cause's hazard at each node.
    """
    cov = current.covariate
    total = 0.0
    for w1, w2 in law.STRATA:
        g = current.g(w1, w2)
        for natural_level in range(current.arms):
            a = regimen(natural_level)
            for path in itertools.product((1, -1), repeat=horizon - 1):
                weight = 1.0
                previous = 0.0
                for value in path:
                    p = float(
                        expit(cov.intercept + cov.w1 * w1 + cov.arm[a] + cov.previous * previous)
                    )
                    weight *= p if value == 1 else 1.0 - p
                    previous = 1.0 if value == 1 else 0.0
                survival, incidence = 1.0, 0.0
                for node in range(1, horizon + 1):
                    flag = 0.0 if node == 1 else (1.0 if path[node - 2] == 1 else 0.0)
                    taken, hazards = 0.0, []
                    for factor in current.hazards:
                        eta = (
                            factor.intercept
                            + factor.trend * (node - 1)
                            + factor.arm[a]
                            + factor.w1 * w1
                            + factor.w2 * w2
                            + factor.l * flag
                        )
                        hazards.append(float(expit(eta)) * (1.0 - taken))
                        taken += hazards[-1]
                    incidence += survival * hazards[cause - 1]
                    survival *= 1.0 - taken
                total += law.MASS * g[natural_level] * weight * incidence
    return total


# ---------------------------------------------------------------------------- truths


class TestTheTruths:
    @pytest.mark.parametrize("law_name", list(law.LAWS))
    def test_the_backward_truth_is_the_forward_sum(self, law_name: str) -> None:
        current = law.LAWS[law_name]
        for label, regimen in study.maps(law_name).items():
            for cause in range(1, current.causes + 1):
                for horizon in range(1, current.n_times + 1):
                    assert current.truth(regimen, horizon, cause) == pytest.approx(
                        _forward_truth(current, regimen, horizon, cause), abs=1e-14
                    ), (label, cause, horizon)

    @pytest.mark.parametrize("fit_label", study.SCENARIOS)
    def test_the_exact_sample_fit_returns_every_truth(self, fit_label: str) -> None:
        result = _exact_fit(fit_label)
        for name, value in study.truths(study.FITS[fit_label][0]).items():
            assert result[name].psi == pytest.approx(value, abs=EXACT), name

    def test_the_support_is_the_law(self) -> None:
        for current in law.LAWS.values():
            assert current.probs.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.all(current.probs > 0.0)
            assert len(set(current.support)) == len(current.support)

    def test_a_large_draw_matches_the_support_jointly(self) -> None:
        """The support-point frequencies of 200,000 draws against the exact probabilities.

        Each unit is mapped to the support point it realises, and a chi-square test over the
        points with expected count at least 5 (the rest pooled) must not reject at 1e-4.  A
        draw that read ``L_{k-1}`` where the law reads ``L_k`` keeps the marginal means close
        and fails here.
        """
        from collections import Counter

        from scipy.stats import chisquare

        n = 200_000
        for current in law.LAWS.values():
            frame = current.draw(n, stream_seed(study.STUDY, "design_joint", current.name, 0))
            counts = Counter(law._keys(current, frame))
            assert set(counts) <= set(current.support)
            expected = current.probs * n
            observed = np.array([counts.get(point, 0) for point in current.support], dtype=float)
            large = expected >= 5.0
            pooled_expected = np.append(expected[large], expected[~large].sum())
            pooled_observed = np.append(observed[large], observed[~large].sum())
            assert chisquare(pooled_observed, pooled_expected).pvalue > 1e-4, current.name

    def test_a_large_draw_matches_the_support(self) -> None:
        """Every outcome column's mean on 200,000 draws lies within 4 SE of its exact mean."""
        for current in law.LAWS.values():
            exact = current.exact_frame()
            frame = current.draw(200_000, stream_seed(study.STUDY, "design_draw", current.name, 0))
            assert list(frame.columns) == [name for name in exact.columns if name != "wt"]
            for name in frame.columns:
                observed = frame[name].notna()
                kept = exact[name].notna()
                assert observed.mean() == pytest.approx(
                    exact.loc[kept, "wt"].sum(), abs=4.0 * np.sqrt(0.25 / len(frame))
                ), name
                mean = float(np.average(exact.loc[kept, name], weights=exact.loc[kept, "wt"]))
                values = frame.loc[observed, name].to_numpy(dtype=float)
                spread = values.std() / np.sqrt(values.size)
                assert abs(values.mean() - mean) < 4.0 * spread + 1e-12, name

    def test_the_declared_truths(self) -> None:
        np.testing.assert_allclose(
            [
                properties.TRUTHS[f"{fit}_{arm}"]
                for fit in study.SCENARIOS
                for arm in properties.ARMS
            ],
            [-0.458239, 0.192996, -0.322846, 0.124564, -0.162513, 0.137545],
            atol=5e-6,
        )
        for arm in properties.ARMS:
            assert properties.TRUTHS[f"crossfit_{arm}"] == properties.TRUTHS[f"one_cause_{arm}"]
        for scenario in study.SCENARIOS:
            assert study.TRUTH[scenario] == study.truths(study.FITS[scenario][0])

    def test_the_curves_rise(self) -> None:
        for name, current in law.LAWS.items():
            for regimen in study.maps(name).values():
                for cause in range(1, current.causes + 1):
                    values = [
                        current.truth(regimen, t, cause) for t in range(1, current.n_times + 1)
                    ]
                    assert np.all(np.diff(values) > 0.0)

    def test_the_bounds_never_bind(self) -> None:
        """Positivity over every stratum, level and covariate path.

        The cumulative bound reads :math:`g(a \\mid W) \\prod_k c_k`, whose smallest value is
        above twice the default bound of 0.01 on every law.
        """
        expected = {
            "one_cause": {"g_min": 0.2891, "retained_min": 0.1361, "cumulative_min": 0.0394},
            "competing": {"g_min": 0.2891, "retained_min": 0.2230, "cumulative_min": 0.0644},
            "three_arm": {"g_min": 0.2237, "retained_min": 0.1361, "cumulative_min": 0.0314},
        }
        for name, values in expected.items():
            reading = law.positivity(law.LAWS[name])
            assert reading["cumulative_min"] > 2.0 * DEFAULT_LTMLE_G_BOUNDS[0]
            for key, value in values.items():
                assert reading[key] == pytest.approx(value, abs=1e-4), (name, key)


# ---------------------------------------------------------------------------- held cells


def _cell_masses(current: law.Law) -> dict[tuple[int, int, int], float]:
    """The mass of a cause-``j`` event at node ``k`` among units with treatment level ``a``.

    Keyed ``(a, k, j)``, over every unit followed and at risk at ``k``: the node-hazard cells
    the shipped fit holds when no unit of the cell has the event.
    """
    masses: dict[tuple[int, int, int], float] = {}
    for point, mass in zip(current.support, current.probs, strict=True):
        level = int(point[2])
        for node in range(1, current.n_times + 1):
            retained = current.value(point, "C", node)
            if retained is None:
                break
            if retained != 1:
                continue
            event = current.value(point, "Y", node)
            for cause in range(1, current.causes + 1):
                key = (level, node, cause)
                masses[key] = masses.get(key, 0.0) + (float(mass) if event == cause else 0.0)
    assert len(masses) == current.arms * current.n_times * current.causes
    return masses


def expected_held(current: law.Law, n: int) -> float:
    """The expected count of held node-hazard cells in one draw of ``n`` rows.

    A cell of mass :math:`v` holds no event with probability :math:`(1 - v)^n`; the sum over
    every cell bounds the chance that a draw holds one.  A continuation cell cannot be held:
    its soft labels come from logistic hazards and are positive.
    """
    return float(sum((1.0 - mass) ** n for mass in _cell_masses(current).values()))


def held_units() -> list[tuple[str, str, int, int]]:
    """Every set of draws the study and its pilot fit: ``(what, law, n, draws)``.

    The property fit sets at their own n and replicate count, the primary scenarios, and every
    pilot label (``ic_gap`` at each of its sizes).  A key that runs several fits of one draw
    counts once, since a held cell depends on the draw alone.
    """
    units = [
        (f"{family}/{stream}", study.FITS[fit][0], n, replicates)
        for family, stream, _, n, replicates, fit in properties.FIT_SETS
    ]
    units += [
        (f"primary/{scenario}", study.FITS[scenario][0], study.PRIMARY_N, study.PRIMARY_REPLICATES)
        for scenario in study.SCENARIOS
    ]
    units += [
        (f"pilot/{label}", study.FITS[fit][0], pilot.PILOT_N, draws)
        for label, (fit, draws) in pilot.PILOTS.items()
        if label != "ic_gap"
    ]
    sizes: dict[tuple[str, int], int] = {}
    for fit, n, _ in pilot.IC_GAP_JOBS:
        sizes[(fit, n)] = sizes.get((fit, n), 0) + 1
    units += [
        (f"pilot/ic_gap/{fit}/{n}", study.FITS[fit][0], n, draws)
        for (fit, n), draws in sizes.items()
    ]
    return units


class TestTheHeldCells:
    def test_the_held_fit_is_a_recorded_lost_replicate(self) -> None:
        """A draw that holds a cell raises a recordable ``DataError`` before any row reads it.

        At 80 rows of ``three_arm`` a held cell is all but certain (level 2 at node 1 has mass
        0.0065 per unit); the harness records a ``DataError`` as a lost replicate, and the pilot
        writes it as ``held``.
        """
        from cleverly.exceptions import DataError
        from cleverly.validation._failures import recordable

        assert expected_held(law.THREE_ARM, 80) > 0.5
        frame = law.THREE_ARM.draw(80, 0)
        with pytest.raises(study.HeldCellError) as caught:
            _quiet(study.fit_recorded, "three_arm", frame)
        assert isinstance(caught.value, DataError)
        assert recordable(caught.value)

    def test_the_pilot_writes_a_held_draw_as_held(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pilot, "PILOT_N", 40)
        rows = [_quiet(pilot._one, ("calibration_three_arm", r)) for r in range(4)]
        held = [found for found in rows if len(found) == 1 and found[0]["held"] == 1]
        assert held, "no small draw held a cell"
        assert all(found[0]["refusal"] == "" for found in held)

    def test_the_expected_held_fits_stay_within_each_allowance(self) -> None:
        """Over the whole study and its pilot, a held fit is rare and never exhausts a bound.

        Each set of draws may lose ``floor(0.01 R)`` replicates (the failure bound, and the
        pilot readings' own limit; a smoke-sized set may drop one, :func:`held_allowance`).  The
        expected held count of each set is ``R`` times :func:`expected_held` at its own n, its
        excess over the allowance is a Poisson tail, and the union of those tails stays below
        1e-3.  The summed expected count over the study and the pilot is recorded: about 0.05
        fits, nearly all on ``three_arm`` at n = 2,000 and ``one_cause`` at n = 1,000.
        """
        from scipy.stats import poisson

        per_fit = {(name, n): expected_held(law.LAWS[name], n) for _, name, n, _ in held_units()}
        total = 0.0
        tails = {}
        for what, name, n, draws in held_units():
            mean = draws * per_fit[(name, n)]
            total += mean
            tails[what] = float(poisson.sf(held_allowance(draws), mean))
        assert sum(tails.values()) < 1e-3, tails
        assert total == pytest.approx(HELD_TOTAL, abs=0.005), total
        assert total < 0.1
        assert per_fit[("three_arm", study.PRIMARY_N)] == pytest.approx(2.85e-6, rel=0.01)
        assert per_fit[("one_cause", min(properties.RATE_SIZES))] < 1e-5


#: The expected count of held fits over the study and its pilot (design review b1, finding 3).
HELD_TOTAL = 0.053


# ---------------------------------------------------------------------------- bounds


class TestTheEfficiencyBounds:
    @pytest.mark.parametrize("fit_label", study.SCENARIOS)
    def test_the_exact_fit_influence_curve_is_the_efficient_one(self, fit_label: str) -> None:
        """At the true nuisances the fit's influence curve is the efficient influence function.

        On the exact sample each row's curve is the support point's value times its weight over
        the mean weight, to the walk's tolerance.
        """
        current = study.law_of(fit_label)
        result = _exact_fit(fit_label)
        scale = current.probs * current.probs.size
        for label, name in properties.COEFFICIENTS[fit_label]:
            curve = np.asarray(result[name].influence_curve)
            np.testing.assert_allclose(
                curve, properties.target(label).curve(current) * scale, atol=1e-5
            )

    def test_the_efficient_influence_function_is_centred(self) -> None:
        for label in properties.TRUTHS:
            current = study.law_of(properties.FIT_OF[label])
            curve = properties.target(label).curve(current)
            assert float(np.sum(current.probs * curve)) == pytest.approx(0.0, abs=1e-12)

    def test_the_declared_bounds(self) -> None:
        np.testing.assert_allclose(
            [
                properties.EFFICIENCY_SD[f"{fit}_{arm}"]
                for fit in study.SCENARIOS
                for arm in properties.ARMS
            ],
            [1.06334, 0.55052, 1.02828, 0.45688, 0.43240, 0.44021],
            atol=5e-5,
        )

    def test_the_vectorized_curve_is_the_support_curve(self) -> None:
        for fit_label in study.SCENARIOS:
            current = study.law_of(fit_label)
            frame = current.exact_frame().drop(columns="wt")
            for label, _ in properties.COEFFICIENTS[fit_label]:
                aim = properties.target(label)
                np.testing.assert_allclose(
                    current.frame_curves(aim, frame), aim.curve(current), atol=1e-14
                )


# ---------------------------------------------------------------------------- the learners


class TestTheLearners:
    def test_last_l_keeps_the_current_covariate_and_no_l_drops_every_one(self) -> None:
        design = np.array([[0.0, 2.0, 1.0, -1.0, 1.0, 0.0], [1.0, 3.0, -1.0, 1.0, 0.0, 1.0]])
        assert list(law.LastL(law.CellMeans())._kept(design)) == [0, 1, 3, 4, 5]
        assert list(law.NoL(law.CellMeans())._kept(design)) == [0, 1, 4, 5]
        first = np.array([[0.0, 2.0, 1.0], [1.0, 3.0, 0.0]])
        assert list(law.LastL(law.CellMeans())._kept(first)) == [0, 1, 2]

    def test_the_untargeted_pass_of_the_correct_learners_is_the_truth(self) -> None:
        """The hazard and continuation learners are correctly specified on every node.

        With the walks skipped only the initial fits remain, so the plug-in reaches the truth on
        the exact sample only when each node's design keeps ``L_k``: a learner that kept an
        earlier ``L`` column instead would miss it.
        """
        for fit_label in study.SCENARIOS:
            current = study.law_of(fit_label)
            result = _quiet(
                study.untargeted_fit,
                fit_label,
                current.exact_frame(),
                configuration="both_correct",
                weights="wt",
            )
            for label, name in properties.COEFFICIENTS[fit_label]:
                assert result[name].psi == pytest.approx(properties.TRUTHS[label], abs=EXACT)


# ---------------------------------------------------------------------------- robustness


class TestTheRobustnessCells:
    def test_every_positive_limit_is_the_truth(
        self, exact_fits: dict[tuple[str, str], Any]
    ) -> None:
        for (fit, configuration), result in exact_fits.items():
            if configuration == "both_wrong":
                continue
            for label, name in properties.COEFFICIENTS[fit]:
                assert result[name].psi == pytest.approx(properties.TRUTHS[label], abs=EXACT), (
                    fit,
                    configuration,
                    label,
                )

    @pytest.mark.parametrize("arm", list(PLUG_IN_GAPS), ids=lambda arm: "-".join(arm))
    def test_each_wrong_learner_plug_in_misses(self, arm: tuple[str, str]) -> None:
        """Each wrong-learner arm's untargeted plug-in misses its contrast by the declared gap.

        So the targeted arm's correction is nonzero: on ``continuation_wrong`` the hazard is
        correct, the hazard term of the walk is zero, and the covariate term carries the whole
        1.37 SE (design review b1, finding 1); a dropped or wrongly signed continuation
        fluctuation would leave the estimate near the plug-in.
        """
        fit, configuration = arm
        label = f"{fit}_contrast"
        result = _quiet(
            study.untargeted_fit,
            fit,
            study.law_of(fit).exact_frame(),
            configuration=configuration,
            weights="wt",
        )
        gap = (result[properties.ESTIMAND[label]].psi - properties.TRUTHS[label]) / _scale(
            label, properties.DOUBLE_ROBUST_N
        )
        assert gap == pytest.approx(PLUG_IN_GAPS[arm], abs=1e-3)
        assert set(PLUG_IN_GAPS) == set(properties.WRONG_LEARNER_ARMS)

    def test_the_both_wrong_control_is_discriminated(
        self, exact_fits: dict[tuple[str, str], Any]
    ) -> None:
        """The both-wrong control misses by 4.2 SEs, the untargeted plug-in's own bias.

        Every factor the walk reads is then a function of ``(W, A)``, which the initial fit
        nearly saturates, so the walk barely moves: the control and the untargeted necessity arm
        read one bias, not two.
        """
        result = exact_fits[("one_cause", "both_wrong")]
        gap = result[properties.ESTIMAND[CONTRAST]].psi - properties.TRUTHS[CONTRAST]
        n = properties.DOUBLE_ROBUST_N
        assert gap / _scale(CONTRAST, n) == pytest.approx(BOTH_WRONG_GAP, abs=1e-3)
        assert abs(BOTH_WRONG_GAP - WRONG_GAP) < 0.01
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
#: The pilot label groups whose readings are read together.  A group with some labels committed
#: and others not fails, naming the missing labels; a group with none committed skips.
PILOT_GROUPS = {
    "calibration": tuple(f"calibration_{fit}" for fit in properties.CALIBRATION_FITS),
    "wrong learner": (*pilot.WRONG_LEARNER_LABELS, "fits_differ_mechanism_correct"),
}
#: A pilot label below this many draws is a smoke that sets no count.
SMOKE_DRAWS = 100


def held_allowance(draws: int) -> int:
    """The held draws a pilot label may drop: the failure bound's ``floor(0.01 R)``.

    A label under :data:`SMOKE_DRAWS` draws may drop one, as ``hazard-multinode`` allows: a
    held cell depends on the draw, not on the configuration.
    """
    strict = int(np.floor(properties.FAILURE_BOUND * draws))
    return max(1, strict) if draws < SMOKE_DRAWS else strict


def _group(name: str) -> None:
    """Fail when only some labels of a pilot group are committed; skip when none is."""
    labels = PILOT_GROUPS[name]
    missing = [label for label in labels if _pilot(label).empty]
    if len(missing) == len(labels):
        pytest.skip(f"the {name} pilot has not been committed")
    if missing:
        pytest.fail(
            f"only part of the {name} pilot is committed; missing {missing}. Its readings are "
            "read together, so commit the rest before reading any"
        )


def _kept(label: str) -> pd.DataFrame:
    """One pilot label's rows without its held draws, which the declared run would lose.

    No draw may be refused, every requested draw is kept or held, and the held draws stay
    within :func:`held_allowance`.  The caller has checked the label's group.
    """
    rows = _pilot(label)
    assert not rows.empty, label
    assert rows["refusal"].fillna("").eq("").all(), label
    draws = pilot.PILOTS[label][1]
    assert rows["replicate"].nunique() == draws, label
    held = rows.loc[rows["held"] == 1, "replicate"]
    assert held.nunique() <= held_allowance(draws), (label, held.nunique())
    return rows.loc[rows["held"] == 0]


#: The pass probability each positive bias arm needs (design review d).
POSITIVE_PASS = 0.99
#: Every positive arm whose verdict reads a bias interval, with its replicates.
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


def _wrong_learner_errors(arm: tuple[str, str]) -> np.ndarray:
    """The pilot's targeted contrast errors of one wrong-learner arm at n = 2,000.

    The ``one_cause`` ``mechanism_correct`` arm pools ``fits_differ_mechanism_correct``, whose
    hazard fit is the same wrong-learner fit on its own seeds (plan X70 rev 2.7).
    """
    _group("wrong learner")
    fit, _ = arm
    label = f"{fit}_contrast"
    name = properties.ESTIMAND[label]
    rows = _kept(properties.WRONG_LEARNER_ARMS[arm])
    values = rows.loc[rows["estimand"] == name, "wrong_targeted"].to_numpy(dtype=float)
    if arm == properties.NECESSITY_ARM:
        paired = _kept("fits_differ_mechanism_correct")
        values = np.concatenate(
            [values, paired.loc[paired["estimand"] == name, "estimate"].to_numpy(dtype=float)]
        )
    return values - properties.TRUTHS[label]


#: The pass probability the joint band's pilot reading needs at the declared count.
JOINT_PASS = 0.90


def band_pass_probability(coverage: float, replicates: int, level: float = 0.99) -> float:
    """The chance that the declared joint cell's exact interval lies inside the band.

    At true joint coverage ``coverage`` the cell's covered count over ``replicates`` is
    binomial, and the cell passes when the two-sided Clopper-Pearson interval at ``level`` lies
    inside :attr:`Margins.calibration_coverage`, the rule
    :func:`~tests.studies.evidence.property_verdicts.simultaneous_coverage_verdicts` reads.
    """
    from scipy.stats import beta as beta_law
    from scipy.stats import binom

    hits = np.arange(replicates + 1)
    tail = (1.0 - level) / 2.0
    with np.errstate(invalid="ignore"):
        low = np.where(hits > 0, beta_law.ppf(tail, hits, replicates - hits + 1), 0.0)
        high = np.where(
            hits < replicates, beta_law.ppf(1.0 - tail, hits + 1, replicates - hits), 1.0
        )
    lower, upper = MARGINS.calibration_coverage
    inside = (low >= lower) & (high <= upper)
    return float(binom.pmf(hits[inside], replicates, coverage).sum())


class TestTheWrongLearners:
    def test_the_wrong_learner_rule(self) -> None:
        """The plug-in contrast limit lies at least 3 efficiency SEs from the truth at n = 2,000."""
        limit = law.wrong_plug_in_limit(law.ONE_CAUSE, properties.target(CONTRAST))
        bias = (limit - properties.TRUTHS[CONTRAST]) / _scale(CONTRAST, 2_000)
        assert abs(bias) >= 3.0
        assert bias == pytest.approx(WRONG_GAP, abs=1e-3)

    def test_the_estimator_plug_in_is_the_limit(self) -> None:
        """The untargeted backward pass on the exact sample is the limit computed from the law."""
        result = _quiet(
            study.untargeted_fit,
            "one_cause",
            law.ONE_CAUSE.exact_frame(),
            configuration="mechanism_correct",
            weights="wt",
        )
        for label, name in properties.COEFFICIENTS["one_cause"]:
            limit = law.wrong_plug_in_limit(law.ONE_CAUSE, properties.target(label))
            assert result[name].psi == pytest.approx(limit, abs=1e-8)

    def test_the_dropout_is_what_the_wrong_learners_miss(self) -> None:
        """The control: with drop-out free of ``L_k`` the plug-in without ``L`` misses by little.

        The wrong learners' gap comes from the drop-out reading ``L_k``; on the same law with
        that coefficient set to zero, only the non-collapsibility of the logistic hazard over
        ``L`` is left, below one efficiency SE.
        """
        from dataclasses import replace

        free = replace(law.ONE_CAUSE, dropout=replace(law.DROPOUT, l=0.0))
        aim = properties.target(CONTRAST)
        gap = (law.wrong_plug_in_limit(free, aim) - aim.value(free)) / (
            law.efficiency_sd(free, aim) / np.sqrt(2_000)
        )
        assert abs(gap) < 1.0

    @pytest.mark.parametrize("arm", list(PLUG_IN_GAPS), ids=lambda arm: "-".join(arm))
    def test_the_pilot_plug_in_varies(self, arm: tuple[str, str]) -> None:
        """Each wrong-learner label's plug-in varies and centres on its exact-sample limit."""
        _group("wrong learner")
        fit, _ = arm
        label = f"{fit}_contrast"
        rows = _kept(properties.WRONG_LEARNER_ARMS[arm])
        values = rows.loc[rows["estimand"] == properties.ESTIMAND[label], "wrong_plug_in"]
        values = values.to_numpy(dtype=float)
        assert np.var(values, ddof=1) > 0.0
        limit = properties.TRUTHS[label] + PLUG_IN_GAPS[arm] * _scale(label, pilot.PILOT_N)
        error = np.std(values, ddof=1) / np.sqrt(values.size)
        assert abs(values.mean() - limit) < 3.0 * error + 1e-3 * _scale(label, pilot.PILOT_N)

    @pytest.mark.parametrize("label", list(pilot.WRONG_LEARNER_LABELS))
    def test_the_wrong_learner_fits_converge(self, label: str) -> None:
        """At most 1% of a wrong-learner label's kept draws fail to converge."""
        _group("wrong learner")
        rows = _kept(label).drop_duplicates("replicate")
        share = float((rows["converged"] == 0).mean())
        assert share <= properties.NONCONVERGED_SUSPECT_SHARE, (label, share)


# ---------------------------------------------------------------------------- sizes


class TestTheSizes:
    def test_the_power_cell_has_power(self) -> None:
        """The power cell reads the ``one_cause`` calibration fits; it has power at their count."""
        assert properties.POWER_REPLICATES == properties.CALIBRATION_REPLICATES
        power = design_power(
            properties.TRUTHS[CONTRAST], properties.EFFICIENCY_SD[CONTRAST], properties.POWER_N
        )
        assert power >= 0.80
        assert power_cell_pass_probability(power, properties.POWER_REPLICATES) >= 0.99

    def test_the_derived_controls_leave_the_band(self) -> None:
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
        for spread in (1.0, 1.5):
            probability = sizing.bias_discriminated_probability(
                MARGINS, WRONG_GAP / spread, properties.dr_replicates(*properties.NECESSITY_ARM)
            )
            assert probability >= sizing.discrimination_floor(2)

    @pytest.mark.skipif(CALIBRATION_PILOTED, reason="the committed calibration pilot sizes R")
    def test_the_calibration_replicates_at_a_calibrated_pilot(self) -> None:
        """The arm-pooled rule at a pilot of the declared size that reads a ratio of one."""
        arms = sum(len(properties.COEFFICIENTS[label]) for label in properties.CALIBRATION_FITS)
        assert arms == 8
        replicates, cell, family = mtp_study_tools.smallest_replicates(
            1.0, sizing.CALIBRATION_PILOT_DRAWS, arms
        )
        assert replicates == properties.CALIBRATION_REPLICATES
        assert cell >= mtp_study_tools.CELL_TARGET and family >= mtp_study_tools.FAMILY_TARGET

    @staticmethod
    def _calibration_rows(fit_label: str) -> pd.DataFrame:
        """One calibration label's kept rows; fails while only some labels are committed."""
        _group("calibration")
        assert pilot.PILOTS[f"calibration_{fit_label}"][1] == sizing.CALIBRATION_PILOT_DRAWS
        return _kept(f"calibration_{fit_label}")

    def test_the_calibration_replicates_follow_the_pilot(self) -> None:
        ratios = []
        for fit_label in properties.CALIBRATION_FITS:
            rows = self._calibration_rows(fit_label)
            for _, name in properties.COEFFICIENTS[fit_label]:
                arm = rows.loc[rows["estimand"] == name]
                assert len(arm) == rows["replicate"].nunique()
                ratios.append(float(arm["std_error"].mean() / arm["estimate"].std(ddof=1)))
        reading = mtp_study_tools.replicates_from_pilot(ratios, sizing.CALIBRATION_PILOT_DRAWS)
        assert not reading["suspect"]
        assert all(reading["screened"])
        assert reading["R"] == properties.CALIBRATION_REPLICATES

    def test_every_positive_arm_passes_at_its_exact_limit(self) -> None:
        """Each positive bias arm passes with probability at least 0.99 at standardized bias 0."""
        for cell, replicates in POSITIVE_ARMS.items():
            probability = sizing.bias_equivalent_probability(MARGINS, 0.0, replicates)
            assert probability >= POSITIVE_PASS, (cell, probability)
        assert _tolerated_bias(properties.DOUBLE_ROBUST_REPLICATES) == pytest.approx(
            0.1085, abs=1e-3
        )

    @pytest.mark.parametrize("arm", list(PLUG_IN_GAPS), ids=lambda arm: "-".join(arm))
    def test_the_wrong_learner_arms_are_sized_by_the_pilot_rule(self, arm: tuple[str, str]) -> None:
        """Each arm with a wrong learner takes the pre-committed size (plan X70 rev 2.7).

        The arm's replicates are the larger of
        :data:`~tests.studies.hazard_time_varying_properties.DOUBLE_ROBUST_REPLICATES` and the
        smallest multiple of 400 that passes with :data:`POSITIVE_PASS` at its pilot's
        |standardized bias| plus one of its standard errors, with no cap; a reading that needs
        more than 4,000 stops the declaration for a diagnosis.  The ``one_cause``
        ``mechanism_correct`` reading pools the ``fits_differ_mechanism_correct`` draws, the
        same hazard fit on its own seeds, and the targeted necessity arm reads the same fits.
        """
        error = _wrong_learner_errors(arm)
        standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
        needed = int(properties.DOUBLE_ROBUST_REPLICATES)
        while sizing.bias_equivalent_probability(MARGINS, standardized, needed) < POSITIVE_PASS:
            needed += 400
        assert needed <= 4_000, (arm, standardized, needed)
        fit, configuration = arm
        assert POSITIVE_ARMS[f"{fit}_contrast__{configuration}"] == needed
        assert properties.WRONG_LEARNER_REPLICATES[arm] == needed
        if arm == properties.NECESSITY_ARM:
            assert POSITIVE_ARMS[f"{CONTRAST}__targeted"] == needed

    def test_the_positive_arms_follow_the_calibration_pilot(self) -> None:
        """The ``both_correct`` and ``outcome_correct`` arms and the claimed rungs, at the pilot.

        The ``outcome_correct`` arm walks with censoring weights that omit ``L_k``, and its own
        finite-sample bias is not piloted: the ``both_correct`` reading is its proxy.  At 1,200
        replicates the arm tolerates a standardized bias of 0.108, so the risk is small.
        """
        rows = self._calibration_rows("one_cause")
        for cell_label, name in properties.COEFFICIENTS["one_cause"]:
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

    def test_the_band_pass_probability(self) -> None:
        """The joint-band rule passes a calibrated band and stops at a band near its edge."""
        replicates = properties.CALIBRATION_REPLICATES
        assert band_pass_probability(0.95, replicates) > 0.999
        assert band_pass_probability(0.915, replicates) < 0.01
        assert band_pass_probability(0.93, replicates) < JOINT_PASS

    @pytest.mark.parametrize("joint", study.JOINT_LABELS)
    def test_the_joint_band_pilot_meets_the_band(self, joint: str) -> None:
        """The declared joint cell passes with probability 0.90 at the pilot's coverage +/- 1 SE.

        The calibration pilots of ``one_cause`` and ``competing`` fit with
        ``simultaneous=True``, as the declared calibration fits do, so they read the band the
        ``simultaneous_coverage`` cells gate, over every estimate of the fit, node-1 risks
        included.  At the pilot's joint coverage minus and plus one standard error, the 99%
        interval of the declared cell at
        :data:`~tests.studies.hazard_time_varying_properties.CALIBRATION_REPLICATES` must lie
        inside the band with probability at least :data:`JOINT_PASS`
        (:func:`band_pass_probability`).  The rule was fixed before the pilot was read.  A
        failure stops the declaration for a diagnosis: the band is not widened, and the count
        is not raised after the reading.
        """
        rows = self._calibration_rows(joint.removesuffix("_curve"))
        band = rows.loc[rows["estimand"] == "joint_band", "covered"].to_numpy(dtype=float)
        pointwise = rows.loc[rows["estimand"] == "joint_pointwise", "covered"]
        assert band.size == rows["replicate"].nunique() == pointwise.size
        coverage = float(band.mean())
        error = float(np.sqrt(coverage * (1.0 - coverage) / band.size))
        replicates = properties.CALIBRATION_REPLICATES
        passes = {
            edge: band_pass_probability(edge, replicates)
            for edge in (coverage - error, coverage + error)
        }
        print(joint, coverage, error, passes, "pointwise", float(pointwise.mean()))
        if min(passes.values()) < JOINT_PASS:
            pytest.fail(
                f"{joint}: at the pilot's joint coverage {coverage:.4f} +/- {error:.4f} the "
                f"declared cell at {replicates} replicates passes with probability "
                f"{min(passes.values()):.3f}, below {JOINT_PASS}; the declaration stops for a "
                "diagnosis, and the band is not widened"
            )

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` on the calibration pilots."""
        covered: dict[str, int] = {}
        groups = []
        for scenario in study.SCENARIOS:
            rows = self._calibration_rows(scenario)
            rows = rows.loc[~rows["estimand"].isin(pilot.JOINT_ROWS)]
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


# ---------------------------------------------------------------------------- the IC gap


def _ic_gaps(fit_label: str) -> dict[str, tuple[list[float], float]]:
    rows = _pilot("ic_gap")
    if rows.empty:
        pytest.skip("the ic_gap pilot has not been committed")
    assert rows["refusal"].fillna("").eq("").all()
    assert (rows["held"] == 0).all(), "an ic_gap fit held a cell; rerun the label"
    small_n, large_n = properties.IC_GAP_SIZES
    rows = rows.loc[rows["fit_label"] == fit_label]
    out: dict[str, tuple[list[float], float]] = {}
    for label, _ in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == label]
        small = arm.loc[arm["n"] == small_n, "estimate"].to_numpy(dtype=float)
        large = arm.loc[arm["n"] == large_n, "estimate"].to_numpy(dtype=float)
        assert small.size == properties.IC_GAP_DRAWS and large.size == 1, (fit_label, label)
        out[label.rsplit("_", 1)[1]] = (list(small), float(large[0]))
    return out


class TestTheInfluenceCurveGap:
    @pytest.mark.parametrize("fit_label", study.CALIBRATION_FITS)
    def test_a_fit_meets_the_declared_bound(self, fit_label: str) -> None:
        small_n, large_n = properties.IC_GAP_SIZES
        for arm, (small, large) in _ic_gaps(fit_label).items():
            median = float(np.median(small))
            assert median <= properties.IC_GAP_BOUND[small_n], (arm, small)
            assert large <= properties.IC_GAP_BOUND[large_n], (arm, large)
            assert large <= properties.IC_GAP_FALL * median, (arm, median, large)

    @pytest.mark.parametrize("fit_label", study.CALIBRATION_FITS)
    def test_a_large_fit_reports_the_bound(self, fit_label: str) -> None:
        """At n = 40,000 each fit's influence-curve SD is within 5% of the bound."""
        rows = _pilot("ic_gap")
        if rows.empty:
            pytest.skip("the ic_gap pilot has not been committed")
        rows = rows.loc[(rows["fit_label"] == fit_label) & (rows["n"] == pilot.BOUND_N)]
        assert len(rows) == len(properties.COEFFICIENTS[fit_label])
        for label, ratio in zip(rows["estimand"], rows["estimate"], strict=True):
            assert ratio == pytest.approx(1.0, abs=0.05), label


# ---------------------------------------------------------------------------- pilot readings


def _paired_ratios(
    rows: pd.DataFrame, fit_label: str, column: str = "iterated_estimate"
) -> dict[str, float]:
    ratios = {}
    for label, name in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == name]
        difference = arm["estimate"] - arm[column]
        ratios[label] = float(difference.std(ddof=1)) / _scale(label, pilot.PILOT_N)
    return ratios


@pytest.mark.parametrize("fit_label", study.DIFFERENCE_FITS)
def test_the_agreement_pilot_is_reported(fit_label: str) -> None:
    """The ``iterated_mean_difference`` readings, reported with no floor."""
    if _pilot(f"fits_differ_{fit_label}").empty:
        pytest.skip("the pilot has not been committed")
    rows = _kept(f"fits_differ_{fit_label}")
    decreases = rows.loc[rows["estimand"] == "largest_decrease", "estimate"]
    ratios = _paired_ratios(rows, fit_label)
    ratios |= {
        f"{label} (isotonic)": value
        for label, value in _paired_ratios(rows, fit_label, "isotonic_estimate").items()
    }
    reading = properties.fits_differ_reading(ratios, int((decreases > 0.0).sum()), len(decreases))
    assert all(np.isfinite(value) and value > 0.0 for value in reading["ratios"].values())
    print(reading["text"])


def test_the_agreement_reading_carries_no_floor() -> None:
    reading = properties.fits_differ_reading({"one_cause_contrast": 0.03}, 0, 200)
    assert not {key for key in reading if key.endswith("passes")}
    assert reading["share_interval"] == (0.0, pytest.approx(0.0261, abs=1e-4))
    assert "floor" in reading["text"] and "0.05" not in reading["text"]


# ---------------------------------------------------------------------------- support rows


def _fit_key(fit_label: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The support key of one fit label's fit, read where the estimator classifies it."""
    keys: list[Any] = []
    original = estimator_module.classify_longitudinal_fit

    def spy(estimator: Any, data: Any, **settings: Any) -> Any:
        names = (
            "outcome_kind",
            "cross_fitted",
            "known_factors",
            "has_msm",
            "regimen_kinds",
            "tuned_cap",
            "separable_first",
        )
        key, _ = _support.longitudinal_key(
            estimator, data, **{name: settings[name] for name in names if name in settings}
        )
        keys.append(key)
        return original(estimator, data, **settings)

    monkeypatch.setattr(estimator_module, "classify_longitudinal_fit", spy)
    result = _quiet(study.fit, fit_label, study.law_of(fit_label).draw(600, 1))
    return keys[-1], result


def _axis_exists() -> bool:
    """Whether the support key has gained the ``later_covariates`` option."""
    import inspect

    from cleverly._support_rows import long_row

    return study.LATER_COVARIATES_AXIS in inspect.signature(long_row).parameters


class TestTheSupportRows:
    @pytest.mark.parametrize("row", study.declared_support_rows(), ids=lambda row: row.id)
    def test_each_declared_row_covers_its_fit(
        self, row: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key, result = _fit_key(study.G63_FITS[row.id], monkeypatch)
        assert key.is_covered_by(row.key), key.differing_axes(row.key)
        assert row.evidence == (study.STUDY.slug,)
        assert set(row.cells) <= set(properties.ESTIMAND)
        labels = {estimate.support.label for estimate in result.estimates.values()}
        assert labels == {"experimental"}

    def test_the_rows_are_not_yet_support_rows(self) -> None:
        ids = {row.id for row in LONGITUDINAL_ROWS}
        assert not ids & set(study.G63_FITS)

    def test_the_key_needs_an_axis_for_covariates_after_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why registration adds a key axis: today a registered row covers these keys.

        ``G31`` and ``G32`` license held hazard fits without covariates after baseline.  Their
        keys cover the in-sample, competing and cross-fitted fits here, which only the X70
        condition keeps ``experimental``; a ``G63.held`` row with the same key would collide.
        Once the key gains the ``later_covariates`` option
        (:data:`~tests.studies.canonical_hazard_time_varying.LATER_COVARIATES_AXIS`), the test
        flips: every declared row must name the axis as True and no registered row may cover
        these fits.
        """
        covering = {}
        for row_id in ("G63.held.in_sample", "G63.held.competing", "G63.held.cross_fitted"):
            key, _ = _fit_key(study.G63_FITS[row_id], monkeypatch)
            covering[row_id] = {row.id for row in LONGITUDINAL_ROWS if key.is_covered_by(row.key)}
        if _axis_exists():
            for row in study.declared_support_rows():
                assert getattr(row.key, study.LATER_COVARIATES_AXIS) is True, row.id
            assert not any(covering.values()), covering
            return
        assert covering == {
            "G63.held.in_sample": {"G31.survival"},
            "G63.held.competing": {"G31.competing"},
            "G63.held.cross_fitted": {"G32.survival"},
        }

    def test_the_rows_say_what_the_key_cannot(self) -> None:
        """The banded rows license the gated band, and every row limits its covariate law."""
        from cleverly._support_rows import _LONG_BAND, _LONG_WALD

        for row in study.declared_support_rows():
            kinds = _LONG_BAND if row.id in study.G63_BANDED else _LONG_WALD
            assert frozenset(row.kinds) == kinds, row.id
            text = " ".join(row.limits)
            assert "binary" in text and "Markov" in text and "Regimens:" in text, row.id
        banded = {study.G63_FITS[row_id] for row_id in study.G63_BANDED}
        assert banded == {label.removesuffix("_curve") for label in study.JOINT_LABELS}


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

        records = {record.slug: record for record in registered()}
        for sibling in (canonical_hazard_msm.STUDY, canonical_hazard_mtp.STUDY):
            records.setdefault(sibling.slug, sibling)
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
        seeds += [
            stream_seed(study.STUDY, "pilot", "ic_gap", fit, n, r)
            for fit, n, r in pilot.IC_GAP_JOBS
        ]
        assert len(set(seeds)) == len(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(50)
        }
        assert not foreign & set(seeds)


def test_a_partial_pilot_group_fails_and_names_the_missing_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group with some labels committed fails, naming the rest; with none it skips."""
    module = __import__(__name__, fromlist=["PILOT"])
    partial = pd.DataFrame({"label": ["calibration_one_cause", "wrong_learner_competing"]})
    monkeypatch.setattr(module, "PILOT", partial)
    for name, missing in (
        ("calibration", "calibration_competing"),
        ("wrong learner", "wrong_learner_one_cause"),
    ):
        with pytest.raises(pytest.fail.Exception, match=missing):
            _group(name)
    monkeypatch.setattr(module, "PILOT", pd.DataFrame({"label": ["ic_gap"]}))
    with pytest.raises(pytest.skip.Exception):
        _group("calibration")
