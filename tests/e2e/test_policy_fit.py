"""End-to-end point-treatment fits of every modified treatment policy class.

The exact-law tests in ``tests/unit/test_policy_point_exact.py`` check each policy's clever
covariate and influence curve against a differentiated law.  These fits check what a user
reaches: that ``TMLE(policies=...)`` runs each class on a dose, on a binary treatment and on
a three-level one, by either ratio route, with clusters, and that each fit solves its score
equations and reports the estimand names the policy class declares.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

from cleverly.datasets import make_shift_dose
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.interventions import (
    ModifiedPolicy,
    Piece,
    Piecewise,
    Randomizer,
    RiskRatioTilt,
    Scale,
    Shift,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

INF = np.inf

#: Every continuous policy class, beside the natural course.
CONTINUOUS = [
    Shift(0.0, cap=None),
    Scale(1.1, cap=5.0, name="x1.1"),
    Piecewise(((-INF, 2.0, Shift(0.0, None)), (2.0, INF, Shift(-0.5, None))), name="trim high"),
    ModifiedPolicy(
        "halve above 3",
        pieces=(
            Piece(-INF, 3.0),
            Piece(
                3.0, INF, lambda a, h: 1.5 + a / 2.0, lambda b, h: 2.0 * (b - 1.5), lambda b, h: 2.0
            ),
        ),
        policy_kind="known",
    ),
    ModifiedPolicy(
        "sometimes up",
        pieces={
            "up": (Piece(-INF, INF, lambda a, h: a + 0.5, lambda b, h: b - 0.5, lambda b, h: 1.0),),
            "stay": (Piece(-INF, INF),),
        },
        randomizer=Randomizer(("up", "stay"), (0.4, 0.6)),
        policy_kind="known",
    ),
]


def _dose_fit(**settings: Any) -> Any:
    frame, _ = make_shift_dose(n=800, seed=3)
    options: dict[str, Any] = {
        "outcome_learner": sklearn.linear_model.LinearRegression(),
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "policies": CONTINUOUS,
        "density_bins": 12,
        "simultaneous": False,
        "random_state": 0,
    }
    options.update(settings)
    covariates = [c for c in frame.columns if c.startswith("W")]
    return TMLE(**options).fit(frame, outcome="Y", treatment="A", covariates=covariates).single()


def _binary_frame(n: int = 1200, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.6 * w)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * w + 0.8 * a - 0.2))))
    return pd.DataFrame({"W": w, "A": a, "Y": y})


def _fit_frame(frame: pd.DataFrame, **settings: Any) -> Any:
    options: dict[str, Any] = {
        "outcome_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    options.update(settings)
    return TMLE(**options).fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()


@pytest.fixture(scope="module")
def dose_fit() -> Any:
    return _dose_fit()


def test_every_continuous_class_reports_an_ey_policy(dose_fit: Any) -> None:
    names = {f"ey_policy[{policy.name}]" for policy in CONTINUOUS}
    assert names <= set(dose_fit.estimates)
    assert dose_fit.config.parameter_axis == "policy"
    for name in names:
        assert np.isfinite(dose_fit.estimates[name].psi)


def test_every_score_equation_is_solved(dose_fit: Any) -> None:
    fluctuation = dose_fit.fluctuations["mtp"]
    assert fluctuation.score_norm < 1e-6
    # One equation per component: the randomized policy expands into two.
    assert len(fluctuation.epsilon) == len(CONTINUOUS) + 1


def test_the_natural_course_is_the_outcome_mean(dose_fit: Any) -> None:
    frame, _ = make_shift_dose(n=800, seed=3)
    assert dose_fit.estimates["ey_policy[natural course]"].psi == pytest.approx(
        float(frame["Y"].mean()), abs=1e-6
    )


def test_the_randomized_policy_is_the_weighted_sum_of_its_components() -> None:
    """With the up branch alone declared, the randomized mean is 0.4 * up + 0.6 * E[Y]."""
    up = ModifiedPolicy(
        "up",
        pieces=(Piece(-INF, INF, lambda a, h: a + 0.5, lambda b, h: b - 0.5, lambda b, h: 1.0),),
        policy_kind="known",
    )
    fit = _dose_fit(policies=[Shift(0.0, cap=None), up, CONTINUOUS[-1]])
    natural = fit.estimates["ey_policy[natural course]"].psi
    branch = fit.estimates["ey_policy[up]"].psi
    mixed = fit.estimates["ey_policy[sometimes up]"].psi
    assert mixed == pytest.approx(0.4 * branch + 0.6 * natural, abs=1e-8)


def test_the_support_report_describes_each_policy(dose_fit: Any) -> None:
    report = dose_fit.diagnostics.support()
    assert set(report) == {policy.name for policy in CONTINUOUS}
    assert report["x1.1"].policy == "Scale(factor=1.1, cap=5.0)"
    assert report["sometimes up"].score_load_omission == "randomized_policy_components"
    assert 0.0 < report["sometimes up"].moved_fraction < 1.0


def test_the_support_report_counts_the_rows_a_cap_holds(dose_fit: Any) -> None:
    """A nonzero witness: the cap of ``x1.1`` holds rows, and the report counts them.

    The assessment grades this share against 5%.  A record that omitted it would read as
    zero there, so the expected share is computed here from the declaration.
    """
    frame, _ = make_shift_dose(n=800, seed=3)
    dose = frame["A"].to_numpy(dtype=float)
    expected = float(np.mean(dose * 1.1 > 5.0))
    assert expected > 0.0
    report = dose_fit.diagnostics.support()
    assert report["x1.1"].capped_fraction == pytest.approx(expected, abs=1e-12)
    assert report["trim high"].capped_fraction == 0.0
    assert "capped=" in report["x1.1"].summary()
    by_classifier = _dose_fit(ratio="classifier").diagnostics.support()
    assert by_classifier["x1.1"].capped_fraction == pytest.approx(expected, abs=1e-12)


def test_a_categorical_support_report_evaluates_no_cap() -> None:
    fit = _fit_frame(_binary_frame(), policies=[RiskRatioTilt(1.0), RiskRatioTilt(0.5)])
    for row in fit.diagnostics.support().values():
        assert row.capped_fraction is None


def test_the_classifier_route_runs_and_agrees_with_the_density_route(dose_fit: Any) -> None:
    by_classifier = _dose_fit(ratio="classifier")
    for policy in CONTINUOUS:
        name = f"ey_policy[{policy.name}]"
        assert by_classifier.estimates[name].psi == pytest.approx(
            dose_fit.estimates[name].psi, abs=0.1
        )
    assert by_classifier.nuisance.density is None
    # No density, so no zero can be read: the report says so rather than printing 0.
    row = by_classifier.diagnostics.support()["x1.1"]
    assert row.min_density is None and row.unsupported is None
    assert "unsupported not measured on the classifier route" in row.summary()


def test_a_cross_fitted_clustered_policy_fit_runs() -> None:
    frame, _ = make_shift_dose(n=600, seed=5)
    frame = frame.assign(cluster=np.arange(len(frame)) // 3)
    covariates = [c for c in frame.columns if c.startswith("W")]
    fit = (
        TMLE(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            cross_fit=True,
            n_folds=3,
            policies=[Shift(0.0, cap=None), Scale(1.1, cap=5.0, name="x1.1")],
            density_bins=10,
            ratio="classifier",
            simultaneous=False,
            random_state=0,
            q_bounds=(float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates, id="cluster")
        .single()
    )
    assert np.isfinite(fit.estimates["ate_policy[x1.1 vs natural course]"].psi)


class TestCategoricalPolicies:
    def test_a_risk_ratio_tilt_reports_its_own_estimands(self) -> None:
        frame = _binary_frame()
        fit = _fit_frame(frame, policies=[RiskRatioTilt(1.0), RiskRatioTilt(0.5)])
        assert fit.config.parameter_axis == "rr_tilt"
        assert set(fit.estimates) >= {
            "ey_rr_tilt[natural course]",
            "ey_rr_tilt[rr 0.5]",
            "ate_rr_tilt[rr 0.5 vs natural course]",
        }
        assert fit.estimates["ey_rr_tilt[natural course]"].psi == pytest.approx(
            float(frame["Y"].mean()), abs=1e-6
        )
        # Halving the treated share lowers the risk when treatment raises it.
        assert fit.estimates["ate_rr_tilt[rr 0.5 vs natural course]"].psi < 0.0
        assert fit.fluctuations["mtp"].score_norm < 1e-6

    def test_a_label_map_on_a_binary_treatment(self) -> None:
        frame = _binary_frame()
        treat_high = ModifiedPolicy(
            "treat high W",
            apply=lambda a, h: np.where(np.asarray(h["W"]) > 0.5, 1, a),
            policy_kind="known",
        )
        fit = _fit_frame(frame, policies=[Shift(0.0, cap=None), treat_high])
        assert fit.config.parameter_axis == "policy"
        assert fit.estimates["ate_policy[treat high W vs natural course]"].psi > 0.0
        assert "propensity truncated" in fit.summary()

    def test_a_three_level_policy_and_the_lmtp_example_form(self) -> None:
        rng = np.random.default_rng(6)
        n = 1500
        w = rng.normal(size=n)
        a = rng.integers(0, 6, size=n)
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * w + 0.2 * a - 0.5))))
        frame = pd.DataFrame({"W": w, "A": a, "Y": y})
        # lmtp's man/lmtp_tmle.Rd example: (a - 1) * (a - 1 >= 1) + a * (a - 1 < 1).
        lower = ModifiedPolicy(
            "a - 1 when it stays above 0",
            apply=lambda a, h: np.where(np.asarray(a) - 1 >= 1, np.asarray(a) - 1, a),
            policy_kind="known",
        )
        # A shift that maps level 0 to -1 leaves the support and is refused by name.
        with pytest.raises(DataError, match="not levels of the node"):
            TMLE(
                outcome_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                cross_fit=False,
                policies=[Shift(0.0, cap=None), Shift(-1.0, cap=None, name="minus one")],
            ).fit(frame, outcome="Y", treatment="A", covariates=["W"], treatment_kind="discrete")
        result = (
            TMLE(
                outcome_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
                policies=[Shift(0.0, cap=None), lower],
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], treatment_kind="discrete")
            .single()
        )
        effect = result.estimates["ate_policy[a - 1 when it stays above 0 vs natural course]"]
        assert effect.psi < 0.0
        assert result.fluctuations["mtp"].score_norm < 1e-6
