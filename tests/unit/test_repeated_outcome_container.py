"""A repeated outcome in the container, the study design, and the compositions around a fit.

W4: ``Y_s`` is the first covariate of node ``s + 1``.  W5: presence, family and subset
rules of the container.  W7: a working model over (plan, horizon) cells.  W9: the data
fingerprint reads every node's outcome.  W10: a ``CausalStudy`` design.  W11: the
full-refit bootstrap is a diagnostic.  W12: the truncation replay.  W14: one outcome scale
for the whole curve.  W15: the working-model support check reads every node.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy
from cleverly.exceptions import DataError
from cleverly.longitudinal import LTMLE, LongitudinalData, Repeated
from cleverly.longitudinal.estimator import bootstrap_design_kind
from cleverly.msm import MSM
from cleverly.study import LongitudinalTreatment
from cleverly.utils.bounds import OutcomeScaler

from .. import discrete_law_repeated as laws
from .test_influence_gateaux_repeated import COLUMNS as LAW_COLUMNS
from .test_influence_gateaux_repeated import oracle_fit
from .test_repeated_outcome import panel

ROLES: dict[str, Any] = {
    "outcome": Repeated(["Y1", "Y2", "Y3"]),
    "treatment": ["A1", "A2", "A3"],
    "baseline": ["W"],
    "time_varying": [["L1"], ["L2"], ["L3"]],
    "censoring": ["C1", "C2", "C3"],
}


def _estimator(**overrides: Any) -> LTMLE:
    settings: dict[str, Any] = {
        "reference": "never",
        "outcome_learner": LogisticRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "censoring_learner": LogisticRegression(),
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return LTMLE({"never": 0, "always": 1}, **settings)


def _quiet_fit(estimator: LTMLE, data: Any, **roles: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(data, **roles)


# ---------------------------------------------------------------- W4 and W5


def test_w4_each_earlier_outcome_opens_the_next_block() -> None:
    data = LongitudinalData.from_frame(panel(), **ROLES)
    assert data.time_varying_names == (("L1",), ("Y1", "L2"), ("Y2", "L3"))
    assert data.history_names(3) == ("W", "L1", "Y1", "L2", "Y2", "L3")
    assert "Y1" in data.history_frame(2).columns
    assert data.repeated_names == ("Y1", "Y2", "Y3")
    assert data.is_repeated and data.reports_curve and not data.is_survival


def test_w5_the_outcome_array_keeps_every_node_and_its_presence() -> None:
    frame = panel()
    data = LongitudinalData.from_frame(frame, **ROLES)
    assert data.repeated is not None and data.repeated.shape == (data.n, 3)
    for t in (1, 2, 3):
        observed = data.uncensored_through(t)
        np.testing.assert_array_equal(np.isfinite(data.repeated[:, t - 1]), observed)
        np.testing.assert_array_equal(
            data.outcome_at(t)[observed], frame[f"Y{t}"].to_numpy()[observed]
        )
    np.testing.assert_array_equal(data.outcome_at(3), data.repeated[:, 2])
    pooled = data.observed_outcomes()
    assert pooled.size == sum(int(data.uncensored_through(t).sum()) for t in (1, 2, 3))


def test_w5_the_family_is_resolved_over_every_node() -> None:
    """A binary last node with a continuous earlier node is one gaussian outcome."""
    frame = panel()
    frame.loc[frame["Y1"].notna(), "Y1"] = frame.loc[frame["Y1"].notna(), "Y1"] + 0.5
    data = LongitudinalData.from_frame(frame, **ROLES)
    assert data.family == "gaussian"
    assert LongitudinalData.from_frame(panel(), **ROLES).family == "binomial"


def test_w5_a_subset_keeps_each_unit_whole() -> None:
    data = LongitudinalData.from_frame(panel(), **ROLES)
    index = np.arange(0, data.n, 3)
    part = data.subset(index)
    assert part.repeated is not None and data.repeated is not None
    np.testing.assert_array_equal(part.repeated, data.repeated[index])
    assert part.repeated_names == data.repeated_names


def test_w5_an_end_of_study_container_reads_only_its_last_node() -> None:
    data = LongitudinalData.from_frame(panel(), **{**ROLES, "outcome": "Y3"})
    assert not data.is_repeated and data.repeated_names == ()
    with pytest.raises(DataError, match="end-of-study outcome"):
        data.outcome_at(2)


# ---------------------------------------------------------------- W7: the working model

CELLS = [(d, h) for h in laws.HORIZONS for d in ("never", "always")]
TERMS = tuple(f"{d}@{h}" for d, h in CELLS)


def _saturated(label: Any, horizon: int, frame: Any) -> np.ndarray:
    return np.eye(len(CELLS))[CELLS.index((label, horizon))] * np.ones((len(frame), 1))


def _trend(label: Any, horizon: int, frame: Any) -> np.ndarray:
    n = len(frame)
    return np.column_stack(
        [np.ones(n), np.full(n, float(label == "always")), np.full(n, float(horizon))]
    )


def _msm_fit(design: Any, terms: tuple[str, ...]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return oracle_fit(
            laws.BINARY,
            regimens={"never": 0, "always": 1},
            reference=None,
            msm=MSM(design=design, terms=terms, design_kind="known"),
        )


def test_w7_a_saturated_model_reproduces_each_cell() -> None:
    """One indicator per (plan, horizon) cell gives each cell's mean and its EIF."""
    law = laws.BINARY
    fit = _msm_fit(_saturated, TERMS)
    rows = law.first_row_of()
    for (d, h), term in zip(CELLS, TERMS, strict=True):
        name = f"msm_regimen[{term}]"
        truth = f"ey_regimen[{d} @ t={h}]"
        assert fit.psi(name) == pytest.approx(law.truth[truth], abs=1e-12)
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], law.eif(truth), atol=1e-11, rtol=0
        )


def test_w7_a_trend_model_is_the_projection_of_the_cells() -> None:
    """``m(d, h) = b0 + b1 d + b2 h`` is the least-squares projection of the four means.

    The projection and its influence curve are written out here from the law's truths and
    EIFs.  Each cell reads its own horizon's outcome, so a fit that seeded every cell at
    the last horizon would move the ``h = 1`` cells and the coefficients.
    """
    law = laws.BINARY
    fit = _msm_fit(_trend, ("(intercept)", "always", "t"))
    phi = np.array([[1.0, float(d == "always"), float(h)] for d, h in CELLS])
    projection = np.linalg.solve(phi.T @ phi, phi.T)
    means = np.array([law.truth[f"ey_regimen[{d} @ t={h}]"] for d, h in CELLS])
    curves = np.array([law.eif(f"ey_regimen[{d} @ t={h}]") for d, h in CELLS])
    beta = projection @ means
    influence = projection @ curves
    rows = law.first_row_of()
    for column, term in enumerate(("(intercept)", "always", "t")):
        name = f"msm_regimen[{term}]"
        assert fit.psi(name) == pytest.approx(beta[column], abs=1e-12)
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], influence[column], atol=1e-11, rtol=0
        )


# ---------------------------------------------------------------- W9: the fingerprint


def test_w9_an_earlier_outcome_moves_the_data_fingerprint() -> None:
    """``Y1`` is fingerprinted even where the covariate screen would drop its copy."""
    data = LongitudinalData.from_frame(panel(), **ROLES)
    assert data.repeated is not None
    changed = data.repeated.copy()
    first = int(np.flatnonzero(np.isfinite(changed[:, 0]))[0])
    changed[first, 0] = 1.0 - changed[first, 0]
    other = dataclasses.replace(data, repeated=changed)
    left = _quiet_fit(_estimator(), data)
    right = _quiet_fit(_estimator(), other)
    assert left.provenance.data_fingerprint != right.provenance.data_fingerprint


def test_w9_a_competing_cause_swap_moves_the_data_fingerprint() -> None:
    """Two containers with the same all-cause pattern and a different cause differ."""
    from .. import discrete_law_competing as competing

    roles = {
        "outcome": {"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
        "treatment": ["A1", "A2"],
        "baseline": ["W"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }
    frame = competing.frame()
    data = LongitudinalData.from_frame(frame, **roles)
    assert data.cause_event is not None
    swapped = data.cause_event.copy()
    fired = np.flatnonzero(np.nansum(swapped[:, 0, :], axis=1) == 1)[0]
    swapped[fired, 0, :] = swapped[fired, 0, ::-1]
    other = dataclasses.replace(data, cause_event=swapped)
    estimator = LTMLE(
        {"never": 0, "always": 1},
        outcome_learner=competing.CellMeans(),
        pseudo_learner=competing.CellMeans(),
        treatment_learner=competing.CellMeans(),
        censoring_learner=competing.CellMeans(),
        n_folds=1,
        simultaneous=False,
    )
    left = _quiet_fit(estimator, data)
    right = _quiet_fit(estimator, other)
    assert left.provenance.data_fingerprint != right.provenance.data_fingerprint


# ---------------------------------------------------------------- W10: the study design


def test_w10_a_study_design_declares_a_repeated_outcome() -> None:
    frame = panel()
    design = LongitudinalTreatment(**ROLES)
    data = design.prepare(frame)
    assert data.is_repeated
    assert CausalStudy(data, design=design).data is data
    # The same blocks, with Y1 and Y2 declared as covariates of an end-of-study outcome:
    # only the outcome declaration tells the two containers apart.
    blocks = {"time_varying": [["L1"], ["Y1", "L2"], ["Y2", "L3"]], "outcome": "Y3"}
    end_of_study = LongitudinalData.from_frame(frame, **{**ROLES, **blocks})
    with pytest.raises(DataError, match="repeated outcome nodes"):
        CausalStudy(end_of_study, design=design)
    with pytest.raises(DataError, match="repeated outcome nodes"):
        CausalStudy(data, design=LongitudinalTreatment(**{**ROLES, **blocks}))


def test_w10_a_screened_constant_outcome_copy_does_not_refuse_the_design() -> None:
    frame = panel()
    # Y1 is zero on every row still under observation, so the covariate screen drops its copy.
    frame.loc[(frame["C1"] == 1).to_numpy(), "Y1"] = 0.0
    design = LongitudinalTreatment(**ROLES)
    data = LongitudinalData.from_frame(frame, **ROLES)
    assert "Y1" not in data.time_varying_names[1]
    assert data.repeated_names == ("Y1", "Y2", "Y3")
    assert CausalStudy(data, design=design).data is data


# ---------------------------------------------------------------- W11 and W12


def test_w11_the_bootstrap_is_a_diagnostic_on_a_repeated_fit() -> None:
    data = LongitudinalData.from_frame(panel(), **ROLES)
    result = _quiet_fit(_estimator(n_bootstrap=3), data)
    assert bootstrap_design_kind(data, result.folds, result.config.regimens, None) is None
    for estimate in result.estimates.values():
        assert estimate.bootstrap is None or estimate.bootstrap.inferential is False


def test_w12_the_truncation_replay_reproduces_the_fit() -> None:
    seeded = {
        name: LogisticRegression(random_state=0)
        for name in ("outcome_learner", "treatment_learner", "censoring_learner")
    }
    result = _quiet_fit(_estimator(**seeded), LongitudinalData.from_frame(panel(), **ROLES))
    curve = result.diagnostics.truncation_curve([result.config.g_bounds])
    payload = {column: list(curve[column]) for column in curve.columns}
    assert payload["psi"] == payload["fitted_psi"]
    assert set(payload["estimand"]) == set(result.estimates)


# ---------------------------------------------------------------- W14 and W15


def test_w14_one_scale_covers_every_horizon() -> None:
    """``Y1`` spans ``[-0.5, 2]`` and ``Y2`` spans ``[0, 1]``: the scale reads both.

    The oracle fit of the continuous law then returns ``E[Y_1^d]`` exactly.  A scale read
    off ``Y2`` alone clips ``Y1`` and moves the first horizon.
    """
    law = laws.CONTINUOUS
    fit = oracle_fit(law)
    pooled = OutcomeScaler.from_outcome(fit.data.observed_outcomes(), None)
    last = OutcomeScaler.from_outcome(fit.data.outcome_at(2)[fit.data.uncensored_through(2)], None)
    assert (fit.scaler.lower, fit.scaler.upper) == (pooled.lower, pooled.upper)
    assert fit.scaler.lower < min(law.y1_values) < last.lower
    for regimen in law.regimens:
        name = f"ey_regimen[{regimen} @ t=1]"
        assert fit.psi(name) == pytest.approx(law.truth[name], abs=1e-12)


def test_w15_a_logit_working_model_reads_every_node() -> None:
    """``Y2`` lies in ``[0, 1]`` and ``Y1`` does not, so a logit model is refused."""
    law = laws.CONTINUOUS
    with pytest.raises(DataError, match="Y1/Y2 ranges over"):
        LTMLE(
            {"never": 0, "always": 1},
            msm=MSM(
                design=_trend,
                terms=("(intercept)", "always", "t"),
                design_kind="known",
                link="logit",
            ),
            outcome_learner=laws.CellMeans(),
            pseudo_learner=laws.CellMeans(),
            treatment_learner=laws.CellMeans(),
            censoring_learner=laws.CellMeans(),
            n_folds=1,
            simultaneous=False,
        ).fit(law.frame(), **LAW_COLUMNS)


def test_repeated_is_exported_from_the_root_package() -> None:
    import cleverly

    assert cleverly.Repeated is Repeated
    assert "Repeated" in cleverly.__all__
