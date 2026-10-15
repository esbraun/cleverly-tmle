"""A repeated outcome in the container, the study design, and the compositions around a fit.

W4: ``Y_s`` is the first covariate of node ``s + 1``.  W5: presence, family and subset
rules of the container.  W7: a working model over (plan, horizon) cells.  W9: the data
fingerprint reads every node's outcome.  W10: a ``CausalStudy`` design.  W11: the
full-refit bootstrap is a diagnostic.  W12: the truncation replay.  W14: one outcome scale
for the whole curve.  W15: the working-model support check reads every node.  W16: the
parameter-space warning reads the range of its own horizon.  WK: a fit holds one key for each
plan and horizon.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, RegimeContrast
from cleverly.exceptions import DataError, ParameterSpaceWarning
from cleverly.inference.influence import make_estimate
from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI, LTMLE, SDR, LongitudinalData, Repeated
from cleverly.longitudinal import estimator as sequential
from cleverly.longitudinal.estimator import HORIZON_INFIX, bootstrap_design_kind
from cleverly.msm import MSM
from cleverly.study import LongitudinalTreatment
from cleverly.utils.bounds import OutcomeScaler

from .. import discrete_law_repeated as laws
from .test_influence_gateaux_repeated import COLUMNS as LAW_COLUMNS
from .test_influence_gateaux_repeated import ipsi_fit, oracle_fit
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


def _fresh(kind: type[Any], **overrides: Any) -> Any:
    """``LTMLE`` or ``IPSI`` with parametric learners and one fold."""
    return _estimator(**overrides) if kind is LTMLE else _sequential(kind, **overrides)


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


@dataclasses.dataclass(frozen=True)
class CellGrid:
    """The (plan, horizon) cells of one estimator kind on the exact laws, and their oracles."""

    kind: str
    plans: tuple[str, ...]
    high: str
    delta: dict[str, float]
    #: How closely the trend coefficients meet the written-out projection.  ``IPSI`` fits no
    #: fluctuation, and its projection is solved by the working model's own iteration.
    trend_tolerance: float = 1e-12

    @property
    def cells(self) -> list[tuple[str, int]]:
        return [(d, h) for h in laws.HORIZONS for d in self.plans]

    @property
    def terms(self) -> tuple[str, ...]:
        return tuple(f"{d}@{h}" for d, h in self.cells)

    def truth(self, law: laws.RepeatedLaw, plan: str, horizon: int) -> float:
        if self.kind == "LTMLE":
            return law.truth[f"ey_regimen[{plan} @ t={horizon}]"]
        return law.tilt_functional(law.probs, self.delta[plan], horizon)

    def eif(self, law: laws.RepeatedLaw, plan: str, horizon: int) -> np.ndarray:
        if self.kind == "LTMLE":
            return law.eif(f"ey_regimen[{plan} @ t={horizon}]")
        return law.tilt_eif(self.delta[plan], horizon)

    def fit(self, design: Any, terms: tuple[str, ...]) -> Any:
        """The exact-law fit with a working model: the kind's own estimator, saturated."""
        msm = MSM(design=design, terms=terms, design_kind="known")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.kind == "LTMLE":
                return oracle_fit(
                    laws.BINARY,
                    regimens={"never": 0, "always": 1},
                    reference=None,
                    msm=msm,
                )
            return ipsi_fit(laws.BINARY, tilts=(0.5, 2.0), reference=None, msm=msm)


GRIDS = (
    CellGrid("LTMLE", ("never", "always"), "always", {}),
    CellGrid(
        "IPSI",
        ("odds x0.5", "odds x2"),
        "odds x2",
        {"odds x0.5": 0.5, "odds x2": 2.0},
        trend_tolerance=1e-10,
    ),
)
GRID_IDS = [grid.kind for grid in GRIDS]


def _saturated(grid: CellGrid) -> Any:
    cells = grid.cells

    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        return np.eye(len(cells))[cells.index((label, horizon))] * np.ones((len(frame), 1))

    return design


def _trend(grid: CellGrid) -> Any:
    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        n = len(frame)
        return np.column_stack(
            [np.ones(n), np.full(n, float(label == grid.high)), np.full(n, float(horizon))]
        )

    return design


def check_w7_saturated(grid: CellGrid) -> None:
    """One indicator per (plan, horizon) cell gives each cell's mean and its EIF."""
    law = laws.BINARY
    fit = grid.fit(_saturated(grid), grid.terms)
    rows = law.first_row_of()
    for (d, h), term in zip(grid.cells, grid.terms, strict=True):
        name = f"msm_regimen[{term}]"
        assert fit.psi(name) == pytest.approx(grid.truth(law, d, h), abs=1e-12)
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], grid.eif(law, d, h), atol=1e-11, rtol=0
        )


def check_w7_trend(grid: CellGrid) -> None:
    """``m(d, h) = b0 + b1 d + b2 h`` is the least-squares projection of the four means.

    The projection and its influence curve are written out here from the law's truths and
    EIFs.  Each cell reads its own horizon's outcome, so a fit that seeded every cell at
    the last horizon would move the ``h = 1`` cells and the coefficients.
    """
    law = laws.BINARY
    fit = grid.fit(_trend(grid), ("(intercept)", "always", "t"))
    phi = np.array([[1.0, float(d == grid.high), float(h)] for d, h in grid.cells])
    projection = np.linalg.solve(phi.T @ phi, phi.T)
    means = np.array([grid.truth(law, d, h) for d, h in grid.cells])
    curves = np.array([grid.eif(law, d, h) for d, h in grid.cells])
    beta = projection @ means
    influence = projection @ curves
    rows = law.first_row_of()
    for column, term in enumerate(("(intercept)", "always", "t")):
        name = f"msm_regimen[{term}]"
        assert fit.psi(name) == pytest.approx(beta[column], abs=grid.trend_tolerance)
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], influence[column], atol=1e-11, rtol=0
        )


@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_w7_a_saturated_model_reproduces_each_cell(grid: CellGrid) -> None:
    """W7s and W7i: ``SDR`` joins the grid with its own PR.

    For ``IPSI`` the oracles are the tilt functional and its complex-step derivative.
    """
    check_w7_saturated(grid)


@pytest.mark.parametrize("grid", GRIDS, ids=GRID_IDS)
def test_w7_a_trend_model_is_the_projection_of_the_cells(grid: CellGrid) -> None:
    check_w7_trend(grid)


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


def test_w10_the_study_withholds_the_kinds_that_have_no_reduction() -> None:
    """The fit refuses SDR on a repeated outcome (R7), so the listing withholds it."""
    design = LongitudinalTreatment(**ROLES)
    study = CausalStudy(design.prepare(panel()), design=design)
    effect = study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never"))
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["tmle"].available and methods["ipsi"].available
    assert not methods["sdr"].available
    assert methods["sdr"].reason == sequential.repeated_outcome_refusal("SDR")


# ---------------------------------------------------------------- W11 and W12


@pytest.mark.parametrize("kind", [LTMLE, IPSI], ids=["LTMLE", "IPSI"])
def test_w11_the_bootstrap_is_a_diagnostic_on_a_repeated_fit(kind: type[Any]) -> None:
    data = LongitudinalData.from_frame(panel(), **ROLES)
    result = _quiet_fit(_fresh(kind, n_bootstrap=3), data)
    assert bootstrap_design_kind(data, result.folds, result.config.regimens, None) is None
    for estimate in result.estimates.values():
        assert estimate.bootstrap is None or estimate.bootstrap.inferential is False


@pytest.mark.parametrize("kind", [LTMLE, IPSI], ids=["LTMLE", "IPSI"])
def test_w12_the_truncation_replay_reproduces_the_fit(kind: type[Any]) -> None:
    seeded = {
        name: LogisticRegression(random_state=0)
        for name in ("outcome_learner", "treatment_learner", "censoring_learner")
    }
    result = _quiet_fit(_fresh(kind, **seeded), LongitudinalData.from_frame(panel(), **ROLES))
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


# ---------------------------------------------------------------- WK: one key per cell


def _sequential(kind: type[Any], **overrides: Any) -> Any:
    """``SDR`` or ``IPSI`` on the repeated panel, with parametric learners and one fold."""
    regimens = [Incremental(0.5), Incremental(2.0)] if kind is IPSI else {"never": 0, "always": 1}
    settings: dict[str, Any] = {
        "reference": None if kind is IPSI else "never",
        "outcome_learner": LogisticRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "censoring_learner": LogisticRegression(),
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return kind(regimens, **settings)


def _linear_in_horizon(label: Any, horizon: int, frame: Any) -> np.ndarray:
    n = len(frame)
    return np.column_stack([np.ones(n), np.full(n, float(horizon))])


@pytest.mark.parametrize("kind", [SDR, IPSI], ids=["SDR", "IPSI"])
@pytest.mark.parametrize("working_model", [False, True], ids=["cells", "msm"])
def test_wk_a_fit_holds_one_key_for_each_plan_and_horizon(
    kind: type[Any], working_model: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated outcome is not survival, yet every horizon needs its own fit key.

    Both kinds are added to the allowlist here, because the allowlist refuses ``SDR`` until its
    own evidence exists.  Keyed by ``is_survival``, every horizon of a plan shares one
    key: the fits of the later horizons overwrite the first, and a working model projects the
    same fit for every cell.
    """
    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", frozenset({"tmle", "sdr", "ipsi"}))
    msm = (
        MSM(design=_linear_in_horizon, terms=("(intercept)", "t"), design_kind="known")
        if working_model
        else None
    )
    data = LongitudinalData.from_frame(panel(), **ROLES)
    estimator = _sequential(kind, msm=msm, **({"reference": None} if working_model else {}))
    result = _quiet_fit(estimator, data)
    labels = [label for label, _ in result.config.plan_fingerprints]
    assert len(labels) == 2
    expected = {f"{label}{HORIZON_INFIX}{horizon}" for label in labels for horizon in (1, 2, 3)}
    assert set(result.fits) == expected
    assert len(result.fits) == 6
    horizons = {key: fit.horizon for key, fit in result.fits.items()}
    assert sorted(horizons.values()) == [1, 1, 2, 2, 3, 3]
    for key, horizon in horizons.items():
        assert key.endswith(f"{HORIZON_INFIX}{horizon}")
    estimates = {name for name in result.estimates if name.startswith("ey_regimen[")}
    if not working_model:
        assert len(estimates) == 6


# ---------------------------------------------------------------- W16: the range of a horizon


def _continuous() -> tuple[Any, Any]:
    """The exact ``T = 2`` continuous panel and the pooled scaler a fit derives from it.

    ``Y1`` spans ``[-0.5, 2.0]`` and ``Y2`` spans ``[0, 1]``.
    """
    data = LongitudinalData.from_frame(laws.CONTINUOUS.frame(), **LAW_COLUMNS)
    return data, OutcomeScaler.from_outcome(data.observed_outcomes(), None)


def _level(name: str, psi: float, n: int) -> Any:
    rng = np.random.default_rng(0)
    return make_estimate(name, psi, rng.normal(size=n), n=n, scale="level")


def _warned(
    data: Any, scaler: Any, levels: dict[str, tuple[float, int]], q_bounds: Any = None
) -> list[str]:
    """The ``ParameterSpaceWarning`` messages for ``levels``: ``name -> (psi, horizon)``."""
    estimates = {name: _level(name, psi, data.n) for name, (psi, _) in levels.items()}
    ranges = {
        name: sequential._level_range(data, scaler, q_bounds, horizon)
        for name, (_, horizon) in levels.items()
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sequential._warn_parameter_space(estimates, ranges, estimator="sdr")
    return [str(item.message) for item in caught if item.category is ParameterSpaceWarning]


def test_w16_a_level_outside_its_own_horizon_range_warns() -> None:
    """``E[Y_2^d] = 1.5`` is outside ``Y2``'s range and inside ``Y1``'s: it warns.

    The scale pools both nodes (W14), so a warning that read the scaler's range would stay
    silent.  The message names the range of ``Y2``, padded by 10% as the derived scaler pads.
    """
    data, scaler = _continuous()
    assert scaler.lower < -0.5 and scaler.upper > 2.0
    ranges = {h: sequential._level_range(data, scaler, None, h) for h in (1, 2)}
    assert ranges[2] == pytest.approx((-0.1, 1.1))
    assert ranges[1] == pytest.approx((-0.75, 2.25))
    assert ranges[2] != (scaler.lower, scaler.upper)
    (message,) = _warned(data, scaler, {"ey_regimen[always @ t=2]": (1.5, 2)})
    assert message.startswith("ey_regimen[always @ t=2] = 1.5: outside the outcome range")
    assert "[-0.1, 1.1]" in message
    assert _warned(data, scaler, {"ey_regimen[always @ t=1]": (1.5, 1)}) == []
    assert _warned(data, scaler, {"ey_regimen[always @ t=2]": (1.0, 2)}) == []


def test_w16_each_level_names_the_range_it_left() -> None:
    data, scaler = _continuous()
    (message,) = _warned(
        data,
        scaler,
        {"ey_regimen[never @ t=1]": (2.5, 1), "ey_regimen[never @ t=2]": (1.5, 2)},
    )
    assert "ey_regimen[never @ t=1] = 2.5: outside the outcome range [-0.75, 2.25]" in message
    assert "ey_regimen[never @ t=2] = 1.5: outside the outcome range [-0.1, 1.1]" in message


def test_w16_a_declared_range_replaces_the_derived_one() -> None:
    data, scaler = _continuous()
    scaler = OutcomeScaler.from_outcome(data.observed_outcomes(), (-1.0, 3.0))
    assert sequential._level_range(data, scaler, (-1.0, 3.0), 2) == (-1.0, 3.0)
    assert _warned(data, scaler, {"ey_regimen[always @ t=2]": (1.5, 2)}, (-1.0, 3.0)) == []
    (message,) = _warned(data, scaler, {"ey_regimen[always @ t=2]": (3.5, 2)}, (-1.0, 3.0))
    assert "[-1, 3]" in message


def test_w16_a_binary_outcome_keeps_the_unit_interval() -> None:
    data = LongitudinalData.from_frame(laws.BINARY.frame(), **LAW_COLUMNS)
    for horizon in (1, 2):
        assert sequential._level_range(data, OutcomeScaler.identity(), None, horizon) == (0.0, 1.0)


def test_w16_the_fit_hands_each_level_its_own_horizon_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The call site builds the range map from the fit, one range for each horizon."""
    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", frozenset({"tmle", "sdr"}))
    seen: dict[str, tuple[float, float]] = {}

    def spy(estimates: Any, ranges: Any, *, estimator: str) -> None:
        seen.update(ranges)

    monkeypatch.setattr(sequential, "_warn_parameter_space", spy)
    cells = laws.CellMeans
    fit = SDR(
        {"never": 0, "always": 1},
        reference="never",
        outcome_learner=cells(),
        pseudo_learner=cells(),
        treatment_learner=cells(),
        censoring_learner=cells(),
        n_folds=1,
        g_bounds=(1e-8, 1.0 - 1e-8),
        simultaneous=False,
    ).fit(laws.CONTINUOUS.frame(), **LAW_COLUMNS)
    assert set(seen) == set(fit.estimates)
    for name, (low, high) in seen.items():
        horizon = 1 if "t=1" in name else 2
        expected = (-0.75, 2.25) if horizon == 1 else (-0.1, 1.1)
        assert (low, high) == pytest.approx(expected), name
