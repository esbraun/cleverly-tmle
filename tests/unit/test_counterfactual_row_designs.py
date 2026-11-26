"""Counterfactual designs built per fold are the full designs' rows, byte for byte.

:meth:`CausalData.counterfactual_rows` replaced ``counterfactual_design`` wherever a cross-fit
predicts at a counterfactual treatment, so that a fit no longer holds one ``(n, width)`` copy
of the covariates per arm, level and policy for the whole outcome fit (X106 part K1).  The
change is meant to move no number.  These tests hold it to that: the rows themselves, then the
fitted nuisances and estimates of whole fits against the materialized route, then the
refusals, then a deliberately wrong row builder that the fit-level comparison must catch.
"""

from __future__ import annotations

import dataclasses
import pickle
import tracemalloc
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CapabilityError, DataError
from cleverly.data import CausalData
from cleverly.data.causal_data import RowDesign
from cleverly.dose_response import DoseResponse
from cleverly.estimators import DRTMLE, TMLE, AdaptiveIPTW, DoseResponseTMLE, _nuisance
from cleverly.interventions import LearnedRule, Shift
from cleverly.interventions.tuning import DoseQuantile

LOGIT = LogisticRegression(max_iter=1000)


def _frame(n: int = 600, *, arms: int = 2, seed: int = 11, p: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, p))
    if arms == 2:
        a = rng.binomial(1, expit(0.3 * w[:, 0] - 0.2 * w[:, 1]))
    else:
        logits = np.column_stack([np.zeros(n)] + [0.2 * k * w[:, 0] for k in range(1, arms)])
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        a = (prob.cumsum(axis=1) > rng.uniform(size=(n, 1))).argmax(axis=1)
    y = rng.binomial(1, expit(-0.3 + 0.4 * a + 0.3 * w[:, 0] - 0.2 * w[:, 1]))
    frame = pd.DataFrame(w, columns=[f"W{k}" for k in range(p)])
    frame["A"] = a
    frame["Y"] = y.astype(float)
    return frame


def _covariates(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("W")]


def _layout(array: np.ndarray) -> tuple[bool, bool]:
    """Memory order, which a learner's BLAS call can read: it must match too."""
    return bool(array.flags.c_contiguous), bool(array.flags.f_contiguous)


# --- the rows -----------------------------------------------------------------------------


def _row_cases() -> Iterator[tuple[str, CausalData, Any, float | None]]:
    binary = _frame()
    yield (
        "binary",
        CausalData.from_frame(binary, outcome="Y", treatment="A", covariates=_covariates(binary)),
        1.0,
        None,
    )
    three = _frame(arms=3)
    data3 = CausalData.from_frame(three, outcome="Y", treatment="A", covariates=_covariates(three))
    yield "three arms", data3, 2.0, None
    per_row = np.asarray(three["A"], dtype=float)[::-1].copy()
    yield "per-row arms", data3, per_row, None
    dose = _frame()
    rng = np.random.default_rng(5)
    dose["A"] = np.where(rng.uniform(size=len(dose)) < 0.2, 0.0, rng.uniform(0.0, 3.0, len(dose)))
    data_dose = CausalData.from_frame(
        dose,
        outcome="Y",
        treatment="A",
        covariates=_covariates(dose),
        treatment_kind="continuous",
    ).with_dose_declarations(dose_atoms=(0.0,))
    yield "dose atom, per-row shift", data_dose, np.asarray(dose["A"]) + 0.5, None
    yield "dose atom, fixed dose", data_dose, 0.0, None
    yield "dose atom, fixed dose off the atom", data_dose, 1.5, None
    plain = CausalData.from_frame(
        dose, outcome="Y", treatment="A", covariates=_covariates(dose), treatment_kind="continuous"
    )
    yield "continuous, fixed dose", plain, 1.5, None
    cde = _frame()
    cde["Z"] = np.random.default_rng(6).binomial(1, 0.5, len(cde)).astype(float)
    data_z = CausalData.from_frame(
        cde, outcome="Y", treatment="A", covariates=_covariates(cde), intermediate="Z"
    )
    yield "intermediate level", data_z, 1.0, 0.0


@pytest.mark.parametrize("case", list(_row_cases()), ids=lambda case: case[0])
def test_rows_are_the_full_designs_rows(case: tuple[str, CausalData, Any, float | None]) -> None:
    _, data, value, level = case
    full = data.counterfactual_design(value, intermediate_value=level)
    lazy = data.counterfactual_rows(value, intermediate_value=level)
    assert isinstance(lazy, RowDesign)
    assert lazy.shape == full.shape
    rng = np.random.default_rng(1)
    indices = [
        np.arange(data.n),
        rng.permutation(data.n)[: data.n // 3],
        np.array([7]),
        np.array([], dtype=np.intp),
        np.sort(rng.choice(data.n, 50, replace=False)),
    ]
    for index in indices:
        built = lazy.rows(index)
        expected = full[index]
        assert built.dtype == expected.dtype
        assert built.shape == expected.shape
        assert _layout(built) == _layout(expected)
        assert built.tobytes() == expected.tobytes()
    whole = lazy.rows(None)
    assert whole.tobytes() == full.tobytes()
    assert _layout(whole) == _layout(full)


@pytest.mark.parametrize("case", list(_row_cases()), ids=lambda case: case[0])
def test_a_pickled_row_design_builds_the_same_rows(
    case: tuple[str, CausalData, Any, float | None],
) -> None:
    _, data, value, level = case
    lazy = data.counterfactual_rows(value, intermediate_value=level)
    back = pickle.loads(pickle.dumps(lazy))
    index = np.array([4, 0, 9])
    assert back.rows(index).tobytes() == lazy.rows(index).tobytes()
    assert back.rows(None).tobytes() == lazy.rows(None).tobytes()
    assert _layout(back.rows(None)) == _layout(lazy.rows(None))


def test_a_constant_row_design_pickles_as_one_row() -> None:
    # A worker process is sent one row per key, not n: the covariates dominate the payload.
    frame = _frame(n=20_000)
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=_covariates(frame))
    constant = len(pickle.dumps(data.counterfactual_rows(1.0)))
    per_row = len(pickle.dumps(data.counterfactual_rows(np.ones(data.n))))
    covariates = len(pickle.dumps(data.covariates))
    assert constant < covariates + 2_000
    assert per_row > covariates + 8 * data.n


def test_the_row_design_holds_no_copy_of_the_covariates() -> None:
    frame = _frame()
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=_covariates(frame))
    design = data.counterfactual_rows(1.0)
    assert isinstance(design, RowDesign)
    assert design.covariates is data.covariates


def test_a_constant_value_holds_one_treatment_row() -> None:
    # X106 D2a: one value for everybody keeps one row of the treatment block, broadcast and
    # read-only, so a dose-response fit's thousands of node designs hold O(K) bytes.
    frame = _frame()
    frame["A"] = np.random.default_rng(3).uniform(0.0, 3.0, len(frame))
    data = CausalData.from_frame(
        frame,
        outcome="Y",
        treatment="A",
        covariates=_covariates(frame),
        treatment_kind="continuous",
    )
    for design in (data.counterfactual_rows(1.5), data.counterfactual_rows(np.array([1.5]))):
        assert isinstance(design, RowDesign)
        assert design.treatment.strides[0] == 0
        assert not design.treatment.flags.writeable
        with pytest.raises(ValueError):
            design.treatment[0, 0] = 2.0
        assert design.rows(np.array([0, 3])).flags.writeable
    per_row = data.counterfactual_rows(np.asarray(frame["A"]) + 0.1)
    assert per_row.treatment.strides[0] != 0


# --- refusals -----------------------------------------------------------------------------


def _refusing_cases() -> Iterator[tuple[str, CausalData, Any, float | None]]:
    binary = _frame()
    data = CausalData.from_frame(binary, outcome="Y", treatment="A", covariates=_covariates(binary))
    yield "unknown arm", data, 3.0, None
    yield "wrong length", data, np.zeros(data.n - 1), None
    yield "level without an intermediate", data, 1.0, 1.0
    dose = _frame()
    dose["A"] = np.random.default_rng(2).uniform(0.0, 3.0, len(dose))
    data_dose = CausalData.from_frame(
        dose, outcome="Y", treatment="A", covariates=_covariates(dose), treatment_kind="continuous"
    )
    yield "non-finite dose", data_dose, np.full(data_dose.n, np.nan), None
    yield "non-finite constant dose", data_dose, np.nan, None
    yield "constant level without an intermediate", data_dose, 1.0, 1.0


@pytest.mark.parametrize("case", list(_refusing_cases()), ids=lambda case: case[0])
def test_refusals_are_the_full_designs_refusals(
    case: tuple[str, CausalData, Any, float | None],
) -> None:
    _, data, value, level = case
    with pytest.raises(Exception) as full:
        data.counterfactual_design(value, intermediate_value=level)
    with pytest.raises(Exception) as lazy:
        data.counterfactual_rows(value, intermediate_value=level)
    assert type(lazy.value) is type(full.value)
    assert str(lazy.value) == str(full.value)
    assert isinstance(full.value, (DataError, CapabilityError))


# --- whole fits against the materialized route -------------------------------------------


def _materialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every counterfactual design back through the full matrix."""

    def full(self: CausalData, value: Any, *, intermediate_value: float | None = None) -> Any:
        return self.counterfactual_design(value, intermediate_value=intermediate_value)

    monkeypatch.setattr(CausalData, "counterfactual_rows", full)


def _arrays(value: Any, path: str = "") -> Iterator[tuple[str, bytes]]:
    """Every float or integer array reachable from a result, by path."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fiub":
            yield path, value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            try:
                field = getattr(value, item.name)
            except CapabilityError:  # an outcome-free fit's outcome accessors refuse
                continue
            yield from _arrays(field, f"{path}.{item.name}")
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            yield from _arrays(value[key], f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        for k, item in enumerate(value):
            yield from _arrays(item, f"{path}[{k}]")
    elif isinstance(value, (float, int, np.floating, np.integer)):
        yield path, repr(float(value)).encode()


def _fingerprint(result: Any) -> dict[str, bytes]:
    results = result.results.values() if hasattr(result, "results") else [result]
    out: dict[str, bytes] = {}
    for k, single in enumerate(results):
        for name, estimate in single.estimates.items():
            out.update(
                {f"{k}:{name}{p}": b for p, b in _arrays(estimate)}
                | {f"{k}:{name}.psi": repr(estimate.psi).encode()}
            )
        nuisances = getattr(single, "nuisances", None)
        if nuisances is None:
            nuisance = getattr(single, "nuisance", None)
            nuisances = () if nuisance is None else (nuisance,)
        for nuisance in nuisances:
            out.update({f"{k}:nuisance{p}": b for p, b in _arrays(nuisance)})
    return out


def _binary_fit() -> Any:
    frame = _frame()
    return TMLE(
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        n_folds=3,
        random_state=4,
        estimands=("ate", "att", "rr"),
        simultaneous=False,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _single_fold_fit() -> Any:
    frame = _frame()
    return TMLE(
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        cross_fit=False,
        random_state=4,
        estimands=("ate",),
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _three_arm_fit() -> Any:
    frame = _frame(arms=3)
    return TMLE(
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        n_folds=3,
        random_state=4,
        simultaneous=False,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _intermediate_fit() -> Any:
    frame = _frame()
    rng = np.random.default_rng(8)
    frame["Z"] = rng.binomial(1, expit(-0.2 + 0.6 * frame["A"] + 0.3 * frame["W0"])).astype(float)
    return TMLE(
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        intermediate_learner=LOGIT,
        n_folds=3,
        random_state=4,
        estimands=("ate",),
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame), intermediate="Z")


def _missing_outcome_fit() -> Any:
    frame = _frame()
    rng = np.random.default_rng(9)
    frame["R"] = (rng.uniform(size=len(frame)) >= 0.15).astype(float)
    frame.loc[frame["R"] == 0.0, "Y"] = np.nan
    return TMLE(
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        missingness_learner=LOGIT,
        n_folds=3,
        random_state=4,
        estimands=("ate",),
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame), delta="R")


def _parallel_fit() -> Any:
    frame = _frame()
    return TMLE(
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        n_folds=3,
        random_state=4,
        estimands=("ate",),
        n_jobs=2,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _shift_fit() -> Any:
    frame = _frame()
    rng = np.random.default_rng(10)
    frame["A"] = rng.normal(1.0 + 0.3 * frame["W0"], 0.7)
    return TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LinearRegression(),
        policies=[Shift(0.4, cap=None)],
        ratio="classifier",
        n_folds=3,
        random_state=4,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _foldwise_drtmle_fit() -> Any:
    frame = _frame()
    return DRTMLE(
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=LOGIT,
        reduced_mechanism_residual_learner=LinearRegression(),
        reduced_crossfit="foldwise",
        n_folds=3,
        random_state=4,
        estimands=("ate", "ey1", "ey0"),
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _dose_response_fit() -> Any:
    """Quadrature doses are predicted stacked, many designs per learner call."""
    rng = np.random.default_rng(7)
    a = rng.uniform(0.0, 4.0, 800)
    frame = pd.DataFrame({"A": a, "W0": rng.normal(size=a.size)})
    frame["Y"] = 0.5 * a + 0.3 * frame["W0"] + rng.normal(scale=0.3, size=a.size)
    curve = DoseResponse(doses=(1.0, 2.5), bandwidths=(0.6,), support=(0.5, 3.5))
    return DoseResponseTMLE(
        curve,
        outcome_learner=LinearRegression(),
        treatment_learner=DummyClassifier(strategy="prior"),
        n_folds=2,
        fluctuation="linear",
        q_bounds=(-5, 5),
        n_replicates=100,
        random_state=7,
    ).fit(frame, outcome="Y", treatment="A", covariates=["W0"])


def _learned_rule_fit() -> Any:
    """A plug-in rule reads every fold's model at every row of each arm's design."""
    frame = _frame()
    return TMLE(
        learned_rule=LearnedRule(),
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        n_folds=3,
        random_state=4,
        simultaneous=False,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _adaptive_iptw_fit() -> Any:
    """``fit_fold_mechanisms`` reads the missingness model at each arm's design."""
    frame = _frame()
    rng = np.random.default_rng(12)
    frame["R"] = (rng.uniform(size=len(frame)) >= 0.15).astype(float)
    frame.loc[frame["R"] == 0.0, "Y"] = np.nan
    return AdaptiveIPTW(
        outcome_learner=LinearRegression(),
        treatment_learner=LOGIT,
        missingness_learner=LOGIT,
        n_folds=3,
        random_state=4,
        estimands=("ate",),
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame), delta="R")


def _tuned_cap_fit() -> Any:
    """A data-tuned cap reads the outcome regression at probe doses."""
    frame = _frame()
    rng = np.random.default_rng(13)
    frame["A"] = rng.normal(1.0 + 0.3 * frame["W0"], 0.7)
    return TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LinearRegression(),
        policies=[Shift(0.5, cap=DoseQuantile(0.5), target="population", name="tuned")],
        ratio="classifier",
        n_folds=3,
        random_state=4,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _dose_response_single_fold_fit() -> Any:
    """The stacked path at ``rows=None``: every quadrature design built whole once."""
    rng = np.random.default_rng(14)
    a = rng.uniform(0.0, 4.0, 600)
    frame = pd.DataFrame({"A": a, "W0": rng.normal(size=a.size)})
    frame["Y"] = 0.5 * a + 0.3 * frame["W0"] + rng.normal(scale=0.3, size=a.size)
    curve = DoseResponse(doses=(1.5,), bandwidths=(0.6,), support=(0.5, 3.5))
    return DoseResponseTMLE(
        curve,
        outcome_learner=LinearRegression(),
        treatment_learner=DummyClassifier(strategy="prior"),
        cross_fit=False,
        fluctuation="linear",
        q_bounds=(-5, 5),
        n_replicates=100,
        random_state=7,
    ).fit(frame, outcome="Y", treatment="A", covariates=["W0"])


def _learned_rule_intermediate_fit() -> Any:
    """Rule companions at every ``(arm, level)`` of a binary intermediate."""
    frame = _frame()
    rng = np.random.default_rng(15)
    frame["Z"] = rng.binomial(1, expit(-0.2 + 0.6 * frame["A"] + 0.3 * frame["W0"])).astype(float)
    return TMLE(
        learned_rule=LearnedRule(),
        outcome_learner=LOGIT,
        treatment_learner=LOGIT,
        intermediate_learner=LOGIT,
        n_folds=3,
        random_state=4,
        simultaneous=False,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame), intermediate="Z")


FITS: dict[str, Callable[[], Any]] = {
    "learned rule, intermediate": _learned_rule_intermediate_fit,
    "tuned cap, probe doses": _tuned_cap_fit,
    "dose response, single fold": _dose_response_single_fold_fit,
    "learned rule, plug-in": _learned_rule_fit,
    "adaptive IPTW, missing outcomes": _adaptive_iptw_fit,
    "dose response, stacked": _dose_response_fit,
    "binary": _binary_fit,
    "single fold": _single_fold_fit,
    "three arms": _three_arm_fit,
    "intermediate": _intermediate_fit,
    "missing outcomes": _missing_outcome_fit,
    "process pool": _parallel_fit,
    "policy shift": _shift_fit,
    "foldwise DR-TMLE": _foldwise_drtmle_fit,
}


@pytest.mark.parametrize("name", list(FITS))
def test_fits_are_bitwise_the_materialized_fits(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[int] = []
    rows = RowDesign.rows

    def counting(self: RowDesign, index: Any = None) -> Any:
        built.append(1)
        return rows(self, index)

    with monkeypatch.context() as patch:
        patch.setattr(RowDesign, "rows", counting)
        lazy = _fingerprint(FITS[name]())
    # The row route was taken (a process pool builds its rows in the workers, out of sight).
    assert built or name == "process pool"
    with monkeypatch.context() as patch:
        _materialized(patch)
        full = _fingerprint(FITS[name]())
    assert lazy.keys() == full.keys()
    assert len(lazy) > 4
    assert [key for key in lazy if lazy[key] != full[key]] == []


def test_a_wrong_row_builder_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: rows that keep the observed treatment must move the fit."""
    reference = _fingerprint(_binary_fit())

    def observed_rows(self: RowDesign, index: Any = None) -> Any:
        # The observed design's treatment block in place of the counterfactual one.
        frame_codes = np.asarray(_frame()["A"], dtype=float).reshape(-1, 1)
        treatment = frame_codes if index is None else frame_codes[index]
        covariates = self.covariates if index is None else self.covariates[index]
        return np.hstack([treatment, covariates])

    monkeypatch.setattr(RowDesign, "rows", observed_rows)
    mutated = _fingerprint(_binary_fit())
    assert any(mutated[key] != reference[key] for key in reference)


# --- what it saves ------------------------------------------------------------------------


def _memory_fit() -> Any:
    frame = _frame(20_000, p=50, seed=3)
    return TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LOGIT,
        n_folds=3,
        random_state=4,
        estimands=("ate",),
        simultaneous=False,
    ).fit(frame, outcome="Y", treatment="A", covariates=_covariates(frame))


def _peak(fit: Callable[[], Any]) -> int:
    tracemalloc.start()
    try:
        fit()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_the_counterfactual_designs_are_not_held_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fit peaks at least 60% of the two counterfactual designs below the full route.

    Both routes run here, in one process on one set of library versions, so the bound reads
    the change and not a library's allocations.  The full route holds both ``(n, 1 + p)``
    counterfactual designs through the outcome fit; the row route holds one fold's rows of
    one design at a time.
    """
    n, p = 20_000, 50
    design_bytes = 2 * n * (1 + p) * 8
    lazy = _peak(_memory_fit)
    with monkeypatch.context() as patch:
        _materialized(patch)
        full = _peak(_memory_fit)
    assert full - lazy >= 0.6 * design_bytes


def test_only_the_observed_design_reaches_the_cross_fit_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every counterfactual key of the outcome regression is a ``RowDesign``."""
    seen: list[dict[str, Any]] = []
    original = _nuisance.cross_fit_companion

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(kwargs["predict_designs"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(_nuisance, "cross_fit_companion", spy)
    _binary_fit()
    outcome = [designs for designs in seen if "observed" in designs and len(designs) > 1]
    assert outcome, "the outcome regression was not seen"
    for designs in outcome:
        assert isinstance(designs["observed"], np.ndarray)
        assert all(
            isinstance(design, RowDesign) for key, design in designs.items() if key != "observed"
        )


#: Fits whose counterfactual designs reach more than the outcome regression, and the least
#: number of cross-fits each must hand a ``RowDesign``.
SECONDARY_ROUTES = {
    "missing outcomes": 2,  # the outcome regression and the missingness mechanism
    "intermediate": 2,  # the outcome regression and the intermediate mechanism
    "adaptive IPTW, missing outcomes": 1,  # fit_fold_mechanisms' missingness slabs
    "foldwise DR-TMLE": 2,  # the outcome regression and fit_inner_designs
}


@pytest.mark.parametrize("name", list(SECONDARY_ROUTES))
def test_every_reader_of_the_designs_gets_rows(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each cross-fit that predicts at a counterfactual treatment is handed a ``RowDesign``.

    The whole-fit equality above would pass if one of these readers went back to whole
    designs, so the route is counted per reader here.
    """
    routed: list[int] = []
    original = _nuisance.cross_fit_companion

    def spy(*args: Any, **kwargs: Any) -> Any:
        designs = kwargs["predict_designs"].values()
        if any(isinstance(design, RowDesign) for design in designs):
            routed.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_nuisance, "cross_fit_companion", spy)
    FITS[name]()
    assert len(routed) >= SECONDARY_ROUTES[name]


def test_a_process_pool_fit_takes_the_row_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool builds rows in its workers; the parent must still have made row designs."""
    made: list[int] = []
    original = CausalData.counterfactual_rows

    def counting(self: CausalData, *args: Any, **kwargs: Any) -> Any:
        made.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CausalData, "counterfactual_rows", counting)
    _parallel_fit()
    assert made
