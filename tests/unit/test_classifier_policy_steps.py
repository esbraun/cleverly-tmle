"""The classifier route's policy steps (``classifier_ratio(steps=)``).

Díaz, Williams, Hoffman and Schenck (2023, Equation 3) put every jump of the true ratio in
the dose at a jump of a smooth piece's image indicator ``s_k`` or an identity piece's
membership ``t_j``.  A smooth stacked classifier cannot place such a jump from the dose
alone, so the route appends those indicators to the dose (``_policy_steps``).

- W11a: on five laws the steps lower the ratio error where the policy makes a jump, and
  leave it where it makes none; each kind of column has a mutation that fails.
- W11b: fold-tuned caps use each fold's map, and empirical bounds read each fold's fitted
  training rows, at risk at a longitudinal node.
- W11c: the column order ``[A, atoms, steps, H]``, and the ``policy_steps = "omit"`` opt-out.
- W11d: the longitudinal node's columns are its lowered maps' indicators.
- W11e: a constant column, or one equal to an atom indicator, is dropped per fold.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import cleverly.interventions.policy as policy_module
import cleverly.learners.density_ratio as density_ratio
from cleverly.data import CausalData
from cleverly.interventions import ModifiedPolicy, Piece, Shift, Support
from cleverly.interventions.policy import PolicySet
from cleverly.learners.crossfit import make_folds

from .. import mtp_density_tails_law as law

pytestmark = pytest.mark.xdist_group("classifier_policy_steps")

#: Law G's population 0.9 quantile of ``A``, a cap inside the support.
CAP_G = 3.30


class Logit(LogisticRegression):  # type: ignore[misc]
    """An effectively unpenalised logistic regression, the generic classifier."""

    def __init__(self, C: float = 1e6, max_iter: int = 5000) -> None:
        super().__init__(C=C, max_iter=max_iter)


class OmitLogit(Logit):
    """The same classifier, opted out of the step columns."""

    policy_steps = "omit"


def _one(b: Any, h: Any) -> np.ndarray:
    return np.ones_like(np.asarray(b, dtype=float))


def _plus(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + 1.0


def _minus(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - 1.0


def _upper_b(h: Any) -> np.ndarray:
    return law.upper("B", np.asarray(h["W1"], dtype=float), np.asarray(h["W2"], dtype=float))


def _knee_b(h: Any) -> np.ndarray:
    return _upper_b(h) - 1.0


P2 = ModifiedPolicy(
    "p2",
    pieces=(Piece(-np.inf, _knee_b, _plus, _minus, _one), Piece(_knee_b, np.inf)),
    policy_kind="known",
)
TNEG = ModifiedPolicy(
    "tneg",
    pieces=(Piece(-np.inf, law.LOW + 1.0), Piece(law.LOW + 1.0, np.inf, _minus, _plus, _one)),
    policy_kind="known",
)

#: Per case: the law drawn, the policy, the declared support or None, and the draw seed base.
CASES: dict[str, tuple[str, Any, Any, int]] = {
    "T": ("T", Shift(1.0, cap=law.HIGH), Support(law.LOW, law.HIGH), 1),
    "Gcap": ("G", Shift(1.0, cap=CAP_G), None, 7),
    "Bp2": ("B", P2, Support(0.0, _upper_b), 9),
    "Gneg": ("G", Shift(-1.0, cap=None), None, 8),
    "Tneg": ("T", TNEG, Support(law.LOW, law.HIGH), 2),
}


def _r0(case: str, a: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    name = CASES[case][0]

    def g(v: np.ndarray) -> np.ndarray:
        return law.pdf(name, v, w1, w2)

    if case in ("T", "Bp2"):
        return law.r0(name, a, w1, w2)
    if case == "Gcap":
        return g(a - 1.0) / g(a) * (a <= CAP_G) + (a > CAP_G - 1.0)
    if case == "Gneg":
        return g(a + 1.0) / g(a)
    return g(a + 1.0) / g(a) * (a + 1.0 <= law.HIGH) + (a < law.LOW + 1.0)


def _data(case: str, n: int, draw: int) -> tuple[CausalData, pd.DataFrame]:
    name, _, support, seed = CASES[case]
    frame = law.draw(name, n, 9400 * n + draw + seed * 100000)
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_kind="continuous"
    )
    if support is not None:
        data = data.with_dose_declarations(support)
    return data, frame


def _ratio(case: str, draw: int, learner: Any, n: int = 2000) -> tuple[np.ndarray, pd.DataFrame]:
    data, frame = _data(case, n, draw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluated = PolicySet.evaluate_by_classifier(
            (CASES[case][1],), data, learner, make_folds(n, 5, random_state=0)
        )
    return np.asarray(evaluated.ratio[:, 0], dtype=float), frame


def _error(case: str, draw: int, learner: Any) -> float:
    rn, frame = _ratio(case, draw, learner)
    w1, w2, a, _ = law.columns(frame)
    return float(np.sqrt(np.mean((rn - _r0(case, a, w1, w2)) ** 2)))


def _top_bias(draw: int, learner: Any) -> float:
    rn, frame = _ratio("Tneg", draw, learner)
    w1, w2, a, _ = law.columns(frame)
    top = a > law.HIGH - 1.0
    return float(np.mean((rn - _r0("Tneg", a, w1, w2))[top]))


def _median(case: str, learner: Any) -> float:
    return float(np.median([_error(case, d, learner) for d in range(5)]))


# ------------------------------------------------------------------ W11a


def _mutated_steps(kind: str) -> Any:
    """A ``_policy_steps`` with one kind of column removed or rewritten: ``image`` drops the
    smooth pieces' image indicators, ``identity`` the identity pieces', ``support`` the
    support term of ``s_k``, and ``rev3`` types the steps on the dose alone (the region below
    the smallest dose plus one, and the held region above the pooled knee)."""

    def build(branches_by_fold: Any, frame: Any, n: int, bounds: Any, observed: Any) -> Any:
        natural = np.asarray(observed, dtype=float)
        maps = [branch.lowered(frame, n) for branch in branches_by_fold[0]]
        if kind == "rev3":
            capped = hasattr(branches_by_fold[0][0], "shift")
            knee = CAP_G - 1.0 if capped else float(np.median(_knee_b(frame())))

            def dose_only(dose: Any, rows: Any, fold: int, train: Any) -> np.ndarray:
                x = np.asarray(dose, dtype=float)
                bottom = x < natural[train].min() + 1.0
                return np.column_stack([bottom, x > knee]).astype(float)

            return dose_only

        def steps(dose: Any, rows: Any, fold: int, train: Any) -> np.ndarray:
            low, high = (natural[train].min(), natural[train].max()) if bounds is None else bounds
            x = natural.copy()
            x[rows] = dose
            columns = []
            for lowered in maps:
                if kind != "image":
                    for piece in lowered.smooth:
                        with np.errstate(all="ignore"):
                            source = np.asarray(piece.inverse(x), dtype=float)
                        ok = np.isfinite(source)
                        safe = np.where(ok, source, x)
                        inside = ok & np.asarray(piece.inside(safe), dtype=bool)
                        if kind != "support":
                            inside &= (safe >= low) & (safe <= high)
                        columns.append(inside[rows])
                if kind != "identity":
                    columns.extend(np.asarray(m(x), dtype=bool)[rows] for m in lowered.identity)
            if not columns:
                return np.zeros((np.size(rows), 0))
            return np.column_stack(columns).astype(float)

        return steps

    return build


def test_w11a_the_steps_lower_the_error_where_the_policy_jumps() -> None:
    for case in ("T", "Gcap", "Bp2"):
        steps, plain = _median(case, Logit()), _median(case, OmitLogit())
        assert steps <= 0.75 * plain, (case, steps, plain)
    steps, plain = _median("Gneg", Logit()), _median("Gneg", OmitLogit())
    assert abs(steps / plain - 1.0) <= 0.10, (steps, plain)
    assert max(abs(_top_bias(d, Logit())) for d in range(5)) <= 0.05


@pytest.mark.parametrize(
    ("kind", "cases"),
    [
        ("image", ("Gcap",)),
        ("identity", ("T", "Gcap", "Bp2")),
        ("support", ("T",)),
        ("rev3", ("Gcap", "Bp2")),
    ],
)
def test_w11a_each_kind_of_column_is_needed(
    kind: str, cases: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = {case: _median(case, Logit()) for case in cases}
    monkeypatch.setattr(policy_module, "_policy_steps", _mutated_steps(kind))
    for case in cases:
        mutated = _median(case, Logit())
        assert mutated >= 1.3 * shipped[case], (kind, case, mutated, shipped[case])
    if kind == "support":
        assert np.median([_top_bias(d, Logit()) for d in range(5)]) >= 0.15


# ------------------------------------------------------------------ W11b


def _record_steps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Every step evaluation of a fold's fit: ``(fold, train, dose, columns)``."""
    calls: list[tuple[Any, ...]] = []
    real = density_ratio._steps_at

    def record(steps: Any, dose: Any, rows: Any, fold: int, train: Any) -> np.ndarray:
        out = real(steps, dose, rows, fold, train)
        if np.array_equal(rows, train):
            calls.append((fold, np.asarray(train), np.asarray(dose, dtype=float), out))
        return out

    monkeypatch.setattr(density_ratio, "_steps_at", record)
    return calls


def _whole_sample_bounds(monkeypatch: pytest.MonkeyPatch, rows: Any) -> None:
    """The mutation: empirical bounds read from ``rows`` instead of the fold's own."""
    real = policy_module._policy_steps

    def build(*args: Any) -> Any:
        steps = real(*args)
        return lambda dose, at, fold, train: steps(dose, at, fold, rows)

    monkeypatch.setattr(policy_module, "_policy_steps", build)


def _gneg_bounds_hold(monkeypatch: pytest.MonkeyPatch) -> bool:
    calls = _record_steps(monkeypatch)
    data, frame = _data("Gneg", 400, 0)
    a = frame["A"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PolicySet.evaluate_by_classifier(
            (Shift(-1.0, cap=None),), data, Logit(), make_folds(400, 5, random_state=0)
        )
    held = []
    for _, train, dose, columns in calls:
        low, high = a[train].min(), a[train].max()
        expected = (dose + 1.0 >= low) & (dose + 1.0 <= high)
        held.append(np.array_equal(columns[:, 0], expected.astype(float)))
    return len(calls) == 10 and all(held)


def test_w11b_empirical_bounds_read_the_fold_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _gneg_bounds_hold(monkeypatch)


def test_w11b_bounds_over_the_whole_sample_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _whole_sample_bounds(monkeypatch, np.arange(400))
    assert not _gneg_bounds_hold(monkeypatch)


def test_w11b_each_fold_lowers_its_own_tuned_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_steps(monkeypatch)
    data, _ = _data("Gcap", 400, 0)
    caps = (2.6, 2.9, 3.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PolicySet.evaluate_by_classifier(
            (Shift(1.0, cap=CAP_G),),
            data,
            Logit(),
            make_folds(400, 3, random_state=0),
            by_fold=[(Shift(1.0, cap=cap),) for cap in caps],
        )
    assert {fold for fold, *_ in calls} == {0, 1, 2}
    for fold, _, dose, columns in calls:
        np.testing.assert_array_equal(columns[:, 1], (dose > caps[fold] - 1.0).astype(float))


def _censored_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Two continuous doses on ``[4, 10]`` with monotone censoring after each node."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.uniform(4.0, 10.0, n)
    c1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.5 + 0.3 * w)))).astype(float)
    l2 = rng.normal(0.1 * a1, 1.0)
    a2 = np.clip(rng.uniform(4.0, 10.0, n) + 0.1 * l2, 4.0, 10.0)
    c2 = rng.binomial(1, 0.9, n).astype(float)
    y = rng.normal(0.1 * a1 + 0.1 * a2 + w, 1.0)
    alive1, alive2 = c1 == 1.0, (c1 == 1.0) & (c2 == 1.0)
    return pd.DataFrame(
        {
            "W": w,
            "A1": a1,
            "C1": c1,
            "L2": np.where(alive1, l2, np.nan),
            "A2": np.where(alive1, a2, np.nan),
            "C2": np.where(alive1, c2, np.nan),
            "Y": np.where(alive2, y, np.nan),
        }
    )


def _node_two_bounds_hold(monkeypatch: pytest.MonkeyPatch) -> bool:
    from sklearn.linear_model import LinearRegression

    from cleverly.longitudinal import LTMLE, DynamicRegimen

    calls = _record_steps(monkeypatch)
    frame = _censored_panel()
    up = Shift(1.0, cap=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LTMLE(
            {"up": DynamicRegimen("up", (up, up))},
            ratio="classifier",
            outcome_learner=LinearRegression(),
            treatment_learner=Logit(),
            censoring_learner=LogisticRegression(),
            q_bounds=(-20.0, 20.0),
            n_folds=3,
            random_state=0,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
            continuous_treatment=["A1", "A2"],
        )
    a2 = frame["A2"].to_numpy()
    node_two = [c for c in calls if np.array_equal(c[2], np.nan_to_num(a2, nan=0.0)[c[1]])]
    held = []
    for _, train, dose, columns in node_two:
        assert np.all(np.isfinite(a2[train]))
        low, high = a2[train].min(), a2[train].max()
        expected = (dose - 1.0 >= low) & (dose - 1.0 <= high)
        held.append(np.array_equal(columns[:, 0], expected.astype(float)))
    return len(node_two) == 3 and all(held)


def test_w11b_a_node_reads_its_bounds_from_the_rows_at_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _node_two_bounds_hold(monkeypatch)


def test_w11b_bounds_that_read_the_filler_dose_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: a bound over every training row reads node 2's filler 0 of the censored
    rows, so the bottom step leaves the data."""
    real = policy_module._policy_steps

    def build(branches: Any, frame: Any, n: int, bounds: Any, observed: Any) -> Any:
        steps = real(branches, frame, n, bounds, observed)
        filler = np.flatnonzero(np.asarray(observed) == 0.0)
        return lambda dose, at, fold, train: steps(dose, at, fold, np.union1d(train, filler))

    import cleverly.longitudinal.sequential as sequential

    monkeypatch.setattr(sequential, "_policy_steps", build)
    assert not _node_two_bounds_hold(monkeypatch)


# ------------------------------------------------------------------ W11c


def _designs(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Every stacked design a classifier of the ratio receives."""
    seen: list[np.ndarray] = []
    real = density_ratio.fit_learner

    def record(learner: Any, design: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(np.array(design, dtype=float, copy=True))
        return real(learner, design, *args, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    return seen


def test_w11c_a_generic_learner_receives_dose_atoms_steps_then_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cleverly.learners import Folds

    seen = _designs(monkeypatch)
    frame = law.draw("G", 400, 3)
    # An atom above the cap, which the policy holds in place.
    frame.loc[frame.index[::5], "A"] = 5.0
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_kind="continuous"
    ).with_dose_declarations(dose_atoms=(5.0,))
    policy = Shift(1.0, cap=CAP_G)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PolicySet.evaluate_by_classifier((policy,), data, Logit(), Folds.single(400))
    (design,) = seen
    a = frame["A"].to_numpy()
    assigned = np.where(a + 1.0 <= CAP_G, a + 1.0, a)
    stacked = np.concatenate([a, assigned])
    rows = np.arange(400)
    (branch,) = [b for _, b in policy_module.policy_branches(policy)]
    steps = policy_module._policy_steps([[branch]], lambda: frame[["W1", "W2"]], 400, None, a)
    expected_steps = np.vstack([steps(a, rows, 0, rows), steps(assigned, rows, 0, rows)])
    np.testing.assert_array_equal(design[:, 0], stacked)
    np.testing.assert_array_equal(design[:, 1], (stacked == 5.0).astype(float))
    np.testing.assert_array_equal(design[:, 2:4], expected_steps)
    np.testing.assert_array_equal(design[:, 4:], np.vstack([frame[["W1", "W2"]]] * 2))


def _pricing_designs(
    learner: Any, monkeypatch: pytest.MonkeyPatch, steps: bool
) -> list[Any] | None:
    """The designs of a law P fit, or ``None`` when the learner cannot read them."""
    import cleverly.longitudinal.sequential as sequential

    from ..studies import canonical_mtp_pricing as pricing
    from ..studies import mtp_pricing_common as common

    with monkeypatch.context() as patch:
        seen = _designs(patch)
        if not steps:
            patch.setattr(sequential, "_node_steps", lambda *args: None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pricing.fit(
                    common.sample(300, 5, 3),
                    3,
                    (common.REFERENCE, "price_up"),
                    treatment_learner=learner,
                    n_folds=2,
                )
            except ValueError:
                return None
    return seen


def _band_designs(monkeypatch: pytest.MonkeyPatch, steps: bool) -> list[Any] | None:
    import cleverly.longitudinal.sequential as sequential

    from ..studies import refuter_calibration_band as band

    with monkeypatch.context() as patch:
        seen = _designs(patch)
        if not steps:
            patch.setattr(sequential, "_node_steps", lambda *args: None)
            patch.setattr(policy_module, "_policy_steps", lambda *args: None)
        try:
            band.fit_cell("PJ1", band.draw_point(300, 4), 0)
            band.fit_cell("LJ1", band.draw_panel(300, 4), 0)
        except ValueError:
            return None
    return seen


def _identical(left: list[Any] | None, right: list[Any] | None) -> bool:
    if left is None or right is None:
        return False
    return len(left) == len(right) and all(
        a.shape == b.shape and a.tobytes() == b.tobytes() for a, b in zip(left, right, strict=True)
    )


def _opted_out_designs_hold(monkeypatch: pytest.MonkeyPatch) -> bool:
    from ..studies import mtp_pricing_common as common
    from ..studies import refuter_calibration_band as band

    held = [
        _identical(
            _pricing_designs(learner, monkeypatch, True),
            _pricing_designs(learner, monkeypatch, False),
        )
        for learner in (common.QuadraticLogistic(), common.OracleRatio())
    ]
    held.append(_identical(_band_designs(monkeypatch, True), _band_designs(monkeypatch, False)))
    assert band.QuadraticLogistic.policy_steps == "omit"
    return all(held)


def test_w11c_an_opted_out_learner_receives_the_design_without_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _opted_out_designs_hold(monkeypatch)


def test_w11c_ignoring_the_opt_out_changes_the_design(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(density_ratio, "omits_policy_steps", lambda learner: False)
    assert not _opted_out_designs_hold(monkeypatch)


# ------------------------------------------------------------------ W11d


def _expected_up(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """``Shift(0.5, cap=5.5)``: its image indicator with the fold's bounds, then its held set."""
    from ..studies import longitudinal_mtp_common as common

    source = x - common.STEP
    image = (x <= common.CAP) & (source >= low) & (source <= high)
    return np.column_stack([image, x > common.CAP - common.STEP]).astype(float)


def _expected_price_up(x: np.ndarray, engaged: np.ndarray) -> np.ndarray:
    """``price_up`` at a node: the scaled piece below ``u(L) / 1.12`` on the declared support
    ``[4 + 2 L, 8 + 2 L]``, then the identity piece above the knee."""
    from ..studies import mtp_pricing_common as common

    low, high = common.lower(engaged), common.upper(engaged)
    knee = high / common.UP
    source = x / common.UP
    image = (source < knee) & (source >= low) & (source <= high)
    return np.column_stack([image, x >= knee]).astype(float)


def _node_columns_hold(monkeypatch: pytest.MonkeyPatch, law_name: str, seed: int) -> bool:
    from sklearn.linear_model import LinearRegression

    from cleverly.longitudinal import LTMLE

    from ..studies import canonical_mtp_pricing as pricing
    from ..studies import longitudinal_mtp_common as continuous
    from ..studies import mtp_pricing_common as common

    calls = _record_steps(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if law_name == "continuous":
            frame = continuous.sample_continuous(400, seed)
            LTMLE(
                {"natural": continuous.NATURAL, "up": continuous.UP},
                ratio="classifier",
                outcome_learner=LogisticRegression(),
                pseudo_learner=LinearRegression(),
                treatment_learner=Logit(),
                n_folds=2,
                random_state=0,
            ).fit(
                frame,
                outcome="Y",
                treatment=["A1", "A2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
                continuous_treatment=["A1", "A2"],
            )
        else:
            frame = common.sample(400, seed, 2)
            pricing.fit(
                frame, 2, (common.REFERENCE, "price_up"), treatment_learner=Logit(), n_folds=2
            )
    held = []
    for node in (1, 2):
        dose = frame[f"A{node}"].to_numpy()
        mine = [c for c in calls if np.array_equal(c[2], dose[c[1]]) and c[3].shape[1] >= 2]
        held.append(len(mine) == 2)
        for _, train, x, columns in mine:
            if law_name == "continuous":
                expected = _expected_up(x, dose[train].min(), dose[train].max())
            else:
                expected = _expected_price_up(x, frame[f"L{node}"].to_numpy()[train])
            held.append(np.array_equal(columns[:, :2], expected))
    return all(held)


@pytest.mark.parametrize("law_name", ["continuous", "pricing"])
def test_w11d_each_node_reads_its_lowered_maps(
    law_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for seed in range(5):
        assert _node_columns_hold(monkeypatch, law_name, seed), seed


def test_w11d_without_the_image_indicators_at_node_two_the_contract_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cleverly.longitudinal.sequential as sequential

    real = sequential._node_steps

    def mutated(plan: Any, time: int, *args: Any) -> Any:
        steps = real(plan, time, *args)
        if time != 2 or steps is None:
            return steps
        return lambda *call: steps(*call)[:, 1:]

    monkeypatch.setattr(sequential, "_node_steps", mutated)
    assert not _node_columns_hold(monkeypatch, "continuous", 0)


# ------------------------------------------------------------------ W11e


def _degenerate_case(name: str) -> tuple[CausalData, Any, int]:
    """The data, the policy and the index of its degenerate step column.

    ``covering``: a ``Piecewise`` whose pieces cover the line, so its ``uncovered`` identity
    column (the last) is zero.  ``atom``: an identity piece ``(-inf, 0]`` on a dose with an
    atom at 0 and no negative dose, so its column is the atom's indicator.
    """
    from cleverly.interventions import Piecewise

    frame = law.draw("G", 400, 6)
    if name == "covering":
        policy = Piecewise(
            pieces=((-np.inf, 3.0, Shift(1.0, cap=None)), (3.0, np.inf, Shift(0.0, cap=None)))
        )
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_kind="continuous"
        )
        return data, policy, 2
    frame["A"] = np.abs(frame["A"])
    frame.loc[frame.index[::5], "A"] = 0.0
    policy = ModifiedPolicy(
        "up off the atom",
        pieces=(
            Piece(-np.inf, 0.0, closed="right"),
            Piece(0.0, np.inf, _plus, _minus, _one, closed="right"),
        ),
        policy_kind="known",
    )
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_kind="continuous"
    ).with_dose_declarations(dose_atoms=(0.0,))
    return data, policy, 1


def _degenerate_fit(name: str, monkeypatch: pytest.MonkeyPatch, drop: bool) -> tuple[Any, Any]:
    """The design the classifier receives and the ratio, with the degenerate column left to
    the route (``drop=False``) or removed from the steps before the route sees it."""
    data, policy, degenerate = _degenerate_case(name)
    with monkeypatch.context() as patch:
        seen = _designs(patch)
        if drop:
            real = policy_module._policy_steps

            def build(*args: Any) -> Any:
                steps = real(*args)
                return lambda *call: np.delete(steps(*call), degenerate, axis=1)

            patch.setattr(policy_module, "_policy_steps", build)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluated = PolicySet.evaluate_by_classifier(
                (policy,), data, Logit(), make_folds(400, 2, random_state=0)
            )
    return seen, evaluated.ratio


@pytest.mark.parametrize("name", ["covering", "atom"])
def test_w11e_a_degenerate_column_is_dropped(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    designs, ratio = _degenerate_fit(name, monkeypatch, drop=False)
    expected, reference = _degenerate_fit(name, monkeypatch, drop=True)
    assert _identical(designs, expected)
    np.testing.assert_array_equal(ratio, reference)


@pytest.mark.parametrize("name", ["covering", "atom"])
def test_w11e_keeping_the_degenerate_column_changes_the_design(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected, _ = _degenerate_fit(name, monkeypatch, drop=True)
    monkeypatch.setattr(
        density_ratio, "_retained_steps", lambda columns, atoms: np.arange(columns.shape[1])
    )
    designs, _ = _degenerate_fit(name, monkeypatch, drop=False)
    assert not _identical(designs, expected)
