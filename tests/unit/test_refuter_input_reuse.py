"""Refuter input reuse (X106 part G5, comparison plan sections G5b, G5c, G5e, G5g).

Each change removes work no reported number reads, so each witness compares a refutation with an
in-process baseline that does the work, under the bitwise field walk of
``_refute_checkpoint_support.walk``, and asserts that the baseline did the work, so the equality
cannot pass vacuously.

- G5c the future-treatment control fits no band; G5g a point refit of a fit that built a band
  builds none; G5b and G5c name the control's learner and family.
- G5e a clustered point plasmode computes its cluster members once; a static regimen's density
  is resolved once on an all-reachable panel, with mutation controls for a dynamic regimen and
  for a resolution on the law's support.
"""

from __future__ import annotations

import functools
import sys
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, Regimen
from cleverly.validation import _longitudinal_plasmode as plasmode_module
from cleverly.validation import refute
from cleverly.validation._longitudinal_plasmode import LongitudinalPlasmodeLaw
from cleverly.validation._point_generated import PlasmodeLaw
from tests.unit import _refute_checkpoint_support as support

refute_module: Any = sys.modules["cleverly.validation.refute"]
tmle_module: Any = sys.modules["cleverly.estimators.tmle"]

R = 3


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


@contextmanager
def _counting_bands() -> Iterator[Counter[str]]:
    """Count the simultaneous bands the point fit path builds."""
    seen: Counter[str] = Counter()
    original = tmle_module.simultaneous_bands

    def counted(*args: Any, **kwargs: Any) -> Any:
        seen["bands"] += 1
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tmle_module, "simultaneous_bands", counted)
        yield seen


def _numbers(test: Any) -> Any:
    """Every number a test reports, without its detail text."""
    return support.walk(
        (test.values, test.standard_errors, test.statistic, test.statistic_pvalue, test.failures)
    )


# ----------------------------------------------------------------- longitudinal fixtures


def _frame(clustered: bool = False) -> pd.DataFrame:
    frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    if clustered:
        frame = frame.assign(g=np.arange(len(frame)) // 6)
    return frame


@functools.cache
def _longitudinal(clustered: bool = False, classifier: bool = False) -> Any:
    estimator = LTMLE(
        {"always": 1, "never": 0},
        outcome_learner=LogisticRegression() if classifier else GLM(family="binomial"),
        pseudo_learner=LinearRegression(),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=0,
    )
    return _quiet(
        lambda: estimator.fit(
            _frame(clustered),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            **({"id": "g"} if clustered else {}),
        )
    )


def _contrast(result: Any) -> str:
    return next(name for name in result if name.startswith("ate"))


def _control(kind: str, n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.integers(0, 2, n).astype(float) if kind == "binary" else rng.normal(size=n)


# ----------------------------------------------------------------- G5c future treatment


def _future(result: Any, control: np.ndarray) -> Any:
    return _quiet(
        lambda: refute(
            result,
            estimand=_contrast(result),
            tests=["future_treatment"],
            future_treatment_control=control,
            random_state=0,
        )["future_treatment"]
    )


def _banded_future_fit(*args: Any, **kwargs: Any) -> Any:
    """The control fit as it stood before G5c, with the default band."""
    fitter, *rest = _ORIGINAL_FUTURE_FIT(*args, **kwargs)
    fitter.simultaneous = True
    return (fitter, *rest)


_ORIGINAL_FUTURE_FIT = refute_module._future_treatment_fit


@pytest.mark.parametrize(
    ("clustered", "kind"),
    [(False, "binary"), (False, "continuous"), (True, "continuous")],
    ids=["binary", "continuous", "clustered"],
)
def test_g5c_the_control_fit_builds_no_band_and_reports_the_same_numbers(
    clustered: bool, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _longitudinal(clustered)
    control = _control(kind)
    with _counting_bands() as baseline_bands, monkeypatch.context() as patch:
        patch.setattr(refute_module, "_future_treatment_fit", _banded_future_fit)
        baseline = _future(result, control)
    assert baseline_bands["bands"] == 1
    with _counting_bands() as bands:
        candidate = _future(result, control)
    assert bands["bands"] == 0
    assert _numbers(candidate) == _numbers(baseline)


def test_g5c_the_detail_names_the_control_learner() -> None:
    binary = _future(_longitudinal(), _control("binary"))
    assert "the control fit models its binomial outcome with" in binary.detail
    continuous = _future(_longitudinal(classifier=True), _control("continuous"))
    assert "pseudo-outcome learner" in continuous.detail
    assert "because the fit's outcome learner is a classifier" in continuous.detail


# ----------------------------------------------------------------- G5b negative control


def _negative(result: Any, control: np.ndarray, **arguments: Any) -> Any:
    return _quiet(
        lambda: refute(
            result,
            tests=["negative_control_outcome"],
            negative_control_outcome=control,
            random_state=0,
            **arguments,
        )["negative_control_outcome"]
    )


def test_g5b_the_negative_control_detail_names_its_learner_and_family() -> None:
    from tests.unit.test_planned_refits import _point

    continuous = _negative(_point(), _control("continuous"))
    assert "the control fit models its gaussian outcome with" in continuous.detail
    binary = _negative(_point(), _control("binary"))
    assert "the control fit models its binomial outcome with" in binary.detail
    result = _longitudinal(classifier=True)
    swapped = _negative(result, _control("continuous"), estimand=_contrast(result))
    assert "pseudo-outcome learner" in swapped.detail


def test_g5b_a_structural_continuous_control_names_its_regression_learner() -> None:
    from cleverly.longitudinal.structural import StructuralNestedMean

    estimator = StructuralNestedMean(
        {"always": 1, "never": 0},
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        regression_learner=LinearRegression(),
        n_folds=2,
        random_state=0,
        simultaneous=False,
    )
    result = _quiet(
        lambda: estimator.fit(
            _frame(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
        )
    )
    test = _negative(result, _control("continuous"), estimand="blip[A1:1.0:intercept]")
    assert "models its gaussian outcome with linear regression" in test.detail


# ----------------------------------------------------------------- G5g point refits


@functools.cache
def _banded() -> Any:
    """A direct fit of three estimands, whose default band the fit builds."""
    from cleverly.estimators import TMLE
    from tests.unit.test_generated_point_families import _frame as point_frame

    frame = point_frame(300)
    estimator = TMLE(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        estimands=("ate", "rr", "or"),
        n_folds=2,
        random_state=1,
    )
    return _quiet(
        lambda: estimator.fit(frame, outcome="Yb", treatment="A", covariates=["W1", "W2"]).single()
    )


#: The plan lists ``dummy_outcome`` too; the generated tests refuse a direct fit, which carries
#: no identification metadata, and a study fit of one estimand builds no band. The bootstrap
#: test stands in for the drawn-data path (``_run_empirical_refits``), which the generated
#: tests share.
G5G_TESTS = [
    "placebo",
    "random_common_cause",
    "subset",
    "negative_control_outcome",
    "bootstrap_stability",
]


@pytest.mark.parametrize("test", G5G_TESTS)
def test_g5g_refits_build_no_band_and_report_the_same_numbers(
    test: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _banded()
    assert result.simultaneous is not None
    arguments: dict[str, Any] = {"tests": [test], "n_replicates": R, "random_state": 0}
    if test == "negative_control_outcome":
        arguments["negative_control_outcome"] = _control("binary")
    with _counting_bands() as baseline_bands, monkeypatch.context() as patch:
        # Mutation control and baseline: a refit that keeps the band on.
        patch.setattr(refute_module, "_bandless", lambda fit: fit.estimator)
        baseline = _quiet(lambda: refute(result, **arguments))
    assert baseline_bands["bands"] > 0
    with _counting_bands() as bands:
        candidate = _quiet(lambda: refute(result, **arguments))
    assert bands["bands"] == 0
    assert support.walk(candidate) == support.walk(baseline)


@pytest.mark.parametrize("test", ["subset", "bootstrap_stability"])
def test_g5g_a_held_rule_refits_with_the_bandless_estimator(
    test: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay of a learned rule wraps the estimator ``_bandless`` returns, on the subset
    path (the refit closure) and the drawn-data path (``_run_empirical_refits``)."""
    import copy

    from cleverly.interventions import learned
    from tests.unit import _learned_rule_support as rules
    from tests.unit.test_refute_learned_rule import _fit as learned_fit

    result = learned_fit()
    marked: list[Any] = []

    def bandless(fit: Any) -> Any:
        copied = copy.copy(fit.estimator)
        marked.append(copied)
        return copied

    seen: list[Any] = []
    replay = learned.frozen_rule_replay

    def recording(estimator: Any, rule: Any, rows: Any = None) -> Any:
        seen.append(estimator)
        return replay(estimator, rule, rows)

    monkeypatch.setattr(refute_module, "_bandless", bandless)
    monkeypatch.setattr(learned, "frozen_rule_replay", recording)
    _quiet(lambda: refute(result, estimand=rules.NAME, tests=[test], n_replicates=R))
    assert seen and all(any(item is mark for mark in marked) for item in seen)


def test_g5g_a_fit_that_built_no_band_refits_as_it_is() -> None:
    """The rule reads the fit's band, not the flag: a one-estimate fit keeps its estimator, so
    a refit that would band and refuse still does."""
    from tests.unit.test_planned_refits import _point

    result = _point()
    assert result.estimator.simultaneous and result.simultaneous is None
    assert refute_module._bandless(result) is result.estimator
    banded = _banded()
    copied = refute_module._bandless(banded)
    assert copied is not banded.estimator and copied.simultaneous is False
    assert banded.estimator.simultaneous is True


# ----------------------------------------------------------------- G5e point plasmode, clustered


def _old_rows(self: PlasmodeLaw, rng: np.random.Generator) -> tuple[np.ndarray, Any]:
    """``PlasmodeLaw._rows`` as it stood before G5e, with the members built on every draw."""
    n = self.covariates.shape[0]
    if self.clusters is None:
        return np.asarray(rng.integers(0, n, size=n)), None
    labels = np.unique(self.clusters)
    members = [np.flatnonzero(self.clusters == label) for label in labels]
    chosen = rng.integers(0, labels.size, size=labels.size)
    rows = np.concatenate([members[index] for index in chosen])
    ids = np.concatenate([np.full(members[index].size, copy) for copy, index in enumerate(chosen)])
    return rows, np.asarray(ids, dtype=np.int64)


@functools.cache
def _clustered_point() -> Any:
    from cleverly import ATE, CausalStudy, PointTreatment

    # Interleaved labels, so a grouping that assumed contiguous clusters would differ.
    frame = support.frame(300).assign(g=np.random.default_rng(3).permutation(np.arange(300) // 5))
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"), cluster="g"),
    )
    return _quiet(
        lambda: study.identify(ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    )


def test_g5e_a_clustered_plasmode_computes_its_members_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _clustered_point()
    arguments: dict[str, Any] = {"tests": ["plasmode"], "n_replicates": 10, "random_state": 0}
    with monkeypatch.context() as patch:
        patch.setattr(PlasmodeLaw, "_rows", _old_rows)
        baseline = _quiet(lambda: refute(result, **arguments))
    seen: Counter[str] = Counter()
    original = PlasmodeLaw.__post_init__

    def counted(self: PlasmodeLaw) -> None:
        if self.clusters is not None:
            seen["members"] += 1
        original(self)

    monkeypatch.setattr(PlasmodeLaw, "__post_init__", counted)
    candidate = _quiet(lambda: refute(result, **arguments))
    assert seen["members"] == 1
    assert support.walk(candidate) == support.walk(baseline)


# ----------------------------------------------------------------- G5e static regimens


def _old_density(law: Any, regimen: Any, node: int, rows: Any, blocks: Any, treatment: Any) -> Any:
    """``_policy_density`` as it stood before G5e: one resolution per simulated history."""
    from cleverly.longitudinal.regimen import resolve_plans

    time = node + 1
    (plan,) = resolve_plans((regimen,), plasmode_module._histories(law, rows, blocks, treatment))
    if plan.is_policy_node(time):
        return np.asarray(plan.policy_at(time)[:, 1], dtype=float)
    return np.asarray(plan.values[:, node] == 1.0, dtype=float)


def _support_density(
    law: Any, regimen: Any, node: int, rows: Any, blocks: Any, treatment: Any
) -> Any:
    """Mutation: a static regimen read from the law's support, where an unreachable node
    encodes 0."""
    from cleverly.longitudinal.regimen import resolve_plans

    if type(regimen) is not Regimen:
        return _old_density(law, regimen, node, rows, blocks, treatment)
    (plan,) = resolve_plans((regimen,), law.support)
    return np.asarray(plan.values[rows, node] == 1.0, dtype=float)


def _dynamic_shortcut(
    law: Any, regimen: Any, node: int, rows: Any, blocks: Any, treatment: Any
) -> Any:
    """Mutation: the static shortcut applied to every regimen, a rule's included."""
    from cleverly.longitudinal.regimen import resolve_plans

    index = np.arange(10)
    zeros = [np.zeros((index.size, block.shape[1])) for block in blocks]
    panel = plasmode_module._reachable(law, index, zeros, np.zeros((index.size, node)))
    (plan,) = resolve_plans((regimen,), panel)
    if plan.is_policy_node(node + 1):
        return _old_density(law, regimen, node, rows, blocks, treatment)
    return np.full(rows.size, float(plan.values[0, node] == 1.0))


@functools.cache
def _laws() -> dict[str, Any]:
    from tests.unit.test_longitudinal_plasmode import COIN, RULE, _binary_panel
    from tests.unit.test_longitudinal_plasmode import _ltmle as plasmode_ltmle
    from tests.unit.test_survival_plasmode import _fit as survival_fit

    censored = _quiet(
        lambda: plasmode_ltmle({"always": 1, "never": 0, "rule": RULE, "coin": COIN}).fit(
            _binary_panel(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    )
    return {
        "censored": LongitudinalPlasmodeLaw.fit(censored),
        "survival": LongitudinalPlasmodeLaw.fit(survival_fit(400)),
    }


def _regimens() -> dict[str, Any]:
    from tests.unit.test_longitudinal_plasmode import COIN, RULE

    return {
        "always": Regimen("always", (1, 1)),
        "never": Regimen("never", (0, 0)),
        "per-node": Regimen("r", (0, 1)),
        "rule": RULE,
        "coin": COIN,
    }


def _process(law: str) -> Any:
    from tests.unit.test_longitudinal_plasmode import PROCESS
    from tests.unit.test_survival_plasmode import HAZARD

    return HAZARD if law == "survival" else PROCESS


def _truths(law: str, regimen: str, density: Any = None) -> Any:
    fitted, process = _laws()[law], _process(law)
    plan = _regimens()[regimen]
    horizon = {"horizon": 2} if law == "survival" else {}
    with pytest.MonkeyPatch.context() as patch:
        if density is not None:
            patch.setattr(plasmode_module, "_policy_density", density)
        exact = fitted.truth(process, plan, draws=None, **horizon)
        sampled = fitted.truth(process, plan, draws=64, seed=4, **horizon)
    return support.walk((exact, sampled))


@pytest.mark.parametrize("regimen", ["always", "never", "per-node", "rule", "coin"])
@pytest.mark.parametrize("law", ["censored", "survival"])
def test_g5e_a_truth_equals_the_per_history_resolution(law: str, regimen: str) -> None:
    if law == "survival" and regimen == "coin":
        pytest.skip("the survival fit declares no stochastic regimen")
    assert _truths(law, regimen) == _truths(law, regimen, _old_density)


def test_g5e_a_static_truth_simulates_no_history(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Counter[str] = Counter()
    original = plasmode_module._histories

    def counted(*args: Any) -> Any:
        seen["histories"] += 1
        return original(*args)

    monkeypatch.setattr(plasmode_module, "_histories", counted)
    _truths("censored", "per-node")
    assert seen["histories"] == 0
    _truths("censored", "rule")
    assert seen["histories"] > 0


def test_g5e_the_mutations_change_a_truth() -> None:
    """The shortcut on a rule, and a static resolution on the support, each move a truth."""
    assert _truths("censored", "rule", _dynamic_shortcut) != _truths("censored", "rule")
    for law in ("censored", "survival"):
        assert _truths(law, "always", _support_density) != _truths(law, "always")


def test_g5e_a_longitudinal_plasmode_refute_equals_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_longitudinal_plasmode import PROCESS, _binary_panel
    from tests.unit.test_longitudinal_plasmode import _ltmle as plasmode_ltmle

    result = _quiet(
        lambda: plasmode_ltmle({"always": 1, "never": 0}).fit(
            _binary_panel(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    )
    arguments: dict[str, Any] = {
        "estimand": _contrast(result),
        "tests": ["plasmode"],
        "n_replicates": 10,
        "random_state": 1,
        "simulated_outcome": PROCESS,
    }
    with monkeypatch.context() as patch:
        patch.setattr(plasmode_module, "_policy_density", _old_density)
        baseline = _quiet(lambda: refute(result, **arguments))
    assert support.walk(_quiet(lambda: refute(result, **arguments))) == support.walk(baseline)
