"""The resolved ``ratio=`` route and the isolated-dose-cluster warning (W13, W14).

``ratio=None`` resolves once per fit: the classifier for a policy on a continuous or vector
dose, and the binned density otherwise, and in these cases even with such a dose: a fit that
sets ``density_bins``, a treatment learner that declares ``bin_design``, a declared treatment
mechanism, and an estimator that reads the density only.  The result records the route and the
rule that chose it, and a refit reads the recorded route.
"""

from __future__ import annotations

import copy
import importlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    AdaptiveIPTWMethod,
    CollaborativeTMLEMethod,
    DRTMLEMethod,
    IPSIMethod,
    ModelSpec,
    TMLEMethod,
)
from cleverly import _ratio_route as route_module
from cleverly.data import CausalData
from cleverly.estimators import CTMLE, DRTMLE, TMLE, AdaptiveIPTW
from cleverly.exceptions import DataWarning, MethodConfigurationError, PositivityWarning
from cleverly.interventions import PolicySet, Shift
from cleverly.learners import density as density_module
from cleverly.learners import isolated_dose_clusters
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
    SurvivalCTMLE,
    SurvivalDRTMLE,
)
from cleverly.validation import refute
from tests import mtp_density_tails_law as tails
from tests.studies import longitudinal_mtp_common as panel_law
from tests.unit import _known_mechanism_support as known
from tests.unit import _mtp_route_kinds as kinds

tmle_module = importlib.import_module("cleverly.estimators.tmle")
longitudinal_module = importlib.import_module("cleverly.longitudinal.estimator")

CLASSIFIER, DENSITY = "classifier", "density"
UP = Shift(0.5, cap=5.0)
STATIC = {"always": 1, "never": 0}


class IndexLogit(LogisticRegression):
    """A hazard learner that declares the index bin design."""

    bin_design = "index"


@pytest.fixture(scope="module")
def point_dose() -> CausalData:
    """The prepared continuous dose of :func:`tests.unit._mtp_route_kinds.fit_shift`."""
    return kinds.fit_shift(None).data


@pytest.fixture(scope="module")
def panel_dose() -> Any:
    """The prepared two-node continuous panel of :func:`~tests.unit._mtp_route_kinds.fit_panel`."""
    return kinds.fit_panel(None).data


def declared_mechanism() -> CausalData:
    frame = known.point_frame()
    return CausalData.from_frame(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2"],
        treatment_probabilities=known.mechanism(frame),
    )


def _panel(estimator: type, regimens: Any = None, **settings: Any) -> Callable[[], Any]:
    plans = {"natural": panel_law.NATURAL, "up": panel_law.UP} if regimens is None else regimens
    return lambda: estimator(plans, **settings)


#: One row per estimator class: its configuration with ``ratio=None``, the data it reads, and
#: the route and rule it must resolve to.
TABLE: dict[str, tuple[Callable[[], Any], str, str, str]] = {
    "TMLE policy": (lambda: TMLE(policies=[UP]), "point", CLASSIFIER, route_module.POLICY_DOSE),
    "TMLE density_bins": (
        lambda: TMLE(policies=[UP], density_bins=30),
        "point",
        DENSITY,
        route_module.DENSITY_BINS,
    ),
    "TMLE bin_design": (
        lambda: TMLE(policies=[UP], treatment_learner=IndexLogit()),
        "point",
        DENSITY,
        route_module.BIN_DESIGN,
    ),
    # The dose-response readers and the continuous MSM read a density, not a policy ratio.
    "TMLE no policy": (lambda: TMLE(), "point", DENSITY, route_module.NO_DOSE),
    "TMLE declared mechanism": (
        lambda: TMLE(policies=[Shift(0.0, cap=None)]),
        "declared",
        DENSITY,
        "default: a declared treatment mechanism fixes the ratio's denominator",
    ),
    "CTMLE": (lambda: CTMLE(policies=[UP]), "point", DENSITY, "default: this estimator"),
    "DRTMLE": (lambda: DRTMLE(), "point", DENSITY, "default: this estimator"),
    "AdaptiveIPTW": (lambda: AdaptiveIPTW(), "point", DENSITY, "default: this estimator"),
    "LTMLE": (_panel(LTMLE), "panel", CLASSIFIER, route_module.POLICY_DOSE),
    "LTMLE hazard": (
        _panel(LTMLE, survival_model="hazard"),
        "panel",
        CLASSIFIER,
        route_module.POLICY_DOSE,
    ),
    "LTMLE density_bins": (
        _panel(LTMLE, density_bins=30),
        "panel",
        DENSITY,
        route_module.DENSITY_BINS,
    ),
    "SDR": (_panel(SDR), "panel", CLASSIFIER, route_module.POLICY_DOSE),
    "SequentialGComputation": (
        _panel(SequentialGComputation),
        "panel",
        CLASSIFIER,
        route_module.POLICY_DOSE,
    ),
    "SequentialIPW": (_panel(SequentialIPW), "panel", DENSITY, "default: this estimator"),
    "ParametricGFormula": (
        _panel(ParametricGFormula),
        "panel",
        DENSITY,
        "default: this estimator",
    ),
    "IPSI": (lambda: IPSI({"double": 2.0}), "panel", DENSITY, "default: this estimator"),
    "SurvivalCTMLE": (_panel(SurvivalCTMLE, STATIC), "panel", DENSITY, "default: this estimator"),
    "SurvivalDRTMLE": (
        _panel(SurvivalDRTMLE, STATIC),
        "panel",
        DENSITY,
        "default: this estimator",
    ),
}


def table_mismatches(datasets: dict[str, Any]) -> list[str]:
    """Every row of :data:`TABLE` whose resolution is refused or differs from the row."""
    found = []
    for name, (build, data, route, reason) in TABLE.items():
        try:
            resolved = build().ratio_route(datasets[data])
        except Exception as error:
            found.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if resolved.route != route or not resolved.reason.startswith(reason):
            found.append(f"{name}: {resolved}")
    return found


@pytest.fixture(scope="module")
def datasets(point_dose: CausalData, panel_dose: Any) -> dict[str, Any]:
    return {"point": point_dose, "panel": panel_dose, "declared": declared_mechanism()}


def test_w13_each_estimator_class_resolves_its_route(datasets: dict[str, Any]) -> None:
    assert table_mismatches(datasets) == []


def test_w13_a_resolution_that_always_reads_the_classifier_fails(
    datasets: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: the rows that must read the density differ, or refuse."""

    def always(ratio: Any, **_: Any) -> Any:
        return route_module.RatioRoute(CLASSIFIER, route_module.POLICY_DOSE)

    for module in (tmle_module, longitudinal_module):
        monkeypatch.setattr(module, "resolve_ratio", always)
    found = " ".join(table_mismatches(datasets))
    for name in ("TMLE density_bins", "TMLE bin_design", "TMLE declared mechanism"):
        assert f"{name}:" in found
    for name in ("SurvivalCTMLE", "SurvivalDRTMLE", "SequentialIPW", "LTMLE density_bins"):
        assert f"{name}:" in found


def test_w13_the_result_records_the_route_and_the_estimator_keeps_none() -> None:
    estimator = TMLE(**kinds._learners(policies=[UP]))
    frame = kinds.shift_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(frame, outcome="Y", treatment="A").single()
    assert estimator.ratio is None
    assert result.estimator.ratio == CLASSIFIER
    assert result.config.ratio_route == route_module.RatioRoute(
        CLASSIFIER, route_module.POLICY_DOSE
    )
    assert "Policy ratio" in str(result.summary())
    panel = kinds.fit_panel(None)
    assert panel.config.ratio_route == route_module.RatioRoute(CLASSIFIER, route_module.POLICY_DOSE)


@pytest.mark.parametrize(
    "estimator", [SurvivalCTMLE, SurvivalDRTMLE], ids=["SurvivalCTMLE", "SurvivalDRTMLE"]
)
def test_w13_a_density_only_estimator_takes_none_and_refuses_the_classifier(
    estimator: type,
) -> None:
    estimator(STATIC)
    with pytest.raises(Exception, match="ratio='classifier'"):
        estimator(STATIC, ratio=CLASSIFIER)


@pytest.mark.parametrize(
    "method", [CollaborativeTMLEMethod, DRTMLEMethod, AdaptiveIPTWMethod, IPSIMethod]
)
def test_w13_the_study_route_refuses_only_an_explicit_classifier(method: type) -> None:
    """``ModelSpec(ratio=None)`` builds every method; a method that fits no policy ratio
    refuses only ``ratio="classifier"``."""
    assert method().models.ratio is None
    method().with_overrides(ratio=DENSITY)
    with pytest.raises(MethodConfigurationError):
        method().with_overrides(ratio=CLASSIFIER)


def test_w13_a_study_fit_records_the_resolved_route() -> None:
    from tests.unit._capability_sweep_support import fit_policy_means

    assert TMLEMethod().models.ratio is None and ModelSpec().ratio is None
    result = fit_policy_means()
    assert result.config.ratio_route.route == CLASSIFIER


def test_w13_the_classifier_refuses_a_density_setting() -> None:
    with pytest.raises(MethodConfigurationError, match="density_bins=7 would change nothing"):
        TMLE(policies=[UP], ratio=CLASSIFIER, density_bins=7)
    with pytest.raises(MethodConfigurationError, match="declares bin_design"):
        LTMLE(STATIC, ratio=CLASSIFIER, treatment_learner=IndexLogit())


def _route_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    for name, route in (("evaluate_by_classifier", CLASSIFIER), ("evaluate", DENSITY)):
        original = vars(PolicySet)[name].__func__

        def spy(cls: Any, *args: Any, _f: Any = original, _r: str = route, **kw: Any) -> Any:
            calls.append(_r)
            return _f(cls, *args, **kw)

        monkeypatch.setattr(PolicySet, name, classmethod(spy))
    return calls


@pytest.mark.parametrize("forced", [None, DENSITY], ids=["recorded", "mutation"])
def test_w13_a_band_rank_placebo_refits_on_the_recorded_route(
    forced: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``ratio=None`` fit on a declared band resolves to the classifier, and every
    refit of its band-rank placebo reads it.  The mutation forces the refit onto the
    density route, and the spy sees it."""
    result = kinds.fit_band(None)
    assert result.estimator.ratio == CLASSIFIER
    if forced is not None:
        original = TMLE.refit

        def other_route(self: TMLE, *args: Any, **kw: Any) -> Any:
            moved = copy.copy(self)
            moved.ratio = forced  # type: ignore[assignment]
            return original(moved, *args, **kw)

        monkeypatch.setattr(TMLE, "refit", other_route)
    calls = _route_calls(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refute(
            result,
            estimand="ate_policy[inside vs natural course]",
            tests=["placebo"],
            n_replicates=1,
            random_state=0,
        )
    assert calls
    assert set(calls) == ({CLASSIFIER} if forced is None else {DENSITY})


def test_the_classifier_route_ignores_density_bins_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The study helpers pass ``density_bins`` only on the density route; dropping it on the
    classifier route is result-neutral, which is checked with the refusal lifted."""
    for module in (route_module, tmle_module, longitudinal_module):
        monkeypatch.setattr(module, "refuse_classifier_settings", lambda *a: None)
    frame = kinds.shift_frame()
    psi = []
    for bins in (None, 30):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = TMLE(**kinds._learners(policies=[UP], ratio=CLASSIFIER, density_bins=bins))
            psi.append(fit.fit(frame, outcome="Y", treatment="A").single().estimates)
    assert {k: v.psi for k, v in psi[0].items()} == {k: v.psi for k, v in psi[1].items()}
    panel = []
    for bins in (None, 30):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(
                {"natural": panel_law.NATURAL, "up": panel_law.UP},
                reference="natural",
                outcome_learner=LogisticRegression(max_iter=1000),
                pseudo_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                n_folds=1,
                ratio=CLASSIFIER,
                density_bins=bins,
                simultaneous=False,
            ).fit(
                panel_law.sample_continuous(300, 0),
                outcome="Y",
                treatment=["A1", "A2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
                continuous_treatment=["A1", "A2"],
            )
        panel.append({k: v.psi for k, v in result.estimates.items()})
    assert panel[0] == panel[1]


# ------------------------------------------------------------------ W14: the warning


def _fires(law: str, n: int, seed: int, **settings: Any) -> bool:
    return bool(isolated_dose_clusters(tails.draw(law, n, seed)["A"].to_numpy(), **settings))


def test_w14_the_warning_fires_on_a_far_cluster_and_stays_silent_on_smooth_laws() -> None:
    assert all(_fires("C", 20_000, seed) for seed in range(5))
    for law in ("G", "T"):
        for n in (2_000, 20_000):
            assert not any(_fires(law, n, seed) for seed in range(5)), (law, n)
    assert sum(_fires("S", 2_000, seed) for seed in range(20)) <= 2


def test_w14_the_gap_rule_disabled_is_silent_on_the_far_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: a grid with no gap bins misses law C's far cluster."""
    monkeypatch.setattr(density_module, "_refine", lambda a, core: core)
    assert not any(_fires("C", 20_000, seed) for seed in range(5))


def test_w14_an_eager_rule_fails_the_silent_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eager control: an absolute gap test with no tail-spacing guard, at one row, fires on
    the smooth law G."""
    monkeypatch.setattr(density_module, "TAIL_ROWS", 1)
    assert any(_fires("G", n, seed, min_rows=1) for n in (2_000, 20_000) for seed in range(5))


def _far_frame(n: int = 600, seed: int = 0, far: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = w + rng.normal(size=n)
    a[:far] = 15.0 + 0.1 * rng.normal(size=far)
    return pd.DataFrame({"W": w, "A": a, "Y": a + w + rng.normal(size=n)})


@pytest.mark.parametrize("far", [0, 6])
def test_w14_a_classifier_fit_warns_of_a_far_cluster(far: int) -> None:
    frame = _far_frame(far=far)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = (
            TMLE(**kinds._learners(policies=[Shift(0.5, cap=None)], q_bounds=(-30.0, 40.0)))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
    messages = [str(w.message) for w in caught if issubclass(w.category, DataWarning)]
    clustered = [m for m in messages if "beyond an empty stretch" in m]
    assert bool(clustered) is (far > 0)
    if far:
        assert "6 observed doses" in clustered[0]
        assert any("beyond an empty stretch" in w.message for w in result.fit_warnings)


@pytest.mark.parametrize("ratio", [None, DENSITY])
def test_a_policy_that_moves_no_dose_warns_on_either_route(ratio: str | None) -> None:
    """The density route reads the no-move case off its flat bins; the classifier fits none."""
    frame = _far_frame(far=0)
    low = float(frame["A"].min()) - 1.0
    estimator = TMLE(
        **kinds._learners(
            policies=[Shift(1.0, cap=low, name="unreachable")], ratio=ratio, q_bounds=(-30.0, 40.0)
        )
    )
    with pytest.warns(UserWarning, match="changes no row.s dose"):
        result = estimator.fit(frame, outcome="Y", treatment="A").single()
    assert result.config.ratio_route.route == (ratio or CLASSIFIER)


@pytest.mark.parametrize("ratio", [None, DENSITY])
def test_a_dose_past_the_observed_range_warns_on_either_route(ratio: str | None) -> None:
    frame = _far_frame(far=0)
    estimator = TMLE(
        **kinds._learners(policies=[Shift(3.0, cap=None)], ratio=ratio, q_bounds=(-30.0, 40.0))
    )
    with pytest.warns(PositivityWarning, match="above the largest one observed"):
        estimator.fit(frame, outcome="Y", treatment="A")


def _censored_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Node 2 doses in ``[5, 10]`` for the rows at risk; 3% are censored after node 1."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = 7.5 + w + rng.normal(size=n)
    c1 = (rng.random(n) > 0.03).astype(float)
    a2 = rng.uniform(5.0, 10.0, n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.1 * a1 + 0.1 * a2 - 1.5))), n).astype(float)
    a2 = np.where(c1 == 1.0, a2, np.nan)
    y = np.where(c1 == 1.0, y, np.nan)
    return pd.DataFrame({"W": w, "A1": a1, "C1": c1, "A2": a2, "C2": c1, "Y": y})


@pytest.mark.parametrize("filler", [False, True], ids=["at_risk", "mutation"])
def test_w14_a_longitudinal_node_reads_only_its_doses_at_risk(
    filler: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rows censored before node 2 hold no dose there.  The mutation reads the filler 0
    the recursion gives them, which lies far below the doses at risk and warns."""
    if filler:

        def every_row(self: Any, data: Any) -> None:
            route_module.warn_isolated_clusters(
                {"A2": np.nan_to_num(np.asarray(data.treatment[:, 1], dtype=float), nan=0.0)}
            )

        monkeypatch.setattr(
            longitudinal_module._SequentialEstimator, "_warn_dose_clusters", every_row
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LTMLE(
            {"natural": Shift(0.0, cap=None), "up": Shift(0.5, cap=None)},
            reference="natural",
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            n_folds=1,
            simultaneous=False,
        ).fit(
            _censored_panel(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            censoring=["C1", "C2"],
            continuous_treatment=["A1", "A2"],
        )
    clustered = [str(w.message) for w in caught if "beyond an empty stretch" in str(w.message)]
    assert bool(clustered) is filler
