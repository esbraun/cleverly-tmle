"""The refuter holds one subset or band-rank placebo draw at a time (X106 part G1).

``refute()`` used to draw every subset, and every band-rank placebo data set, before its first
refit and to hold them all until the test ended. It now refits each draw and drops it. Where a
check reads every draw before the first refit, it keeps the generator state before each draw
and regenerates the draw at its refit. The witnesses follow ``rows/X106/G1/comparison-plan.md``:

- W1 every refit sees the same data, bit for bit, as under the old code;
- W2 a later test that shares the generator draws the same data;
- W3 a draw that fails its check refuses with the old refusal;
- W4 memory does not grow with the number of draws;
- W5 a replayed generator makes the same draws.

The reference is a copy of the old draw code kept in this module and run in the same process,
so a NumPy or scikit-learn release that moves the streams or the floats moves both sides.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import struct
import tracemalloc
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.datasets import make_clustered, make_longitudinal
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, CleverlyError, DataError
from cleverly.interventions import Support
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, DynamicRegimen, StructuralNestedMean
from cleverly.survival import ContinuousSurvivalTMLE
from cleverly.utils.random import generator_state, replay
from cleverly.validation import refute
from cleverly.validation._band_placebo import PanelBands, PointBands
from cleverly.validation._structural_refute import structural_subset_refusal
from tests.conftest import linear_in_sample

from .. import dose_declarations as law
from .test_band_placebo import UNCAPPED, VECTOR_SUPPORTS, _low_doses, _vector_frame
from .test_mtp_declared_support import INSIDE, NATURAL

refute_module = importlib.import_module("cleverly.validation.refute")
survival_module = importlib.import_module("cleverly.validation._continuous_survival_refute")

POLICY = "ate_policy[inside vs natural course]"


# ----------------------------------------------------------------- digests and field walk


def _digest(value: Any, out: Any) -> None:
    """Feed every array, number and string reachable through dataclass fields to ``out``."""
    if isinstance(value, np.ndarray):
        out.update(f"{value.dtype}{value.shape}".encode())
        out.update(np.ascontiguousarray(value).tobytes())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        out.update(type(value).__name__.encode())
        for field in dataclasses.fields(value):
            out.update(field.name.encode())
            _digest(getattr(value, field.name), out)
    elif isinstance(value, (tuple, list)):
        out.update(b"[")
        for item in value:
            _digest(item, out)
        out.update(b"]")
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            out.update(repr(key).encode())
            _digest(value[key], out)
    elif isinstance(value, float):
        out.update(struct.pack("<d", value))
    elif isinstance(value, (int, str, bool, type(None))):
        out.update(repr(value).encode())


def digest(data: Any) -> str:
    out = hashlib.sha256()
    _digest(data, out)
    return out.hexdigest()


def _same_bits(first: Any, second: Any) -> bool:
    """Bitwise equality: floats through their bytes, so ``nan`` equals itself and ``-0.0``
    differs from ``0.0``; everything else exactly."""
    if isinstance(first, float) and isinstance(second, float):
        return struct.pack("<d", first) == struct.pack("<d", second)
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(map(_same_bits, first, second))
    if dataclasses.is_dataclass(first) and type(first) is type(second):
        return all(
            _same_bits(getattr(first, f.name), getattr(second, f.name))
            for f in dataclasses.fields(first)
        )
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and digest(first) == digest(second)
    return bool(first == second)


def _record(monkeypatch: pytest.MonkeyPatch, cls: type, *, stub: bool = False) -> list[str]:
    """Record a digest of the data each ``cls.refit`` call fits; ``stub`` skips the fit."""
    seen: list[str] = []
    original = cls.refit  # type: ignore[attr-defined]

    def recording(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(digest(data))
        if stub:
            return _Stub(0.1 + 1e-3 * len(seen))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(cls, "refit", recording)
    return seen


class _Stub:
    """A refit stand-in: every estimand reads one value."""

    def __init__(self, psi: float) -> None:
        self.estimate = SimpleNamespace(psi=psi, plugin_std_error=0.5)

    def __getitem__(self, name: str) -> Any:
        return self.estimate


# ----------------------------------------------------------------- the old draw code


def reference_subsets(
    data: Any, rng: np.random.Generator, count: int, fraction: float = 0.7
) -> list[np.ndarray]:
    """The old point, longitudinal and structural subset draws, all up front."""
    drawn = [refute_module._subset_draw(data, rng, fraction) for _ in range(count)]
    return [d[0] for d in drawn]


def reference_survival_subsets(n: int, rng: np.random.Generator, count: int) -> list[np.ndarray]:
    """The old continuous-survival subset draws."""
    size = min(n, max(20, int(0.7 * n)))
    return [np.sort(rng.choice(n, size=size, replace=False)) for _ in range(count)]


def reference_band_sets(data: Any, bands: Any, rng: np.random.Generator, count: int) -> list[Any]:
    """The old band-rank placebo data sets, all up front."""
    sets = []
    for _ in range(count):
        if isinstance(bands, PanelBands):
            doses, drawn = bands.placebo(data, rng)
            sets.append(
                data.with_treatment(
                    doses,
                    known_mechanisms=refute_module.placebo_known_mechanisms(
                        data, bands.kernel, drawn
                    ),
                )
            )
        else:
            sets.append(data.with_treatment(bands.draw(data, rng, None)))
    return sets


def reference_noise(data: Any, rng: np.random.Generator, count: int) -> list[str]:
    return [digest(refute_module._noise_data(data, rng, k, False)) for k in range(count)]


def reference_placebo(data: Any, rng: np.random.Generator, count: int) -> list[str]:
    return [digest(refute_module._placebo_data(data, rng, None, None)) for _ in range(count)]


# ----------------------------------------------------------------- fixtures


def _quiet(call: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _point(n: int = 300) -> Any:
    rng = np.random.default_rng(0)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * w + 0.5 * a))))
    frame = pd.DataFrame({"W": w, "A": a, "Y": y.astype(float)})
    estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2))
    return _quiet(
        lambda: estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()
    )


def _clustered() -> Any:
    frame, _ = make_clustered(n=300, cluster_size=5, seed=7)
    estimator = TMLE(**linear_in_sample())
    return _quiet(lambda: estimator.fit(frame, outcome="Y", treatment="A", id="cluster").single())


def _structural(n: int = 300) -> Any:
    frame, _ = make_longitudinal(n=n, seed=0, censoring=False)
    estimator = StructuralNestedMean(
        {"always": 1, "never": 0},
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=0,
        simultaneous=False,
    )
    return _quiet(
        lambda: estimator.fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
        )
    )


def _survival(n: int = 300) -> Any:
    from tests.unit.test_continuous_survival_refute import _fit, _frame

    return _fit(_frame(n=n))


def _banded(n: int = 400) -> Any:
    estimator = TMLE(policies=[NATURAL, INSIDE], ratio="density", **law.point_learners())
    return _quiet(
        lambda: estimator.fit(
            law.point_frame(n=n),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            treatment_support=Support(0.0, law.upper),
        ).single()
    )


def _panel_banded() -> Any:
    regimens = {"natural": NATURAL, "up at 2": DynamicRegimen("up at 2", (NATURAL, INSIDE))}
    estimator = LTMLE(regimens, reference="natural", **law.panel_learners())
    return _quiet(
        lambda: estimator.fit(
            law.panel_frame(),
            continuous_treatment={"A1": Support(0.0, law.upper), "A2": Support(0.0, law.upper)},
            **law.PANEL,
        )
    )


SURVIVAL_RD = "ate_regimen[1 vs 0, 1 @ t=2]"
BLIP = "blip[A1:1.0:intercept]"


# ----------------------------------------------------------------- W1, W2 refit inputs


def _expect(cases: list[tuple[str, int]], data: Any, seed: int, extra: Any = None) -> list[str]:
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for name, count in cases:
        if name == "subset":
            out += [digest(data.subset(index)) for index in reference_subsets(data, rng, count)]
        elif name == "placebo":
            out += reference_placebo(data, rng, count)
        elif name == "random_common_cause":
            out += reference_noise(data, rng, count)
    return out


@pytest.mark.parametrize(
    ("fixture", "estimand", "tests"),
    [
        (_point, "ate", ("subset", "placebo")),
        (_point, "ate", ("subset", "random_common_cause")),
        (_clustered, "ate", ("subset", "random_common_cause")),
        (_structural, BLIP, ("subset", "random_common_cause")),
    ],
    ids=["point-placebo", "point-noise", "clustered", "structural"],
)
def test_w1_w2_each_subset_refit_and_the_later_test_see_the_old_data(
    monkeypatch: pytest.MonkeyPatch, fixture: Any, estimand: str, tests: tuple[str, ...]
) -> None:
    result = fixture()
    seen = _record(monkeypatch, type(result.estimator), stub=True)
    refute(result, estimand=estimand, tests=tests, n_replicates=4, random_state=5)
    assert seen == _expect([(name, 4) for name in tests], result.data, 5)


def test_w1_a_subset_value_is_the_refit_of_the_old_draw() -> None:
    """With the real estimator, every reported value is the refit of the old draw, bitwise."""
    result = _point()
    report = refute(result, estimand="ate", tests=("subset",), n_replicates=3, random_state=2)
    rng = np.random.default_rng(2)
    expected = tuple(
        float(
            result.estimator.refit(
                result.data.subset(index),
                **refute_module._intermediate_option(result),
                random_state=2,
                frozen_folds=refute_module._sliced_folds(result, index),
            )["ate"].psi
        )
        for index in reference_subsets(result.data, rng, 3)
    )
    assert _same_bits(report["subset"].values, expected)
    again = refute(result, estimand="ate", tests=("subset",), n_replicates=3, random_state=2)
    assert _same_bits(report, again)


def test_w1_w2_the_survival_subset_and_the_tests_before_it_see_the_old_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subsets are drawn before the placebo and the noise, and refitted after them."""
    result = _survival()
    seen = _record(monkeypatch, ContinuousSurvivalTMLE, stub=True)
    tests = ("placebo", "random_common_cause", "subset")
    refute(result, estimand=SURVIVAL_RD, tests=tests, n_replicates=3, random_state=4)
    data, n = result.data, result.data.n
    rng = np.random.default_rng(4)
    subsets = reference_survival_subsets(n, rng, 3)
    treatment = np.asarray(data.treatment)
    placebo = [
        digest(
            dataclasses.replace(
                data, treatment=survival_module._placebo_treatment(treatment, rng, "permute")
            )
        )
        for _ in range(3)
    ]
    noise = [
        digest(
            dataclasses.replace(
                data,
                baseline=np.column_stack([data.baseline, rng.normal(size=n)]),
                names=(*data.names, f"_noise_{k}"),
            )
        )
        for k in range(3)
    ]
    rows = [digest(survival_module._rows(data, index)) for index in subsets]
    assert seen == placebo + noise + rows


@pytest.mark.parametrize("later", ["random_common_cause", "subset"])
def test_w1_w2_each_band_placebo_refit_and_the_later_test_see_the_old_data(
    monkeypatch: pytest.MonkeyPatch, later: str
) -> None:
    result = _banded()
    seen = _record(monkeypatch, type(result.estimator), stub=True)
    refute(result, estimand=POLICY, tests=(later, "placebo"), n_replicates=3, random_state=1)
    data = result.data
    rng = np.random.default_rng(1)
    bands = PointBands.prepare(data)
    sets = [digest(item) for item in reference_band_sets(data, bands, rng, 3)]
    if later == "subset":
        tail = [digest(data.subset(index)) for index in reference_subsets(data, rng, 3)]
    else:
        tail = reference_noise(data, rng, 3)
    assert seen == tail + sets


def test_w1_a_band_placebo_value_is_the_refit_of_the_old_data_set() -> None:
    result = _banded()
    report = refute(result, estimand=POLICY, tests=("placebo",), n_replicates=3, random_state=1)
    bands = PointBands.prepare(result.data)
    expected = tuple(
        float(
            result.estimator.refit(
                item, **refute_module._intermediate_option(result), random_state=1
            )[POLICY].psi
        )
        for item in reference_band_sets(result.data, bands, np.random.default_rng(1), 3)
    )
    assert _same_bits(report["placebo"].values, expected)


def test_w1_w2_each_panel_band_placebo_refit_and_the_later_test_see_the_old_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _panel_banded()
    seen = _record(monkeypatch, type(result.estimator), stub=True)
    estimand = "ate_regimen[up at 2 vs natural]"
    refute(result, estimand=estimand, tests=("subset", "placebo"), n_replicates=3, random_state=2)
    data = result.data
    bands = PanelBands.prepare(data)
    assert bands is not None
    rng = np.random.default_rng(2)
    sets = [digest(item) for item in reference_band_sets(data, bands, rng, 3)]
    tail = [digest(data.subset(index)) for index in reference_subsets(data, rng, 3)]
    assert seen == tail + sets


def test_w1_a_vector_band_draw_replays_and_reads_each_support_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vector band evaluates every support function once to check, once to refit."""
    calls = {"count": 0}

    def low(h: Any) -> np.ndarray:
        calls["count"] += 1
        return VECTOR_SUPPORTS["D2"].lower(h)  # type: ignore[operator]

    from cleverly.data import CausalData

    supports = {**VECTOR_SUPPORTS, "D2": Support(low, VECTOR_SUPPORTS["D2"].upper)}
    data = CausalData.from_frame(
        _vector_frame(),
        outcome="Y",
        treatment=["D1", "D2"],
        covariates=["W"],
        continuous_treatment=["D1", "D2"],
    ).with_dose_declarations(supports)
    bands = PointBands.prepare(data)
    assert bands is not None and bands.vector
    old = np.random.default_rng(6)
    reference = [bands.draw(data, old, None) for _ in range(3)]
    # A vector data set without a fit has no arm check to pass; the draw itself is the subject,
    # so the replacement is the drawn dose matrix, and the fit's checks are no-ops here.
    monkeypatch.setattr(CausalData, "with_treatment", lambda self, doses, **_: doses)
    monkeypatch.setattr(refute_module, "_placebo_preflight", lambda result, item: None)
    monkeypatch.setattr(refute_module, "_band_extremes_check", lambda result, bands: None)
    calls["count"] = 0
    rng = np.random.default_rng(6)
    states = refute_module._band_placebo_states(SimpleNamespace(data=data), bands, rng, None, 3)
    checked = calls["count"]
    replayed = [refute_module._band_placebo_draw(data, bands, replay(s), None) for s in states]
    assert [digest(item) for item in replayed] == [digest(item) for item in reference]
    assert rng.bit_generator.state == old.bit_generator.state
    assert checked > 0 and calls["count"] == 2 * checked


def test_w2_mutation_a_replay_from_the_shared_generator_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass 2 drawing from the shared generator moves the refits and the later test."""
    result = _structural()
    seen = _record(monkeypatch, type(result.estimator), stub=True)
    shared: list[np.random.Generator] = []
    original = generator_state

    def capture(rng: np.random.Generator) -> dict[str, Any]:
        shared[:] = [rng]
        return original(rng)

    monkeypatch.setattr(refute_module, "generator_state", capture)
    monkeypatch.setattr(refute_module, "replay", lambda state: shared[0])
    tests = ("subset", "random_common_cause")
    refute(result, estimand=BLIP, tests=tests, n_replicates=4, random_state=5)
    assert seen != _expect([(name, 4) for name in tests], result.data, 5)


# ----------------------------------------------------------------- W3 refusals


def _first_failing(check: Any, draws: list[Any]) -> int | None:
    for k, draw in enumerate(draws, start=1):
        if check(k, draw):
            return k
    return None


def test_w3_a_structural_draw_that_fails_refuses_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _structural()
    fraction = 0.1
    for seed in range(200):
        draws = reference_subsets(result.data, np.random.default_rng(seed), 6, fraction)
        k = _first_failing(lambda _, d: structural_subset_refusal(result, [d]), draws)
        if k is not None and k > 1:
            break
    else:  # pragma: no cover - the fixture always has one
        pytest.fail("no seed whose first failing draw is not the first")
    expected = structural_subset_refusal(result, draws)
    assert expected is not None
    with pytest.raises(CapabilityError) as caught:
        refute(
            result,
            estimand=BLIP,
            tests=("subset",),
            n_replicates=6,
            subset_fraction=fraction,
            random_state=seed,
        )
    assert caught.value.refusal.reason is expected.reason
    assert caught.value.refusal.condition == expected.condition
    # Mutation: checking only the first draw lets the test run.
    checked: list[int] = []
    real = refute_module.structural_subset_refusal

    def first_only(result: Any, draws: Any) -> Any:
        checked.append(1)
        return real(result, draws) if len(checked) == 1 else None

    monkeypatch.setattr(refute_module, "structural_subset_refusal", first_only)
    _record(monkeypatch, type(result.estimator), stub=True)
    refute(
        result,
        estimand=BLIP,
        tests=("subset",),
        n_replicates=6,
        subset_fraction=fraction,
        random_state=seed,
    )


def test_w3_a_survival_draw_that_fails_refuses_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _survival()
    data, n = result.data, result.data.n
    guard = survival_module._subset_guard(result.estimator, data, 20)
    fraction = 0.1
    size = min(n, max(20, int(fraction * n)))
    found = None
    for seed in range(400):
        rng = np.random.default_rng(seed)
        draws = [np.sort(rng.choice(n, size=size, replace=False)) for _ in range(20)]
        k = _first_failing(guard, draws)
        if k is not None and k > 1:
            found = (seed, guard(k, draws[k - 1]))
            break
    assert found is not None
    seed, expected = found
    assert expected is not None
    with pytest.raises(CapabilityError) as caught:
        refute(
            result,
            estimand=SURVIVAL_RD,
            tests=("subset",),
            subset_fraction=fraction,
            random_state=seed,
        )
    assert str(caught.value) == str(expected)
    # Mutation: checking only the first draw misses the refusal.
    real = survival_module._subset_guard

    def first_only(estimator: Any, data: Any, count: int) -> Any:
        check = real(estimator, data, count)
        return lambda k, rows: check(k, rows) if k == 1 else None

    monkeypatch.setattr(survival_module, "_subset_guard", first_only)
    _record(monkeypatch, ContinuousSurvivalTMLE, stub=True)
    refute(
        result,
        estimand=SURVIVAL_RD,
        tests=("subset",),
        subset_fraction=fraction,
        random_state=seed,
    )


def test_w3_a_band_draw_that_fails_its_preflight_refuses_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = TMLE(policies=[NATURAL, UNCAPPED], ratio="density", **law.point_learners())
    result = _quiet(
        lambda: estimator.fit(
            _low_doses(n=12),  # small, so a draw passes the preflight often enough
            outcome="Y",
            treatment="A",
            covariates=["W"],
            treatment_support=Support(0.0, law.upper),
        ).single()
    )
    monkeypatch.setattr(refute_module, "_band_extremes_check", lambda result, bands: None)
    bands = PointBands.prepare(result.data)

    def fails(_: int, item: Any) -> bool:
        try:
            refute_module._placebo_preflight(result, item)
        except CleverlyError:
            return True
        return False

    found = None
    for seed in range(200):
        sets = reference_band_sets(result.data, bands, np.random.default_rng(seed), 6)
        k = _first_failing(fails, sets)
        if k is not None and k > 1:
            found = (seed, k)
            break
    assert found is not None
    seed, k = found
    item = reference_band_sets(result.data, bands, np.random.default_rng(seed), k)[-1]
    with pytest.raises(CleverlyError) as reference:
        try:
            refute_module._placebo_preflight(result, item)
        except CleverlyError as error:
            raise refute_module._on_draw(
                error, f"on placebo draw {k} of the band-rank placebo"
            ) from error
    estimand = "ate_policy[uncapped vs natural course]"
    with pytest.raises(DataError, match=f"on placebo draw {k} of the band-rank placebo") as caught:
        refute(result, estimand=estimand, tests=("placebo",), n_replicates=6, random_state=seed)
    assert caught.value.refusal.reason is reference.value.refusal.reason
    assert caught.value.refusal.condition == reference.value.refusal.condition
    # Mutation: checking only the first draw lets the failing draw reach its refit.
    real = refute_module._placebo_preflight
    checked: list[int] = []

    def first_only(result: Any, replacement: Any) -> None:
        checked.append(1)
        if len(checked) == 1:
            real(result, replacement)

    monkeypatch.setattr(refute_module, "_placebo_preflight", first_only)
    with pytest.raises(CleverlyError) as caught:
        refute(result, estimand=estimand, tests=("placebo",), n_replicates=6, random_state=seed)
    assert f"on placebo draw {k} of the band-rank placebo" not in str(caught.value)


# ----------------------------------------------------------------- W4 memory


def _peak(call: Any) -> int:
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


N_LARGE = 20_000


@pytest.mark.parametrize("fixture", ["point", "structural", "survival", "banded"])
def test_w4_memory_does_not_grow_with_the_draw_count(
    monkeypatch: pytest.MonkeyPatch, fixture: str
) -> None:
    if fixture == "point":
        result, estimand, test = _point(N_LARGE), "ate", "subset"
    elif fixture == "structural":
        result, estimand, test = _structural(N_LARGE), BLIP, "subset"
    elif fixture == "survival":
        # A Cox fit at 20,000 rows takes minutes; 2,000 keeps the draws far above the bound.
        result, estimand, test = _survival(2_000), SURVIVAL_RD, "subset"
    else:
        result, estimand, test = _banded(5_000), POLICY, "placebo"
    cls = ContinuousSurvivalTMLE if fixture == "survival" else type(result.estimator)
    _record(monkeypatch, cls, stub=True)

    def run(count: int) -> int:
        return _peak(
            lambda: refute(
                result, estimand=estimand, tests=(test,), n_replicates=count, random_state=0
            )
        )

    run(5)  # warm caches the first call fills
    n = result.data.n
    width = n if test == "placebo" else int(0.7 * n)
    # Two draws at a time at most, plus one recorded generator state (under 1 KiB) per draw.
    assert run(50) - run(5) <= 2 * width * 8 + 45 * 1024


def test_w4_mutation_keeping_each_regenerated_draw_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _structural(N_LARGE)
    _record(monkeypatch, type(result.estimator), stub=True)
    kept: list[Any] = []
    real = refute_module.replay

    def keeping(state: dict[str, Any]) -> np.random.Generator:
        rng = real(state)
        kept.append(refute_module._subset_draw(result.data, real(state), 0.7))
        return rng

    monkeypatch.setattr(refute_module, "replay", keeping)

    def run(count: int) -> int:
        kept.clear()
        return _peak(
            lambda: refute(
                result, estimand=BLIP, tests=("subset",), n_replicates=count, random_state=0
            )
        )

    run(5)
    assert run(50) - run(5) > 2 * int(0.7 * N_LARGE) * 8


# ----------------------------------------------------------------- W5 replay


@pytest.mark.parametrize(
    "draw",
    [
        lambda rng: rng.choice(1_000, size=700, replace=False),
        lambda rng: rng.choice(100_000, size=20, replace=False),
        lambda rng: rng.choice(np.arange(50) * 3, size=10, replace=True),
        lambda rng: rng.permutation(500),
        lambda rng: rng.integers(0, 7, size=300),
        lambda rng: rng.normal(size=300),
    ],
    ids=["choice-shuffle", "choice-set", "choice-replace", "permutation", "integers", "normal"],
)
@pytest.mark.parametrize("odd", [False, True], ids=["even", "has_uint32"])
def test_w5_a_replayed_generator_makes_the_same_draw(draw: Any, odd: bool) -> None:
    rng = np.random.default_rng(11)
    if odd:
        rng.integers(0, 10, dtype=np.uint32)
    state = generator_state(rng)
    assert state["has_uint32"] == int(odd)
    first = draw(rng)
    after = generator_state(rng)
    twin = replay(state)
    assert digest(draw(twin)) == digest(first)
    assert twin.bit_generator.state == after
    # The replay leaves the source untouched, and the recorded state is a copy.
    assert generator_state(rng) == after
    assert generator_state(replay(state)) == state


def test_w5_a_generator_other_than_pcg64_is_refused() -> None:
    with pytest.raises(TypeError, match="PCG64"):
        generator_state(np.random.Generator(np.random.MT19937(0)))
