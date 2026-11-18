"""The EP learner's size step reads every sieve member off one family design (X106 K5).

``SieveDesigns`` evaluates the sieve family's largest member once per set of rows and reads each
smaller member's columns off it, when the member's columns are computed as the family's are
(:func:`cleverly.estimators.ep.nested_columns`).  The debiasing inputs of a nuisance fit are read
once and shared by every size.  Both are equivalent implementations: every fit here is compared
byte for byte with the per-size path, which the ``uncached`` fixture restores by evaluating each
member itself and re-reading the inputs for each size.
"""

from __future__ import annotations

import gc
import hashlib
import warnings
import weakref
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import (
    Additive,
    BSpline,
    Constant,
    Cosine,
    Indicators,
    Interactions,
    Linear,
    Modifiers,
    Sieve,
    Tensor,
    UserBasis,
    resolve,
)
from cleverly.estimators import TMLE, ep
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import LearnedRule

DOMAINS = {"W1": (-1.0, 1.0), "W2": (-1.0, 1.0)}
MODIFIERS = Modifiers(["W1", "W2"], domains=DOMAINS)
WITH_LEVELS = Modifiers(["W1", "W2", "C"], domains=DOMAINS, levels={"C": (0.0, 1.0, 2.0, 3.0)})
#: The levels the fitted rows hold; a declared level no row holds makes a member collinear.
FIT_LEVELS = Modifiers(["W1", "W2", "C"], domains=DOMAINS, levels={"C": (0.0, 1.0, 2.0)})


def frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, size=(n, 2))
    w[:2] = [[-1.0, 1.0], [1.0, -1.0]]
    c = rng.integers(0, 3, n).astype(float)  # level 3 is declared and absent
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.4 * w[:, 0] - 0.3 * w[:, 1]))))
    eta = w[:, 0] + a * (0.5 + np.sin(2.0 * w[:, 0]) + w[:, 1] ** 2) - 0.5
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "C": c, "A": a, "Y": y})


def rows() -> dict[str, np.ndarray]:
    data = frame(37, seed=4)
    return {name: np.asarray(data[name]) for name in ("W1", "W2", "C")}


# ---------------------------------------------------------------- nesting, bit for bit

NESTED = [
    pytest.param(Cosine(["W1"], 5), MODIFIERS, id="cosine"),
    pytest.param(Cosine(["W1", "W2"], 4, intercept=False), MODIFIERS, id="cosine-no-intercept"),
    pytest.param(Linear(["W1", "W2"]), MODIFIERS, id="linear"),
    pytest.param(Indicators("C"), WITH_LEVELS, id="indicators"),
    pytest.param(
        Additive(Cosine(["W1"], 4), Indicators("C"), Linear(["W2"])), WITH_LEVELS, id="additive"
    ),
    pytest.param(
        Additive(Cosine(["W1"], 3, intercept=False), Cosine(["W2"], 5, intercept=False)),
        MODIFIERS,
        id="additive-no-intercepts",
    ),
    pytest.param(
        Additive(Cosine(["W1"], 3), Cosine(["W2"], 2), intercept=False),
        MODIFIERS,
        id="additive-no-shared-intercept",
    ),
    pytest.param(Tensor(Cosine(["W1"], 3), Cosine(["W2"], 2)), MODIFIERS, id="tensor"),
    pytest.param(
        Interactions(Additive(Cosine(["W1"], 3), Cosine(["W2"], 3)), 2),
        MODIFIERS,
        id="interactions",
    ),
    pytest.param(Sieve.default(WITH_LEVELS, max_size=4).basis, WITH_LEVELS, id="default-sieve"),
]


def assert_same_design(cached: np.ndarray, own: np.ndarray) -> None:
    assert cached.dtype == own.dtype == np.float64
    assert cached.shape == own.shape
    assert cached.strides == own.strides
    assert cached.flags.c_contiguous == own.flags.c_contiguous
    assert cached.flags.f_contiguous == own.flags.f_contiguous
    assert cached.tobytes() == own.tobytes()


@pytest.mark.parametrize(("family", "modifiers"), NESTED)
def test_every_member_reads_bit_for_bit_off_the_family(family: Any, modifiers: Modifiers) -> None:
    resolved = ep.sieve_member(family, int(family.max_size), modifiers)
    values = rows()
    designs = ep.SieveDesigns(resolved, modifiers, values)
    for size in range(int(family.max_size) + 1):
        member = ep.sieve_member(family, size, modifiers)
        own = np.asarray(member.evaluate(values, modifiers), dtype=float)
        positions = ep.nested_columns(resolved, member)
        if size == 0 and not getattr(resolved, "intercept", True):
            assert positions is None
        else:
            assert positions is not None, size
        assert_same_design(designs.at(member), own)


def test_a_bspline_is_never_nested() -> None:
    family = resolve(BSpline(["W1"], 4), MODIFIERS)
    for size in range(1, 4):
        assert ep.nested_columns(family, resolve(family.truncate(size), MODIFIERS)) is None
    assert ep.nested_columns(family, family) is not None


def test_a_bspline_treated_as_nested_reads_the_wrong_columns(monkeypatch: Any) -> None:
    """Mutation control: the truncated B-spline names its terms as the family does
    (``bs2(W1)`` and so on) with other knots, so name matching alone would read wrong columns."""
    real = ep._nests
    monkeypatch.setattr(ep, "_nests", lambda basis: isinstance(basis, BSpline) or real(basis))
    family = resolve(BSpline(["W1"], 4), MODIFIERS)
    member = resolve(family.truncate(2), MODIFIERS)
    values = rows()
    assert ep.nested_columns(family, member) is not None
    cached = ep.SieveDesigns(family, MODIFIERS, values).at(member)
    assert cached.tobytes() != np.asarray(member.evaluate(values, MODIFIERS)).tobytes()


def test_a_user_basis_is_nested_only_in_itself() -> None:
    family = UserBasis(
        lambda f: np.column_stack([np.ones(len(f["W1"])), f["W1"]]), ("1", "w"), ("W1",), "known"
    )
    assert ep.nested_columns(family, family) is not None
    other = UserBasis(lambda f: np.ones((len(f["W1"]), 1)), ("1",), ("W1",), "known")
    assert ep.nested_columns(family, other) is None
    assert ep.nested_columns(resolve(Cosine(["W1"], 3), MODIFIERS), Constant()) is not None


def test_fancy_indexing_would_change_the_layout(monkeypatch: Any) -> None:
    """Mutation control: ``a[:, positions]`` is Fortran-ordered, where ``evaluate`` gives C order;
    the final-stage least squares then rounds differently (seen on a five-modifier sieve)."""
    family = resolve(Additive(Cosine(["W1"], 4), Cosine(["W2"], 4)), MODIFIERS)
    member = resolve(family.truncate(2), MODIFIERS)
    values = rows()
    designs = ep.SieveDesigns(family, MODIFIERS, values)
    designs.at(family)
    assert designs._design is not None
    fancy = designs._design[:, ep.nested_columns(family, member)]
    own = np.asarray(member.evaluate(values, MODIFIERS))
    assert fancy.tobytes() == own.tobytes()
    assert fancy.flags.c_contiguous != own.flags.c_contiguous
    assert designs.at(member).flags.c_contiguous


def test_the_family_is_evaluated_once_per_set_of_rows(monkeypatch: Any) -> None:
    family = resolve(Additive(Cosine(["W1"], 4), Cosine(["W2"], 4)), MODIFIERS)
    calls: list[int] = []
    original = Additive.evaluate

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(len(self.terms))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Additive, "evaluate", counting)
    designs = ep.SieveDesigns(family, MODIFIERS, rows())
    for size in (1, 2, 3, 4, 2):
        designs.at(resolve(family.truncate(size), MODIFIERS))
    assert calls == [family.n_terms]


# ---------------------------------------------------------------- refusals keep today's text


def test_a_family_that_raises_falls_back_to_each_member() -> None:
    values = rows()
    values["W1"] = values["W1"].copy()
    values["W1"][3] = 1.5  # outside the declared domain
    family = resolve(Additive(Cosine(["W1"], 4), Cosine(["W2"], 4)), MODIFIERS)
    member = resolve(family.truncate(1), MODIFIERS)
    designs = ep.SieveDesigns(family, MODIFIERS, values)
    with pytest.raises((DataError, CapabilityError)) as cached:
        designs.at(member)
    with pytest.raises((DataError, CapabilityError)) as own:
        member.evaluate(values, MODIFIERS)
    assert type(cached.value) is type(own.value)
    assert str(cached.value) == str(own.value)
    assert designs._unusable


# ---------------------------------------------------------------- cached and per-size fits


@pytest.fixture
def uncached(monkeypatch: Any) -> Any:
    """Restore the per-size path: every member evaluated itself, the inputs re-read per size."""

    def enable() -> None:
        monkeypatch.setattr(
            ep.SieveDesigns,
            "at",
            lambda self, member: np.asarray(member.evaluate(self.frame, self.modifiers), float),
        )
        monkeypatch.setattr(
            ep.FitState, "pieces", property(lambda self: ep.read_only_pieces(self.reader()))
        )

    return enable


def ep_fit(
    data: pd.DataFrame, modifiers: Modifiers = MODIFIERS, scale: Any = "difference", **kw: Any
) -> Any:
    design = {key: kw.pop(key) for key in ("weights", "cluster") if key in kw}
    study = CausalStudy(
        data,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "C"), **design),
    )
    effect = study.identify(ConditionalContrast(modifiers, scale=scale))
    settings = {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": LogisticRegression(),
        "n_folds": 3,
        "random_state": 0,
        **kw,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate("ep", **settings)


def digest(fit: Any, data: pd.DataFrame) -> tuple[Any, ...]:
    record = fit.fit.sieve
    beta = None if fit.model.beta is None else np.asarray(fit.model.beta).tobytes()
    return (
        beta,
        np.asarray(fit.fit.debiasing.coefficients).tobytes(),
        tuple(record.cv_risks),
        record.chosen,
        record.inner_seeds,
        np.asarray(fit.predict(data.iloc[:50])).tobytes(),
    )


def weighted(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["w"] = np.random.default_rng(3).uniform(0.2, 3.0, len(data))
    return data


def clustered(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["cluster"] = np.random.default_rng(5).integers(0, 60, len(data))
    return data


CASES = [
    pytest.param({}, id="cv-difference"),
    pytest.param({"scale": "log_ratio"}, id="cv-log-ratio"),
    pytest.param({"modifiers": FIT_LEVELS}, id="cv-with-levels"),
    pytest.param(
        {"sieve": Sieve(Additive(Cosine(["W1"], 4), Cosine(["W2"], 4)), size=3)}, id="fixed"
    ),
    pytest.param(
        {"sieve": Sieve(Additive(Cosine(["W1"], 4), Cosine(["W2"], 4)), size=lambda n: 2)},
        id="schedule",
    ),
    pytest.param({"sieve": Sieve(Tensor(Cosine(["W1"], 3), Cosine(["W2"], 2)))}, id="tensor"),
    pytest.param({"sieve": Sieve(BSpline(["W1"], 3))}, id="bspline"),
    pytest.param({"final_learner": LinearRegression()}, id="final-learner"),
    pytest.param({"weights": "w", "_data": weighted}, id="weights"),
    pytest.param({"cluster": "cluster", "_data": clustered}, id="clusters"),
]


@pytest.mark.parametrize("case", CASES)
def test_a_cached_fit_equals_the_per_size_fit(case: dict[str, Any], uncached: Any) -> None:
    case = dict(case)
    data = case.pop("_data", lambda d: d)(frame())
    cached = digest(ep_fit(data, **dict(case)), data)
    uncached()
    assert digest(ep_fit(data, **dict(case)), data) == cached


def test_the_layout_mutation_moves_a_fit(monkeypatch: Any) -> None:
    """Mutation control for the fit witness: a Fortran-ordered design sends ``design @ beta``
    through another BLAS kernel, which sums in another order, and the CV risks move."""
    data = frame()
    kept = digest(ep_fit(data, FIT_LEVELS), data)
    original = ep.SieveDesigns.at

    def fortran(self: Any, member: Any) -> np.ndarray:
        return np.asfortranarray(original(self, member))

    monkeypatch.setattr(ep.SieveDesigns, "at", fortran)
    moved = digest(ep_fit(data, FIT_LEVELS), data)
    if moved == kept:
        pytest.skip("this BLAS sums both layouts alike; the layout unit test still gates")
    assert moved[2] != kept[2]


def rule_fit(**rule: Any) -> Any:
    """A learned-rule fit with ``loss="ep"``: its estimates and every fold's EP rule record."""
    from cleverly.interventions import _rule_learners as rule_learners

    records: list[Any] = []
    fold_curve = rule_learners.ep_fold_curve

    def recording(*args: Any, **kwargs: Any) -> Any:
        out = fold_curve(*args, **kwargs)
        record = out[1]
        records.append(
            (record.mode, record.sizes, np.asarray(record.cv_risks).tobytes(), record.chosen)
        )
        return out

    rule_learners.ep_fold_curve = recording
    engine = TMLE(
        learned_rule=LearnedRule(loss="ep", modifiers=MODIFIERS, **rule),
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        n_folds=3,
        simultaneous=False,
        random_state=0,
        cross_fit=True,
        cv_evaluation=True,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.fit(
                frame(600), outcome="Y", treatment="A", covariates=["W1", "W2", "C"]
            )
    finally:
        rule_learners.ep_fold_curve = fold_curve
    estimate = result.single().estimates
    assert records
    return tuple(
        (name, np.float64(e.psi).tobytes(), np.asarray(e.influence_curve).tobytes())
        for name, e in estimate.items()
    ) + tuple(records)


@pytest.mark.parametrize(
    "rule",
    [{}, {"sieve": Sieve.default(MODIFIERS, max_size=3, size=2)}],
    ids=["cv", "fixed"],
)
def test_the_ep_rule_learner_equals_its_per_size_fit(rule: dict[str, Any], uncached: Any) -> None:
    cached = rule_fit(**rule)
    uncached()
    assert rule_fit(**rule) == cached


# ---------------------------------------------------------------- the fold predictor and pieces


def test_each_held_out_mapping_gets_its_own_design(monkeypatch: Any) -> None:
    """``cross_validated_risks`` passes one mapping per fold; a predictor called with another
    mapping must evaluate that one (mutation control: keying by size alone fails)."""
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    original = ep.cross_validated_risks

    def probing(splits: Any, sizes: Any, fit_fold: Any, frame_: Any, *args: Any) -> Any:
        key, train, test = splits[0]
        predict = fit_fold(key, train)
        first = {name: np.asarray(column)[test] for name, column in frame_.items()}
        second = {name: np.asarray(column)[train[:40]] for name, column in frame_.items()}
        for held in (first, second, first):
            seen.append((predict(sizes[-1], held), held["W1"]))
        return original(splits, sizes, fit_fold, frame_, *args)

    monkeypatch.setattr(ep, "cross_validated_risks", probing)
    fit = ep_fit(frame())
    assert fit.fit.sieve.mode == "cv"
    assert len(seen[0][0]) != len(seen[1][0])
    assert seen[0][0].tobytes() == seen[2][0].tobytes()


def test_the_inputs_are_read_once_per_nuisance_fit_and_not_changed(monkeypatch: Any) -> None:
    """One read per nuisance fit (the outer fit and each fold's), and no size step writes into
    the arrays every later size reads."""
    read: list[tuple[Any, str]] = []
    original = ep._pieces

    def counting(fit: Any) -> Any:
        pieces = original(fit)
        read.append((pieces, _hash(pieces)))
        return pieces

    monkeypatch.setattr(ep, "_pieces", counting)
    fit = ep_fit(frame())
    assert len(read) == len(fit.fit.sieve.inner_seeds) + 1
    assert all(_hash(pieces) == before for pieces, before in read)


def _hash(pieces: Any) -> str:
    out = hashlib.sha256()
    for name in (
        "y",
        "treatment",
        "observed",
        "initial_observed",
        "initial_t",
        "initial_r",
        "weight",
        "rows",
    ):
        out.update(np.ascontiguousarray(getattr(pieces, name)).tobytes())
    return out.hexdigest()


def counted_evaluations(monkeypatch: Any, owner: Any, method: str) -> tuple[list[int], Any]:
    """Count ``Additive.evaluate`` calls made inside ``owner.method``, by term count."""
    calls: list[int] = []
    inside: list[int] = []
    original = Additive.evaluate

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(len(self.terms))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Additive, "evaluate", counting)
    wrapped = getattr(owner, method)

    def recording(*args: Any, **kwargs: Any) -> Any:
        start = len(calls)
        try:
            return wrapped(*args, **kwargs)
        finally:
            # One count per call of the rule learner's curve; one entry per evaluation in a fit.
            if method == "ep_fold_curve":
                inside.append(len(calls) - start)
            else:
                inside.extend(calls[start:])

    return inside, recording


def test_the_size_step_evaluates_the_basis_once_per_set_of_rows(monkeypatch: Any) -> None:
    """Work guard inside ``EPEstimator.fit``: with F folds and M sizes, ``2F + 2`` evaluations
    (``check`` and each fold's training and held-out rows read the family; the outer curve
    evaluates its own member), where the per-size path makes ``2FM + M + 1``."""
    inside, recording = counted_evaluations(monkeypatch, ep.EPEstimator, "fit")
    monkeypatch.setattr(ep.EPEstimator, "fit", recording)
    fit = ep_fit(frame())
    folds, sizes = len(fit.fit.sieve.inner_seeds), len(fit.fit.sieve.sizes)
    assert sizes == 6
    assert len(inside) == 2 * folds + 2
    family = fit.fit.estimator.member(sizes).n_terms
    assert inside[:-1] == [family] * (2 * folds + 1)
    assert inside[-1] == fit.fit.estimator.member(fit.fit.sieve.chosen).n_terms


def test_a_single_size_never_evaluates_the_family(monkeypatch: Any) -> None:
    """A fixed size reads one member in ``check`` and one for the curve, as before (design
    review finding 1: a shared family would only add work and memory there)."""
    inside, recording = counted_evaluations(monkeypatch, ep.EPEstimator, "fit")
    monkeypatch.setattr(ep.EPEstimator, "fit", recording)
    family = Additive(Cosine(["W1"], 6), Cosine(["W2"], 6))
    fit = ep_fit(frame(), sieve=Sieve(family, size=2))
    assert inside == [fit.fit.estimator.member(2).n_terms] * 2


def test_the_rule_learner_evaluates_the_basis_once_per_set_of_rows(monkeypatch: Any) -> None:
    from cleverly.interventions import _rule_learners as rule_learners

    folds: list[int] = []
    risks = ep.cross_validated_risks

    def recording_splits(splits: Any, *args: Any) -> Any:
        folds.append(len(splits))
        return risks(splits, *args)

    monkeypatch.setattr(ep, "cross_validated_risks", recording_splits)
    counts, recording = counted_evaluations(monkeypatch, rule_learners, "ep_fold_curve")
    monkeypatch.setattr(rule_learners, "ep_fold_curve", recording)
    rule_fit()
    assert counts and len(counts) == len(folds)
    assert counts == [2 * k + 1 for k in folds]


class SizedCosine(Cosine):
    """A cosine whose columns shrink with the size: a subclass need not nest."""

    def _block(self, values: Any, modifiers: Any) -> np.ndarray:
        return super()._block(values, modifiers) / self.size

    def truncate(self, k: int) -> SizedCosine:
        return SizedCosine(self.columns, k, self.domains, self.intercept)


def test_a_subclass_of_a_nested_basis_is_not_nested(monkeypatch: Any) -> None:
    family = resolve(SizedCosine(["W1"], 4), MODIFIERS)
    member = resolve(family.truncate(2), MODIFIERS)
    values = rows()
    assert ep.nested_columns(family, member) is None
    own = np.asarray(member.evaluate(values, MODIFIERS))
    assert ep.SieveDesigns(family, MODIFIERS, values).at(member).tobytes() == own.tobytes()
    # Mutation control: an isinstance test would read the family's columns, which differ.
    monkeypatch.setattr(ep, "_nests", lambda basis: isinstance(basis, Cosine))
    wrong = ep.SieveDesigns(family, MODIFIERS, values).at(member)
    assert wrong.tobytes() != own.tobytes()


@pytest.mark.parametrize(
    "error",
    [DataError, lambda text: CapabilityError(text, reason=RefusalReason.INVALID_DECLARATION)],
    ids=["data-error", "capability-error"],
)
def test_a_family_that_raises_leaves_each_member_to_itself(monkeypatch: Any, error: Any) -> None:
    """If the family alone raises, a member still gets its own design (mutation control for
    the fallback: without it the family's error escapes)."""
    family = resolve(Additive(Cosine(["W1"], 4), Cosine(["W2"], 4)), MODIFIERS)
    member = resolve(family.truncate(2), MODIFIERS)
    values = rows()
    own = np.asarray(member.evaluate(values, MODIFIERS))
    original = Additive.evaluate

    def failing(self: Any, *args: Any, **kwargs: Any) -> Any:
        if len(self.terms) == family.n_terms:
            raise error("the family alone fails")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Additive, "evaluate", failing)
    designs = ep.SieveDesigns(family, MODIFIERS, values)
    assert designs.at(member).tobytes() == own.tobytes()
    assert designs._unusable
    smaller = resolve(family.truncate(1), MODIFIERS)
    expected = np.asarray(smaller.evaluate(values, MODIFIERS)).tobytes()
    assert designs.at(smaller).tobytes() == expected


def test_the_shared_family_design_is_read_only() -> None:
    family = resolve(Additive(Cosine(["W1"], 3), Cosine(["W2"], 3)), MODIFIERS)
    design = ep.SieveDesigns(family, MODIFIERS, rows()).at(family)
    assert not design.flags.writeable
    with pytest.raises(ValueError):
        design[0, 0] = 0.0


def test_no_fold_state_outlives_its_fold(monkeypatch: Any) -> None:
    """Each fold's cached designs are released before the next fold's nuisance fit, and
    nothing is kept after the fit (design review finding 3; mutation control: dropping the
    release in ``cross_validated_risks`` keeps the previous fold's designs alive)."""
    alive: list[weakref.ref[Any]] = []
    init = ep.SieveDesigns.__init__

    def tracking(self: Any, *args: Any) -> None:
        init(self, *args)
        alive.append(weakref.ref(self))

    seen: list[int] = []
    nuisances = ep.EPEstimator._nuisances

    def probing(self: Any, data: Any, **changes: Any) -> Any:
        # Reference counting alone must free a fold's state: no collection before counting.
        seen.append(sum(ref() is not None for ref in alive))
        return nuisances(self, data, **changes)

    monkeypatch.setattr(ep.SieveDesigns, "__init__", tracking)
    monkeypatch.setattr(ep.EPEstimator, "_nuisances", probing)
    fit = ep_fit(frame())
    assert len(seen) == len(fit.fit.sieve.inner_seeds) + 1
    assert seen == [0] * len(seen)
    gc.collect()
    assert alive and all(ref() is None for ref in alive)


def test_a_size_step_that_writes_into_a_shared_input_raises(monkeypatch: Any) -> None:
    """Mutation control for the shared inputs: a debiasing step that wrote into the weights
    or the design in place would change every later size; the read-only views refuse it."""
    original = ep.debias

    def writing_weight(**kwargs: Any) -> Any:
        kwargs["weight"][:1] = 1.0
        return original(**kwargs)

    monkeypatch.setattr(ep, "debias", writing_weight)
    with pytest.raises(ValueError, match="read-only"):
        ep_fit(frame())

    def writing_design(**kwargs: Any) -> Any:
        kwargs["design"][:1] = 1.0
        return original(**kwargs)

    monkeypatch.setattr(ep, "debias", writing_design)
    with pytest.raises(ValueError, match="read-only"):
        ep_fit(frame(), sieve=Sieve(Additive(Cosine(["W1"], 2), Cosine(["W2"], 2))))


def test_the_rule_learner_releases_each_fold(monkeypatch: Any) -> None:
    """The rule learner's size step shares ``cross_validated_risks`` and so its release: no
    shared design of an earlier inner fold is alive when the next inner fold's nuisances fit."""
    from cleverly.interventions import _rule_learners as rule_learners

    alive: list[weakref.ref[Any]] = []
    init = ep.SieveDesigns.__init__

    def tracking(self: Any, *args: Any) -> None:
        init(self, *args)
        if self.family is not None:
            alive.append(weakref.ref(self))

    seen: list[int] = []
    step = rule_learners._size_step_nuisance

    def probing(*args: Any, **kwargs: Any) -> Any:
        seen.append(sum(ref() is not None for ref in alive))
        return step(*args, **kwargs)

    monkeypatch.setattr(ep.SieveDesigns, "__init__", tracking)
    monkeypatch.setattr(rule_learners, "_size_step_nuisance", probing)
    rule_fit()
    assert len(seen) > 3 and alive
    assert seen == [0] * len(seen)


def test_a_declared_learner_reads_no_held_out_design(monkeypatch: Any) -> None:
    """With a declared final learner the curve reads no basis on held-out rows: ``F + 2``
    evaluations inside ``EPEstimator.fit`` (mutation control: always reading the held-out
    design gives ``2F + 2``) and ``F + 1`` per rule-learner curve."""
    inside, recording = counted_evaluations(monkeypatch, ep.EPEstimator, "fit")
    monkeypatch.setattr(ep.EPEstimator, "fit", recording)
    fit = ep_fit(frame(), final_learner=LinearRegression())
    folds = len(fit.fit.sieve.inner_seeds)
    assert len(inside) == folds + 2


def test_a_declared_learner_in_the_rule_learner_reads_no_held_out_design(
    monkeypatch: Any,
) -> None:
    from cleverly.interventions import _rule_learners as rule_learners

    folds: list[int] = []
    risks = ep.cross_validated_risks

    def recording_splits(splits: Any, *args: Any) -> Any:
        folds.append(len(splits))
        return risks(splits, *args)

    monkeypatch.setattr(ep, "cross_validated_risks", recording_splits)
    counts, recording = counted_evaluations(monkeypatch, rule_learners, "ep_fold_curve")
    monkeypatch.setattr(rule_learners, "ep_fold_curve", recording)
    rule_fit(learner=LinearRegression())
    assert counts and counts == [k + 1 for k in folds]


def test_the_rule_learner_keys_each_held_out_mapping(monkeypatch: Any) -> None:
    """The rule learner's fold predictor, called with two held-out mappings, returns each
    mapping's own prediction (mutation control: keying by size alone fails)."""
    seen: list[np.ndarray] = []
    original = ep.cross_validated_risks

    def probing(splits: Any, sizes: Any, fit_fold: Any, frame_: Any, *args: Any) -> Any:
        if not seen:
            key, train, test = splits[0]
            predict = fit_fold(key, train)
            first = {name: np.asarray(column)[test] for name, column in frame_.items()}
            second = {name: np.asarray(column)[train[:40]] for name, column in frame_.items()}
            for held in (first, second, first):
                seen.append(predict(sizes[-1], held))
        return original(splits, sizes, fit_fold, frame_, *args)

    monkeypatch.setattr(ep, "cross_validated_risks", probing)
    rule_fit()
    assert len(seen[0]) != len(seen[1]) == 40
    assert seen[0].tobytes() == seen[2].tobytes()
