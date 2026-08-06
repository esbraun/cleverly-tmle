r"""Piece F2: a deterministic, component-level trace of the doubly-robust alternation.

``docs/roadmap.md``'s **piece F** is a recovery plan whose first premise is that the
``DRTMLE`` shortfall is *measured and not localized*, and whose order puts this second:
**F2 produces the common state-level instrument F3 and F4 both read.**  F3 is the bounded
differential run against the published ``drtmle`` R package and F4 is the construction
ablations; neither can say *where* two implementations of one algorithm first part company
without a record of what each one's state was at every step.  This module is that record on
the Python side.

**It closes nothing on its own, and that is what it is for.**  A divergence this makes
visible is a *question*, adjudicated against Theorem 1, ``docs/drtmle/theorem-concordance.md``
and the exact-law identities -- never settled by which side R is on.  Changing this package to
match R is [stop-ship 17](../docs/roadmap.md#stop-ship).

What it exports, per :func:`trace`:

* a **frozen fixture** -- observed ``W``/``A``/``Y``, the fold assignment, the weights, the
  truncation bounds, and the **initial** :math:`\hat{\bar Q}(1, \cdot)`,
  :math:`\hat{\bar Q}(0, \cdot)` and :math:`\hat g(1 \mid \cdot)` as columns;
* a **step**, in call order, for every solve the alternation runs: the pre- and post-update
  :math:`\bar Q^*`, :math:`g^*`, :math:`Q_r`, :math:`g_{r,1}` and :math:`g_{r,2}`, the
  fluctuation coefficients, the three empirical means, the Newton counts and the condition
  flags;
* the state **immediately before and after**
  :func:`~cleverly.estimators.targeting._close_at_frozen_reductions`, which is the one stage
  whose input and output differ by a whole re-solve and which no field on a returned fit
  distinguishes;
* the fit-level tail -- ``psi``, ``se``, the reported curve, and the per-arm correction
  arrays :math:`D^*_g` and :math:`D^*_Q`;
* and :func:`identities`, which **recomputes every recorded score from the recorded state**,
  longhand here rather than through the package's own
  :func:`~cleverly.fluctuation._score.score_columns` -- two calls of one function is one
  check, which is the whole lesson of ``docs/roadmap.md``'s item 20.

Three decisions about the fixture are load-bearing, and each is a way this could look right
and be useless.

**The outcome is binary, so the outcome scaler is the identity.**
``benchmarks/drtmle_injection.py`` records why this matters: recovering the affine map for a
continuous outcome carries an :math:`O(n^{-1/2})` error, and *"identical initial* :math:`\bar
Q`" between two implementations is exactly the thing that error would break.  A binary ``Y``
removes the map rather than estimating it, so the ``qn1``/``qn0`` columns are the arrays both
implementations start from with nothing in between.

**Both initial nuisances are misspecified, deliberately.**  At correct nuisances :math:`Q_r`
and :math:`g_{r,2}` vanish row by row, every correction array is zero, and the trace is blind
to a sign, to an update order and to a reduction vintage alike -- ``CLAUDE.md``'s rule about
where an exact-law instrument goes blind, in the place it bites hardest.  A fixture whose
nuisances were right would produce a trace in which every mutation F3 and F4 are hunting for
passes.  :func:`degeneracy` measures that rather than asserting it, and
``tests/unit/test_drtmle_trace.py`` fails if the misspecification is ever tidied away.

**The truncation is slack on every row of it.**  ``g`` is generated interior to
:data:`G_BOUNDS`, so no row clips.  That is not because clipping does not matter -- it is
``docs/roadmap.md``'s item 20 and the whole of piece B1b -- but because the two
implementations' truncation *conventions* differ, and a first-divergence hunt confounded by a
known convention difference locates the convention rather than the defect.
:attr:`Trace.clipped` reports the count so that a reader can see it is zero rather than assume
it, and a later fixture that turns clipping on is a second fixture rather than an edit to this
one.

**No estimator option is added and nothing under** ``src/`` **moves**, which is piece F's own
constraint -- only F7 may change ``src/``.  The two hooks are
:meth:`~cleverly.DRTMLE._reduction`, which
``benchmarks/drtmle_reference.py``'s ``ReferenceReductionDRTMLE`` already uses, and
:meth:`~cleverly.TMLE._solve_reduction`, which installs a **scoped** patch of the module-level
names :func:`~cleverly.estimators.targeting.solve_with_reduction` calls and restores them in a
``finally``.  That the instrument does not move what it measures is not asserted here: a
traced fit is compared against an untraced one, ``psi``, ``se``, ``epsilon`` and curve, in
``tests/unit/test_drtmle_trace.py``.

Running it::

    python -m benchmarks.drtmle_trace --order cleverly       # one trace, identities checked
    python -m benchmarks.drtmle_trace --both                 # both orders, compared
    python -m benchmarks.drtmle_trace --write-fixture        # regenerate the frozen fixture

At ``n = 200`` with ``glm`` reductions one trace is 3.2 s under ``"cleverly"`` and 6.5 s under
``"paper"`` on a four-core box, and **no primary nuisance is fitted at all** -- the initial
:math:`\bar Q` and :math:`g` are the closed forms below, injected.  The size is chosen for
that budget: this is a *localization* instrument, not a study, and the quantity it has to
resolve is a difference between two states rather than a coverage rate.  See :data:`N` for
what the neighbouring sizes do, which is not what a reader would guess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from cleverly import DRTMLE
from cleverly.estimators import targeting as _targeting

__all__ = [
    "COEFFICIENTS",
    "FIXTURES",
    "FIXTURE_VERSION",
    "G_BOUNDS",
    "N_FOLDS",
    "SEED",
    "Fixture",
    "FixtureSpec",
    "FrozenMechanism",
    "FrozenOutcome",
    "IdentityRow",
    "N",
    "State",
    "Step",
    "Trace",
    "TracingDRTMLE",
    "build_fixture",
    "compare",
    "degeneracy",
    "digest",
    "estimator",
    "fixture_path",
    "identities",
    "read_fixture",
    "spec",
    "trace",
    "write_fixture",
    "write_trace",
]

#: The fixture's schema version.  It is in the **filename** rather than in a field alone, so
#: that a second fixture -- one whose truncation binds, say, which this one deliberately does
#: not -- is a new file beside this one rather than an edit that silently reinterprets every
#: trace already compared against it.
FIXTURE_VERSION = "v1"

#: Rows.  Small on purpose: this instrument resolves a difference between two *states*, which
#: is exact arithmetic, rather than a rate, which is not -- so nothing here is bought by ``n``.
#: What ``n`` costs is the whole alternation, reductions refitted every round, twice over for
#: the two update orders.  Measured on four cores: ``n = 200`` is 3.2 s under ``"cleverly"``
#: and 6.5 s under ``"paper"``, and ``n = 300`` is 4.2 s and 10.6 s -- so this is the fast
#: tier's budget rather than a statistical choice.
#:
#: **The neighbouring sizes are not neighbouring experiments**, which is worth knowing before
#: anyone "rounds it up": at ``n = 250`` the same fixture recipe takes 43 rounds under
#: ``"paper"`` and 47 s, and exits on a *stall* rather than the tolerance under
#: ``"cleverly"``.  That is equation (10)'s conditioning, which this package's own docstrings
#: describe as near-singular on exactly the fits anybody wants -- and it is the reason a trace
#: is taken on **one frozen draw** rather than on a size sweep.
N = 200

#: The draw.  Fixed here and recorded in the manifest, because "the same fixture" has to mean
#: the same rows and not the same recipe.
SEED = 20_260_806

#: Outer folds.  Three rather than the estimator's ten: the primary nuisances are injected, so
#: folds reach only the reduced regressions' cross-fit, and three keeps that reproducible in a
#: second implementation without making a fold's training set uninformative.
N_FOLDS = 3

#: The mechanism truncation, **declared** rather than resolved from ``n``.  ``g_bounds="auto"``
#: reads :attr:`~cleverly.data.CausalData.effective_n`, which is a function of the draw -- and
#: a bound F3 has to reproduce in another language must be a number in a manifest.
G_BOUNDS = (0.01, 0.99)

#: The closed forms.  ``truth`` generates the data and ``initial`` is what the estimator is
#: handed, and the two differ in **both** nuisances: the mechanism drops ``w2`` and shrinks
#: ``w1``, and the outcome regression drops the interaction and ``w2`` and understates the
#: treatment effect.  Neither is a caricature -- both are the shape a working analyst's
#: misspecified ``glm`` has -- and the point of the difference is that :math:`Q_r` and
#: :math:`g_{r,2}` are then nonzero, which is the only regime in which this trace can see
#: anything at all.
COEFFICIENTS: dict[str, dict[str, float]] = {
    "truth_mechanism": {"intercept": 0.0, "w1": 0.4, "w2": -0.7},
    "truth_outcome": {"intercept": -0.3, "a": 0.8, "w1": 0.6, "w2": -0.5, "w1w2": 0.4},
    "initial_mechanism": {"intercept": 0.0, "w1": 0.15, "w2": 0.0},
    "initial_outcome": {"intercept": -0.1, "a": 0.5, "w1": 0.3, "w2": 0.0, "w1w2": 0.0},
}


@dataclass(frozen=True)
class FixtureSpec:
    """One frozen experiment: a draw recipe, the bound it is traced under, and its closed forms.

    A *table* rather than module constants, because there are now two fixtures and F2's own
    rule is that the second is a second **file** rather than an edit to the first: *"a fixture
    that turns clipping on is a second fixture, not an edit to this one"*.  Every trace already
    taken is against ``v1``, so ``v1``'s entry is exactly the constants it replaced and
    :data:`FIXTURE_VERSION` still defaults to it -- nothing this module's own CLI does moves.
    """

    version: str
    n: int
    seed: int
    n_folds: int
    g_bounds: tuple[float, float]
    coefficients: dict[str, dict[str, float]]
    #: What the fixture is *for*, carried into the manifest so a reader of the file knows
    #: which of the two they have without diffing the coefficients.
    purpose: str


#: The two fixtures.
#:
#: **``v1`` is interior and ``v2`` clips**, and that is the whole difference.  ``v1``'s initial
#: mechanism is ``0.15 * w1``, which puts every row within a hair of ``0.5`` and can never
#: reach ``G_BOUNDS`` -- deliberately, so that a first-divergence hunt is not confounded by the
#: two implementations' truncation *conventions*.  ``v2`` exists to ask that very question:
#: roadmap item 20 and the whole of piece B1b are about a bound binding, and an instrument that
#: can only run where it does not bind cannot see them.  So ``v2`` strengthens the mechanism
#: until the linear predictor spans the bound, and tightens the bound to meet it.
#:
#: Everything else about ``v2`` is ``v1``: the same draw seed, the same truth, the same
#: misspecified outcome regression, the same fold count.  One thing at a time is what makes the
#: difference between the two readable as truncation rather than as a second experiment.
FIXTURES: dict[str, FixtureSpec] = {
    # Built out of the constants above rather than restating them, so there is one source of
    # truth for the frozen experiment and no way for a table and a module constant to drift.
    "v1": FixtureSpec(
        version="v1",
        n=N,
        seed=SEED,
        n_folds=N_FOLDS,
        g_bounds=G_BOUNDS,
        coefficients=COEFFICIENTS,
        purpose="the truncation is slack on every row; a divergence here is not a bound",
    ),
    "v2": FixtureSpec(
        version="v2",
        n=N,
        seed=SEED,
        n_folds=N_FOLDS,
        # Tightened to meet the mechanism below.  `drtmle` takes a scalar lower bound and this
        # package takes a pair, so on a fixture where the bound binds the two conventions do
        # **not** coincide -- with two arms, a row clipped low on one arm is clipped high on the
        # other, and no choice of bound arranges that away.  That difference is what this
        # fixture measures rather than a confounder it failed to remove; the comparison gates it
        # as `truncation-convention`, before the update order, because here it can bite first.
        g_bounds=(0.15, 0.85),
        coefficients={
            **COEFFICIENTS,
            # `w1` at 1.6 rather than 0.15: `w1` is standard normal, so the linear predictor has
            # sd ~1.7 and reaches past `logit(0.15) = -1.73` on a material share of rows. The
            # outcome regression is `v1`'s untouched, so `Q_r` and `g_{r,2}` stay nonzero for
            # the reason F2 gives -- at correct nuisances the trace goes blind.
            "initial_mechanism": {"intercept": 0.0, "w1": 1.6, "w2": -0.6},
        },
        purpose="the truncation binds; this is the fixture roadmap item 20's question needs",
    ),
}


def spec(version: str | None = None) -> FixtureSpec:
    """The named fixture's specification, defaulting to :data:`FIXTURE_VERSION`."""
    name = FIXTURE_VERSION if version is None else version
    if name not in FIXTURES:
        raise ValueError(f"unknown fixture version {name!r}; choose from {sorted(FIXTURES)}")
    return FIXTURES[name]


#: The step vocabulary, and it is the thing F3 aligns R against.  ``"8"``, ``"9"`` and
#: ``"10"`` are the numbered equations; ``"refit"`` is a re-estimation of the reduced
#: regressions at the current pair, which is a *step* here because the vintage a reduction is
#: taken at is one of the five places the two implementations differ; ``"joint"`` is the
#: closing pass's four-column solve of equations (8) and (10) together, which the R package
#: has no analogue of.
EQUATIONS = ("8", "9", "10", "refit", "joint")

#: Which stage of the fit a step belongs to.  ``"prime"`` is the equation-(8) solve before the
#: first round, ``"round"`` is the alternation, and ``"close"`` is
#: :func:`~cleverly.estimators.targeting._close_at_frozen_reductions`.  The three are
#: recoverable from nowhere on a returned fit, which is why the boundary is recorded rather
#: than inferred from a round count.
PHASES = ("prime", "round", "close")


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _linear(spec: dict[str, float], a: Any, w1: Any, w2: Any) -> Any:
    """One closed form, written once so the fixture and the learners cannot disagree."""
    return (
        spec.get("intercept", 0.0)
        + spec.get("a", 0.0) * np.asarray(a, dtype=float)
        + spec.get("w1", 0.0) * np.asarray(w1, dtype=float)
        + spec.get("w2", 0.0) * np.asarray(w2, dtype=float)
        + spec.get("w1w2", 0.0) * np.asarray(w1, dtype=float) * np.asarray(w2, dtype=float)
    )


def initial_outcome(a: Any, w1: Any, w2: Any, version: str | None = None) -> Any:
    r"""The initial :math:`\hat{\bar Q}(a, W)` -- misspecified, and on the ``[0, 1]`` scale.

    On the outcome's own scale as well, since the outcome is binary and the scaler is the
    identity.  That coincidence is the point of a binary fixture and is not incidental.
    """
    return _expit(_linear(spec(version).coefficients["initial_outcome"], a, w1, w2))


def initial_mechanism(w1: Any, w2: Any, version: str | None = None) -> Any:
    r"""The initial :math:`\hat g(1 \mid W)` -- misspecified.

    Interior to the bound under ``v1`` and **not** under ``v2``, which is the one thing the
    two fixtures differ in.
    """
    return _expit(_linear(spec(version).coefficients["initial_mechanism"], 0.0, w1, w2))


class FrozenOutcome(BaseEstimator):
    """An outcome learner returning :func:`initial_outcome` at whatever rows it is asked for.

    A *function* rather than a table of the fixture's rows, and that is what F3 needs: the R
    side is handed the same closed form, so "identical initial predictions" is a property of
    the specification rather than of a join.  It follows
    ``tests.conftest.OracleOutcome``'s convention -- the design is ``[A, W...]``, arm first.
    """

    def __init__(self, version: str | None = None) -> None:
        # A plain attribute set in `__init__` and named as the parameter is, because sklearn
        # clones an estimator by reading its constructor signature: a version stored any other
        # way would be dropped by `clone` and the fold models would silently answer for `v1`.
        self.version = version

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> FrozenOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        return initial_outcome(design[:, 0], design[:, 1], design[:, 2], self.version)

    def predict(self, X: Any) -> Any:
        return self._mean(X)

    def predict_proba(self, X: Any) -> Any:
        p = self._mean(X)
        return np.column_stack([1.0 - p, p])


class FrozenMechanism(BaseEstimator):
    """A treatment learner returning :func:`initial_mechanism`; its design is ``W`` alone."""

    def __init__(self, version: str | None = None) -> None:
        self.version = version

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> FrozenMechanism:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        p = initial_mechanism(design[:, 0], design[:, 1], self.version)
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------------ the fixture


@dataclass(frozen=True)
class Fixture:
    """The frozen draw, and everything a second implementation needs to start where this did.

    Attributes
    ----------
    frame:
        ``w1``, ``w2``, ``a``, ``y``, ``fold``, ``weight``, ``qn1``, ``qn0``, ``gn``.  The last
        three are the **initial** predictions, evaluated at the fixture's own rows: they are
        what :class:`FrozenOutcome` and :class:`FrozenMechanism` return, and what F3 hands
        ``drtmle``'s ``Qn=``/``gn=``.
    folds:
        The realised outer-fold assignment.  Recorded rather than reconstructed, because a
        split is a function of a seed *and* of the splitter, and the second does not cross a
        language boundary.
    manifest:
        The bounds, the coefficients, the settings and the schema version.  The SHA-256 of the
        CSV bytes lives here, which is what makes "the same fixture" checkable rather than
        asserted.
    """

    frame: pd.DataFrame
    folds: np.ndarray
    manifest: dict[str, Any]

    @property
    def n(self) -> int:
        return len(self.frame)

    def arrays(self) -> dict[str, np.ndarray]:
        """The columns as float64 arrays, which is what every identity below reads."""
        return {name: self.frame[name].to_numpy(dtype=float) for name in self.frame.columns}


def fixture_path(suffix: str, root: Path | None = None, version: str | None = None) -> Path:
    """Where the frozen fixture lives, ``suffix`` being ``"csv"`` or ``"json"``."""
    base = Path(__file__).resolve().parent if root is None else Path(root)
    return base / "fixtures" / f"drtmle_trace_{spec(version).version}.{suffix}"


def build_fixture(*, version: str | None = None) -> pd.DataFrame:
    """The draw, from the seed alone -- no fit, no folds, no predictions yet.

    Separate from :func:`write_fixture` because the fold column can only come from a fit, and
    a function that both draws and fits cannot be used to check the fit's folds against the
    recorded ones.
    """
    settings = spec(version)
    n, coefficients = settings.n, settings.coefficients
    rng = np.random.default_rng(settings.seed)
    w1 = rng.normal(size=n)
    w2 = rng.uniform(-1.0, 1.0, size=n)
    g0 = _expit(_linear(coefficients["truth_mechanism"], 0.0, w1, w2))
    a = (rng.uniform(size=n) < g0).astype(float)
    q0 = _expit(_linear(coefficients["truth_outcome"], a, w1, w2))
    y = (rng.uniform(size=n) < q0).astype(float)
    return pd.DataFrame(
        {
            "w1": w1,
            "w2": w2,
            "a": a,
            "y": y,
            "weight": np.ones(n, dtype=float),
            "qn1": initial_outcome(1.0, w1, w2, settings.version),
            "qn0": initial_outcome(0.0, w1, w2, settings.version),
            "gn": initial_mechanism(w1, w2, settings.version),
        }
    )


def estimator(
    *, order: str = "cleverly", tracing: bool = True, version: str | None = None, **overrides: Any
) -> DRTMLE:
    """The one estimator configuration this fixture is traced under.

    Every keyword that could move a number is written out rather than left to a default, for
    the reason ``CLAUDE.md`` gives about spelled-out fold counts: a default that changes turns
    a frozen fixture into a different experiment with no diff to blame.
    """
    settings = spec(version)
    kwargs: dict[str, Any] = {
        "outcome_learner": FrozenOutcome(settings.version),
        "treatment_learner": FrozenMechanism(settings.version),
        "reduced_outcome_learner": "glm",
        "reduced_treatment_learner": "glm",
        "estimands": ["ey1", "ey0", "ate"],
        "n_folds": settings.n_folds,
        "learner_folds": settings.n_folds,
        "g_bounds": settings.g_bounds,
        "random_state": 0,
        "simultaneous": False,
        "update_order": order,
        **overrides,
    }
    return (TracingDRTMLE if tracing else DRTMLE)(**kwargs)


def _fit(frame: pd.DataFrame, est: DRTMLE) -> Any:
    return est.fit(frame, outcome="y", treatment="a", covariates=["w1", "w2"])[None]


def write_fixture(root: Path | None = None, version: str | None = None) -> Fixture:
    """Regenerate the frozen fixture and write both files.

    The fold column comes out of a real fit rather than out of a re-derivation of the
    splitter, which is what makes it authoritative:
    ``tests/unit/test_drtmle_trace.py`` then asserts a fresh fit reproduces it.
    """
    settings = spec(version)
    frame = build_fixture(version=settings.version)
    result = _fit(frame, estimator(tracing=False, version=settings.version))
    folds = np.asarray(result.repeats[0].nuisance.folds.assignment, dtype=int)
    frame = frame.copy()
    frame.insert(4, "fold", folds)
    # No `float_format`: pandas then writes numpy's shortest **round-trip** repr, and the
    # CSV is the object two implementations share -- a fixed-width format that lost a bit
    # would make "the same fixture" false in the sixteenth digit, which is where a
    # first-divergence hunt would find it and mis-classify it as a learner difference.
    # `%.17g` was tried and does not round-trip here: it came back short by a digit on 97 of
    # 200 rows, at 2.2e-16.
    csv = frame.to_csv(index=False)
    # The realised clipped count travels in the manifest, because it is the one property that
    # says *which* fixture a reader has and it cannot be read off the coefficients without
    # refitting. `v1` records 0 and `v2` records a material share; both are asserted.
    clipped = int(
        np.sum(
            (frame["gn"].to_numpy(dtype=float) < settings.g_bounds[0])
            | (frame["gn"].to_numpy(dtype=float) > settings.g_bounds[1])
        )
    )
    manifest = {
        "version": settings.version,
        "purpose": settings.purpose,
        "n": len(frame),
        "seed": settings.seed,
        "n_folds": settings.n_folds,
        "g_bounds": list(settings.g_bounds),
        "clipped": clipped,
        "coefficients": settings.coefficients,
        "columns": list(frame.columns),
        "outcome": "binary; the outcome scaler is the identity, so qn1/qn0 need no affine map",
        "weights": "unit; a non-unit weight is a second thing to reproduce and buys nothing here",
        "sha256": hashlib.sha256(csv.encode("utf-8")).hexdigest(),
    }
    path = fixture_path("csv", root, settings.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv, encoding="utf-8")
    fixture_path("json", root, settings.version).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return Fixture(frame=frame, folds=folds, manifest=manifest)


def read_fixture(root: Path | None = None, version: str | None = None) -> Fixture:
    """The committed fixture, with its digest checked against the manifest.

    Checked rather than trusted: the CSV is the object two implementations share, so a silent
    edit to it would make every comparison downstream a comparison of two different draws.
    """
    csv = fixture_path("csv", root, version).read_text(encoding="utf-8")
    manifest = json.loads(fixture_path("json", root, version).read_text(encoding="utf-8"))
    digest = hashlib.sha256(csv.encode("utf-8")).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(
            f"the trace fixture's bytes do not match its manifest: {digest} against "
            f"{manifest['sha256']}. Regenerate both with --write-fixture, or restore the CSV; "
            "do not update the digest to match an edited draw."
        )
    # `float_precision="round_trip"`, and it is not a nicety.  pandas' default C parser is
    # fast rather than exact: it read this file's `w1` back short by one unit in the last
    # place on 65 of 200 rows, at 2.2e-16.  That is precisely the size of difference a
    # first-divergence hunt would find between two implementations and mis-classify as a
    # learner difference -- the harness would have manufactured the divergence it was built
    # to locate.
    frame = pd.read_csv(fixture_path("csv", root, version), float_precision="round_trip")
    return Fixture(frame=frame, folds=frame["fold"].to_numpy(dtype=int), manifest=manifest)


# ------------------------------------------------------------------ the record


@dataclass(frozen=True)
class State:
    r"""Everything a score can be recomputed from, at one instant of the alternation.

    ``q`` is ``(n, K)`` in :attr:`Trace.arms` order and ``q_obs`` is at the observed arm; both
    on the ``[0, 1]`` scale, which for this fixture is the outcome's own.  ``g`` is
    :math:`g^*(1 \mid W)` **untruncated**, exactly as it travels through the alternation -- the
    bound is applied at read time by whatever divides by it, and recording the truncated array
    instead would make the one convention item 20 turned on invisible here.
    """

    q_obs: np.ndarray
    q: np.ndarray
    g: np.ndarray
    qr: np.ndarray
    gr1: np.ndarray
    gr2: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "q_obs": self.q_obs,
            "q": self.q,
            "g": self.g,
            "qr": self.qr,
            "gr1": self.gr1,
            "gr2": self.gr2,
        }


@dataclass(frozen=True)
class Step:
    """One solve or one refit, with the state either side of it.

    ``score`` is what the *package* recorded for the equation this step solved, in the arms'
    column order; :func:`identities` is what recomputes it from :attr:`after`.  The two being
    separate fields is the point -- a record that stored one number and called it both would
    be the duplication ``docs/roadmap.md``'s item 20 was hidden by.

    **A** ``"refit"`` **step's** :attr:`after` **is what the refit produced, which is not
    always what the round then used**, and that is deliberate rather than a looseness.  The
    paper order refits in two vintages -- :math:`g_{r,1}` and :math:`g_{r,2}` at the
    once-updated regression, :math:`Q_r` at the twice-updated one -- and adopts one field
    group from each; this package's order adopts all three.  So the state each equation
    *used* is read where its covariate is built rather than where a learner returned, and the
    gap between a refit's output and the next equation's input **is** the reduction-refit
    vintage that ``docs/roadmap.md``'s R3 table names as one of the five places two correct
    implementations of this algorithm differ.  :func:`vintages` reads it off.
    """

    index: int
    phase: str
    round: int
    equation: str
    before: State
    after: State
    epsilon: np.ndarray = field(default_factory=lambda: np.zeros(0))
    score: np.ndarray = field(default_factory=lambda: np.zeros(0))
    score_scale: np.ndarray = field(default_factory=lambda: np.zeros(0))
    names: tuple[str, ...] = ()
    n_iter: int = 0
    loglik: float = float("nan")
    hessian_condition: float = float("nan")
    failure: str | None = None
    converged: bool = True

    @property
    def label(self) -> str:
        return f"{self.phase}[{self.round}] eq({self.equation})"


@dataclass(frozen=True)
class Trace:
    """One fit's whole record: the fixture it ran on, the steps, and the reported tail."""

    order: str
    #: Which frozen fixture this ran on -- ``"v1"``, ``"v2"``.  **On the trace rather than only
    #: in a module global**, because :func:`write_trace` and :func:`digest` both used to read
    #: :data:`FIXTURE_VERSION` instead, which meant a trace taken on ``v2`` was written to
    #: ``drtmle_trace_v1_*`` and digested to a payload that never named its fixture.  The
    #: filename is the provenance a reader has; it has to come from the run.
    fixture_version: str
    arms: tuple[float, ...]
    bounds: tuple[float, float]
    weights: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    observed: np.ndarray
    steps: tuple[Step, ...]
    #: ``psi``, ``se`` and the reported curve, per estimand name.
    estimates: dict[str, dict[str, float]]
    curve: dict[str, np.ndarray]
    #: :math:`D^*_g` and :math:`D^*_Q` per arm, the two terms the reported curve subtracts.
    corrections: dict[str, np.ndarray]
    #: How the alternation ended -- ``exit_reason``, ``rounds``, ``closing``,
    #: ``closing_capped``, ``ill_conditioned``.  Facts about the loop rather than about the
    #: fit, and recoverable from nowhere else.
    exit: dict[str, Any]
    #: How many ``(row, arm)`` pairs the truncation binds on at the exit.  Zero on this
    #: fixture by construction; recorded so that is visible rather than assumed.
    clipped: int

    @property
    def rounds(self) -> int:
        return max((step.round for step in self.steps if step.phase == "round"), default=0)

    def of(self, equation: str) -> tuple[Step, ...]:
        return tuple(step for step in self.steps if step.equation == equation)

    def boundary(self) -> tuple[State, State]:
        """The state immediately before and after the closing pass.

        Read off the steps rather than stored twice: the last ``"round"`` step's ``after`` is
        what the loop left, and the last ``"close"`` step's ``after`` is what the record was
        built from.  Those being different states is the whole of why the closing pass is
        called an anaesthetic in ``docs/drtmle/investigation-log.md``, and it is the pair F4
        compares pre-close against post-close on.
        """
        rounds = [step for step in self.steps if step.phase == "round"]
        closes = [step for step in self.steps if step.phase == "close"]
        if not rounds or not closes:
            raise ValueError("a trace with no round or no closing pass has no boundary")
        return rounds[-1].after, closes[-1].after


# ------------------------------------------------------------------ the instrument


class _Recorder:
    """The mutable state the patched functions write through.

    Mutable and module-private, exactly as
    :class:`~cleverly.estimators.targeting._Companion` is and for the same reason: it is
    carried through the alternation rather than solved for, and it leaves this module only as
    a :class:`Trace`.
    """

    def __init__(self, arms: tuple[float, ...], bounds: tuple[float, float]) -> None:
        self.arms = arms
        self.bounds = bounds
        self.steps: list[Step] = []
        self.phase = "prime"
        # Always zero as recorded; :func:`_number_rounds` assigns the real one afterwards, so
        # that one rule numbers both update orders rather than a counter here needing a case
        # for each of them.
        self.round = 0
        self.q_obs = np.zeros(0)
        self.q = np.zeros((0, len(arms)))
        self.g = np.zeros(0)
        self.qr = np.zeros((0, len(arms)))
        self.gr1 = np.zeros((0, len(arms)))
        self.gr2 = np.zeros((0, len(arms)))

    def state(self) -> State:
        return State(
            q_obs=np.array(self.q_obs, dtype=float),
            q=np.array(self.q, dtype=float),
            g=np.array(self.g, dtype=float),
            qr=np.array(self.qr, dtype=float),
            gr1=np.array(self.gr1, dtype=float),
            gr2=np.array(self.gr2, dtype=float),
        )

    def take_outcome(self, fit: Any) -> None:
        self.q_obs = np.asarray(fit.observed, dtype=float).copy()
        self.q = np.column_stack([np.asarray(fit.arms[arm], dtype=float) for arm in self.arms])

    def take_mechanism(self, g: Any) -> None:
        self.g = np.asarray(g, dtype=float).reshape(-1).copy()

    def take_reduced(self, reduced: Any) -> None:
        self.qr = np.asarray(reduced.qr, dtype=float).copy()
        self.gr1 = np.asarray(reduced.gr1, dtype=float).copy()
        self.gr2 = np.asarray(reduced.gr2, dtype=float).copy()

    def record(self, equation: str, before: State, **fields: Any) -> None:
        self.steps.append(
            Step(
                index=len(self.steps),
                phase=self.phase,
                round=self.round,
                equation=equation,
                before=before,
                after=self.state(),
                **fields,
            )
        )


#: Equation (10)'s covariate columns are named by
#: :func:`~cleverly.fluctuation.reduced.reduced_outcome_submodel`; equation (8)'s by the
#: ``mean`` builder.  Classifying a solve by its submodel's names rather than by call order is
#: what keeps the two update orders readable through one recorder: under ``"paper"`` equation
#: (8) is solved *first*, so position says nothing.
_REDUCED_PREFIX = "h_dr"


def _equation_of(names: tuple[str, ...]) -> str:
    reduced = [name for name in names if name.startswith(_REDUCED_PREFIX)]
    if len(reduced) == len(names):
        return "10"
    return "joint" if reduced else "8"


class TracingDRTMLE(DRTMLE):
    """``DRTMLE`` that records every step of its own alternation.

    **Two hooks and no** ``src/`` **change**, which is piece F's constraint.  ``_reduction``
    wraps the refit closure -- the route
    ``benchmarks/drtmle_reference.py``'s ``ReferenceReductionDRTMLE`` already takes -- and
    ``_solve_reduction`` patches the module-level names
    :func:`~cleverly.estimators.targeting.solve_with_reduction` calls, for the duration of the
    one call, restoring them in a ``finally``.

    The patch is **scoped to the call** rather than installed at import, so an ordinary
    ``DRTMLE`` in the same process is untouched, and a failure inside the alternation cannot
    leave the module patched.  That the instrument does not move the fit is checked rather
    than argued: ``tests/unit/test_drtmle_trace.py`` compares a traced fit against an untraced
    one on ``psi``, ``se``, every ``epsilon`` and the whole curve.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder: _Recorder | None = None

    def _reduction(self, data: Any, nuisance: Any) -> Any:
        spec = super()._reduction(data, nuisance)
        if spec is None:
            return None
        inner = spec.refit

        def refit(current: Any) -> Any:
            recorder = self.recorder
            if recorder is None:
                return inner(current)
            before = recorder.state()
            produced, at_companion = inner(current)
            recorder.take_reduced(produced)
            recorder.record("refit", before)
            return produced, at_companion

        return replace(spec, refit=refit)

    def _solve_reduction(
        self, data: Any, nuisance: Any, group: Any, bounds: Any, lower: Any
    ) -> Any:
        recorder = _Recorder(tuple(nuisance.arms), (float(bounds[0]), float(bounds[1])))
        recorder.take_outcome(nuisance.outcome)
        recorder.take_mechanism(nuisance.propensity.arm(nuisance.arms[1]))
        if nuisance.reduced is not None:
            recorder.take_reduced(nuisance.reduced)
        self.recorder = recorder
        names = (
            "solve_submodel",
            "solve_bounded_mechanism",
            "_close_at_frozen_reductions",
            "reduced_outcome_submodel",
            "reduced_mechanism_covariate",
        )
        original = {name: getattr(_targeting, name) for name in names}

        # The two covariate builders are patched not to record steps of their own but to make
        # the recorder's reduced state *authoritative*: they are handed the set the equation
        # about to be solved actually reads, which under the paper order is a mixture of two
        # refits' output. Reading it off the refit closure instead would have every equation
        # scored against a set the estimator never used -- measured, before this was written:
        # the closing pass's equation-(10) identity missed by 1.3e-08 against a bar of 1e-12,
        # and only under `"paper"`.
        def reduced_outcome_submodel(treatment: Any, reduced: Any, **kw: Any) -> Any:
            recorder.take_reduced(reduced)
            return original["reduced_outcome_submodel"](treatment, reduced, **kw)

        def reduced_mechanism_covariate(reduced: Any, propensity: Any, **kw: Any) -> Any:
            recorder.take_reduced(reduced)
            return original["reduced_mechanism_covariate"](reduced, propensity, **kw)

        def solve_submodel(scaled: Any, initial: Any, submodel: Any, *args: Any, **kw: Any) -> Any:
            before = recorder.state()
            out = original["solve_submodel"](scaled, initial, submodel, *args, **kw)
            recorder.take_outcome(out.targeted)
            recorder.record(
                _equation_of(tuple(submodel.names)),
                before,
                epsilon=np.asarray(out.epsilon, dtype=float).copy(),
                score=np.asarray(out.score, dtype=float).copy(),
                score_scale=np.zeros(0)
                if out.score_scale is None
                else np.asarray(out.score_scale, dtype=float).copy(),
                names=tuple(submodel.names),
                n_iter=int(out.n_iter),
                loglik=float(out.loglik),
                hessian_condition=float(out.hessian_condition),
                failure=out.failure,
                converged=bool(out.converged),
            )
            return out

        def solve_bounded_mechanism(
            indicator: Any, propensity: Any, covariate: Any, weights: Any, **kw: Any
        ) -> Any:
            before = recorder.state()
            out = original["solve_bounded_mechanism"](
                indicator, propensity, covariate, weights, **kw
            )
            recorder.take_mechanism(out.propensity)
            recorder.record(
                "9",
                before,
                epsilon=np.asarray(out.epsilon, dtype=float).copy(),
                score=np.asarray(out.score, dtype=float).copy(),
                score_scale=np.asarray(out.score_scale, dtype=float).copy(),
                names=tuple(f"h_g{arm:g}" for arm in recorder.arms),
                n_iter=int(out.n_iter),
                loglik=float("nan") if out.loglik is None else float(out.loglik),
                hessian_condition=float("nan")
                if out.hessian_condition is None
                else float(out.hessian_condition),
                failure=out.failure,
                converged=bool(out.converged),
            )
            return out

        def close(*args: Any, **kw: Any) -> Any:
            # The boundary is read here, where the state already exists, rather than exposed
            # on a returned fit: piece F2's own constraint, and the reason is that a field on
            # `ReductionFluctuation` would be a public option in everything but name. Only the
            # *phase* is set; the round number is derived from the step stream afterwards, by
            # `_number_rounds`, so that one rule numbers both update orders.
            recorder.phase = "close"
            return original["_close_at_frozen_reductions"](*args, **kw)

        for name, patched in (
            ("solve_submodel", solve_submodel),
            ("solve_bounded_mechanism", solve_bounded_mechanism),
            ("_close_at_frozen_reductions", close),
            ("reduced_outcome_submodel", reduced_outcome_submodel),
            ("reduced_mechanism_covariate", reduced_mechanism_covariate),
        ):
            setattr(_targeting, name, patched)
        try:
            return super()._solve_reduction(data, nuisance, group, bounds, lower)
        finally:
            for name, value in original.items():
                setattr(_targeting, name, value)


def _number_rounds(steps: list[Step]) -> list[Step]:
    """Assign a round number to every recorded step, from the stream's own shape.

    Derived after the fact rather than hooked, because there is nowhere in the estimator to
    hook it without adding the option F2 refuses.  The rule is one sentence and covers both
    update orders **and** any guard, which is why it is not a table of cases: the first step
    is always the priming equation-(8) solve, the step after it is whatever the round *opens*
    with, and every later occurrence of that equation opens the next round.

    Under ``"cleverly"`` with both guards the opener is equation (9); under ``"paper"`` it is
    equation (8), which is exactly why the rule cannot be "a round starts at equation (9)".
    Under ``guard=("g",)`` it is equation (10).  The closing pass names itself -- the wrapper
    around :func:`~cleverly.estimators.targeting._close_at_frozen_reductions` sets the phase --
    so nothing here has to detect it.
    """
    numbered: list[Step] = []
    opener: str | None = None
    current = 0
    for step in steps:
        if step.phase == "close":
            numbered.append(replace(step, round=current + 1))
            continue
        if not numbered:
            numbered.append(replace(step, phase="prime", round=0))
            continue
        if opener is None:
            opener = step.equation
        if step.equation == opener:
            current += 1
        numbered.append(replace(step, phase="round", round=current))
    return numbered


def trace(
    fixture: Fixture | None = None,
    *,
    order: str = "cleverly",
    version: str | None = None,
    **overrides: Any,
) -> Trace:
    """Fit the fixture under ``order`` and return the whole record.

    The fixture's fold column is checked against the fit's rather than assumed: the folds are
    what the reduced regressions are cross-fitted over, so a split that had drifted would make
    every comparison against this trace a comparison against a different experiment.

    ``overrides`` reach :func:`estimator` and default to nothing, so every trace this module's
    own CLI takes is the frozen configuration exactly as it was.  They exist for
    :mod:`benchmarks.drtmle_r_compare`, which has to take a trace whose *reduced learner* is a
    bare unpenalised GLM rather than this module's two-candidate ``"glm"`` Super Learner --
    see that module for why removing a known learner difference is what lets a
    first-divergence hunt find a construction difference instead of finding the learner.
    """
    from cleverly.estimators.tmle import correction_parts
    from cleverly.validation.drtmle import correction_check

    fixture = read_fixture(version=version) if fixture is None else fixture
    version = str(fixture.manifest["version"])
    est = estimator(order=order, version=version, **overrides)
    result = _fit(fixture.frame, est)
    repeat = result.repeats[0]
    realised = np.asarray(repeat.nuisance.folds.assignment, dtype=int)
    if not np.array_equal(realised, fixture.folds):
        raise ValueError(
            "the fit's outer folds are not the fixture's. The reduced regressions are "
            "cross-fitted over them, so a trace taken here would not be a trace of the "
            "frozen experiment. Regenerate the fixture with --write-fixture and say in the "
            "commit what moved."
        )
    recorder = est.recorder
    assert recorder is not None
    fluctuation = repeat.fluctuations["mean"]
    reduction = fluctuation.reduction
    assert reduction is not None

    data = result.data
    scaled = repeat.nuisance.scaler.scale(data.outcome)
    # The one place the corrections are built -- the reported curve comes through it too, so
    # a trace cannot end up describing a different state from the fit it traced.
    parts = correction_parts(data, repeat.nuisance, fluctuation, fluctuation.targeted, scaled)
    assert parts is not None
    check = correction_check(result, tolerance=1.0)
    return Trace(
        order=order,
        fixture_version=version,
        arms=tuple(float(arm) for arm in reduction.reduced.arms),
        bounds=(float(reduction.bounds[0]), float(reduction.bounds[1])),
        weights=np.asarray(data.weights, dtype=float).reshape(-1),
        treatment=np.asarray(data.treatment, dtype=float).reshape(-1),
        outcome=np.asarray(scaled, dtype=float).reshape(-1),
        observed=np.asarray(data.observed, dtype=float).reshape(-1),
        steps=tuple(_number_rounds(recorder.steps)),
        estimates=_estimates(result),
        curve={
            str(name): np.asarray(values, dtype=float)
            for name, values in result.influence_curves.items()
        },
        corrections={
            **{
                f"D*_g[{arm:g}]": np.asarray(values, dtype=float)
                for arm, values in parts.d_g.items()
            },
            **{
                f"D*_Q[{arm:g}]": np.asarray(values, dtype=float)
                for arm, values in parts.d_q.items()
            },
        },
        exit={
            "exit_reason": reduction.exit_reason,
            "rounds": int(reduction.rounds),
            "closing": int(reduction.closing),
            "closing_capped": bool(reduction.closing_capped),
            "ill_conditioned": int(reduction.ill_conditioned),
            "failure": reduction.failure,
        },
        clipped=max((row.clipped for row in check.rows), default=0),
    )


def _estimates(result: Any) -> dict[str, dict[str, float]]:
    """``psi`` and ``se`` per estimand, off the reported table rather than off an internal."""
    frame = result.to_frame()
    return {
        str(row["estimand"]): {"psi": float(row["psi"]), "se": float(row["std_err"])}
        for _, row in frame.iterrows()
    }


# ------------------------------------------------------------------ the identities


@dataclass(frozen=True)
class IdentityRow:
    """One recomputation, and how far it missed.

    ``recorded`` is what the package stored and ``recomputed`` is this module's own longhand
    at the state the step left.  Written out here rather than routed through
    :func:`~cleverly.fluctuation._score.score_columns` deliberately: two calls of one function
    cannot fail against the same class of error, which is ``CLAUDE.md``'s rule about parity in
    the small.
    """

    step: int
    label: str
    quantity: str
    recorded: float
    recomputed: float

    @property
    def residual(self) -> float:
        return abs(self.recorded - self.recomputed)


def _bounded_arms(
    g: np.ndarray, arms: tuple[float, ...], bounds: tuple[float, float]
) -> np.ndarray:
    r""":math:`g^b(a \mid W)` as ``(n, K)``, by the **complement** rule two arms take.

    :meth:`~cleverly.estimators._nuisance.Propensity.bounded` clips :math:`g_1` and takes the
    other arm as its complement rather than clipping both columns, which is not the same thing
    at an asymmetric bound.  Reproduced here rather than imported, because an identity that
    calls the function it is checking checks nothing.
    """
    lower, upper = bounds
    one = np.clip(np.asarray(g, dtype=float), lower, upper)
    columns = {1.0: one, 0.0: 1.0 - one}
    return np.column_stack([columns[float(arm)] for arm in arms])


def identities(trace_: Trace) -> list[IdentityRow]:
    r"""Recompute every recorded score from the state its step left.

    Three equations, in the numbering ``docs/drtmle/theorem-concordance.md`` uses:

    .. math::

        (8)  \quad & P_n[w\,1_a/g^b(a|W)\,\{Y - \bar Q^*(a, W)\}] \\
        (9)  \quad & P_n[w\,Q_r(a, W)/g^b(a|W)\,\{1_a - g^b(a|W)\}] \\
        (10) \quad & P_n[w\,1_a\,g_{r,2}(a|W)/g^b_{r,1}(a|W)\,\{Y - \bar Q^*(a, W)\}]

    **The covariate is read off the state the step started from and the fitted value off the
    state it left, and that asymmetry is the content of the check rather than a detail.**  A
    solver zeroes its score at the covariate it was handed -- the pre-update one -- against
    the fit it produced; recomputing both halves at the *same* state gives a different number
    and would read as a defect.  It is exactly the distinction ``docs/roadmap.md``'s item 20
    turned on, where equation (9) was solved at the raw tilt while the curve subtracted the
    truncated one: two expressions, one array, and a check that read only the array could not
    see it.

    Plus, for every logistic step, that the update **is** the submodel it claims to be:
    :math:`\bar Q^{\text{post}} = \operatorname{expit}(\operatorname{logit}\bar Q^{\text{pre}}
    + H\epsilon)`.  For the mechanism that is exact -- the tilt is formed once from the final
    coefficient (:func:`~cleverly.fluctuation.mechanism.apply_mechanism_tilt`).  For the
    outcome it is a *diagnostic*: the Newton solve applies its tilt once per step and shrinks
    after each, so a caller holding ``(initial, epsilon)`` recovers the endpoint only where no
    iterate touched the shrinkage bound -- which is the whole reason
    :attr:`~cleverly.fluctuation.iterative.Fluctuation.carried` exists.  So it is reported as a
    residual, and a nonzero one is a fact about the path rather than a failure.

    The closing pass's ``"joint"`` step is scored on **both** submodels, since one four-column
    Newton solve is what drives equations (8) and (10) to zero together there.
    """
    arms = trace_.arms
    w = trace_.weights
    mask = trace_.observed.astype(bool)
    a = trace_.treatment
    y = trace_.outcome
    indicator = np.column_stack([(a == float(arm)).astype(float) for arm in arms])
    lower, upper = trace_.bounds
    rows: list[IdentityRow] = []
    for step in trace_.steps:
        if step.equation == "refit":
            continue
        before, after = step.before, step.after
        residual = np.zeros_like(after.q)
        residual[mask] = y[mask, None] - after.q[mask]
        if step.equation in {"8", "joint"}:
            covariate = indicator / _bounded_arms(before.g, arms, trace_.bounds)
            recomputed = (w[:, None] * residual * covariate).mean(axis=0)
            rows.extend(_pair(step, "eq(8)", step.score[: len(arms)], recomputed, arms))
        if step.equation in {"10", "joint"}:
            covariate = indicator * before.gr2 / np.clip(before.gr1, lower, upper)
            recomputed = (w[:, None] * residual * covariate).mean(axis=0)
            recorded = step.score[len(arms) :] if step.equation == "joint" else step.score
            rows.extend(_pair(step, "eq(10)", recorded, recomputed, arms))
        if step.equation == "9":
            # The residual is at the *post*-tilt mechanism and the covariate at the pre-tilt
            # one, which is what `solve_bounded_mechanism` solves and what the reported
            # correction divides by -- both at the truncated array, which is piece B1b.
            covariate = before.qr / _bounded_arms(before.g, arms, trace_.bounds)
            post = _bounded_arms(after.g, arms, trace_.bounds)
            recomputed = (w[:, None] * covariate * (indicator - post)).mean(axis=0)
            rows.extend(_pair(step, "eq(9)", step.score, recomputed, arms))
        rows.append(_update_row(step, trace_))
    return rows


def _pair(
    step: Step, quantity: str, recorded: np.ndarray, recomputed: np.ndarray, arms: tuple[float, ...]
) -> list[IdentityRow]:
    if np.asarray(recorded).size != len(arms):
        return []
    return [
        IdentityRow(
            step=step.index,
            label=step.label,
            quantity=f"{quantity}[{arm:g}]",
            recorded=float(recorded[column]),
            recomputed=float(recomputed[column]),
        )
        for column, arm in enumerate(arms)
    ]


def _update_row(step: Step, trace_: Trace) -> IdentityRow:
    r"""How far the step's endpoint is from one application of its own submodel.

    Exact for equation (9): :func:`~cleverly.fluctuation.mechanism.apply_mechanism_tilt`
    forms the array once from the final coefficient, so
    :math:`g^{\text{post}} = \operatorname{clip}(\operatorname{expit}(\operatorname{logit}
    g^{\text{pre}} + H_g\epsilon))` is a law and not an approximation.

    A diagnostic for the outcome steps, and zero wherever no Newton iterate touched the
    shrinkage bound -- which is the ordinary case and the one this fixture is in.  Nonzero
    says the path mattered, which is a fact about the solve worth having in the record and
    precisely what :attr:`~cleverly.fluctuation.iterative.Fluctuation.carried` exists for.
    """
    arms = trace_.arms
    lower, upper = trace_.bounds
    epsilon = np.asarray(step.epsilon, dtype=float)
    indicator = np.column_stack([(trace_.treatment == float(arm)).astype(float) for arm in arms])
    if step.equation == "9":
        # `reduced_mechanism_covariate`'s own construction: +Q_r/g for the arm the tilt is on
        # and -Q_r/(1-g) for the one whose residual is its negation. The sign is the whole
        # content of that function, so an identity that imported it would check nothing.
        gb = _bounded_arms(step.before.g, arms, trace_.bounds)
        signs = np.array([1.0 if float(arm) == arms[-1] else -1.0 for arm in arms])
        design = signs * step.before.qr / gb
        if epsilon.shape[0] != design.shape[1]:
            return IdentityRow(step.index, step.label, "update(g)", 0.0, float("nan"))
        predicted = np.clip(_expit(_logit(step.before.g) + design @ epsilon), lower, upper)
        gap = float(np.max(np.abs(predicted - np.clip(step.after.g, lower, upper))))
        return IdentityRow(step.index, step.label, "update(g)", 0.0, gap)
    outcome = [
        indicator[:, j] / _bounded_arms(step.before.g, arms, trace_.bounds)[:, j]
        for j in range(len(arms))
    ]
    extra = [
        indicator[:, j] * step.before.gr2[:, j] / np.clip(step.before.gr1[:, j], lower, upper)
        for j in range(len(arms))
    ]
    columns = {"8": outcome, "10": extra, "joint": outcome + extra}[step.equation]
    design = np.column_stack(columns)
    if epsilon.shape[0] != design.shape[1]:
        return IdentityRow(step.index, step.label, "update(Q)", 0.0, float("nan"))
    predicted = _expit(_logit(step.before.q_obs) + design @ epsilon)
    gap = float(np.max(np.abs(predicted - step.after.q_obs)))
    return IdentityRow(step.index, step.label, "update(Q)", 0.0, gap)


def _logit(p: Any) -> Any:
    q = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    return np.log(q / (1.0 - q))


def vintages(trace_: Trace) -> list[dict[str, Any]]:
    """Which reduced regressions a refit produced, and which of them the next equation used.

    One row per ``"refit"`` step: for each of :math:`Q_r`, :math:`g_{r,1}` and
    :math:`g_{r,2}`, whether the array the *next* solve's covariate was built from is the one
    this refit produced.  ``True`` everywhere is a single-vintage round; a ``False`` is a
    field the round kept from an earlier fit.

    This is the fourth row of ``docs/roadmap.md``'s R3 table -- *reduced outcome refit: refits
    all reductions again* against *refits* :math:`Q_r` *after both outcome updates* -- read
    off a run rather than off a reading of two sources.  It is the column F3 aligns on, and
    the reason it exists is that the difference is invisible in every field a fitted result
    carries.
    """
    rows: list[dict[str, Any]] = []
    for position, step in enumerate(trace_.steps):
        if step.equation != "refit":
            continue
        following = next(
            (
                later
                for later in trace_.steps[position + 1 :]
                if later.equation in {"9", "10", "joint"}
            ),
            None,
        )
        row: dict[str, Any] = {"step": step.index, "round": step.round, "phase": step.phase}
        for name in ("qr", "gr1", "gr2"):
            produced = getattr(step.after, name)
            row[name] = (
                None
                if following is None
                else bool(np.array_equal(produced, getattr(following.before, name)))
            )
        rows.append(row)
    return rows


def degeneracy(trace_: Trace) -> dict[str, float]:
    r"""How far the fixture is from the regime in which this instrument is blind.

    At correct nuisances :math:`Q_r` and :math:`g_{r,2}` are zero row by row, both corrections
    are zero arrays, and the reported curve equals :math:`D^*`; a trace taken there passes
    against a flipped sign, a swapped update order and a stale reduction alike.  These four
    numbers are what say the fixture is not there -- and the test that reads them is what stops
    a later "tidy the fixture" from quietly making every comparison downstream vacuous.
    """
    last = trace_.steps[-1].after
    scale = float(np.mean(np.abs(trace_.outcome)))
    return {
        "max|Q_r|": float(np.max(np.abs(last.qr))),
        "max|g_r2|": float(np.max(np.abs(last.gr2))),
        "max|D*_g|": max(
            (
                float(np.max(np.abs(v)))
                for k, v in trace_.corrections.items()
                if k.startswith("D*_g")
            ),
            default=0.0,
        ),
        "max|D*_Q|": max(
            (
                float(np.max(np.abs(v)))
                for k, v in trace_.corrections.items()
                if k.startswith("D*_Q")
            ),
            default=0.0,
        ),
        "mean|Y|": scale,
    }


# ------------------------------------------------------------------ output


def digest(trace_: Trace) -> dict[str, Any]:
    """A hashable summary: every scalar, and a SHA-256 per recorded array.

    What determinism is checked on.  Hashes rather than arrays because the check is *equality*
    -- two runs of one deterministic pipeline either produce the same float64 bytes or they do
    not, and a tolerance here would hide the drift it exists to catch.
    """
    payload: dict[str, Any] = {
        "order": trace_.order,
        # In the payload rather than only in the filename, so that a digest carries which
        # experiment it is a digest *of*. Without it two fixtures' digests are distinguishable
        # only by whatever their arrays happen to differ in, which is not provenance.
        "fixture_version": trace_.fixture_version,
        "arms": list(trace_.arms),
        "bounds": list(trace_.bounds),
        "exit": dict(trace_.exit),
        "clipped": trace_.clipped,
        "estimates": trace_.estimates,
        "steps": [],
    }
    for step in trace_.steps:
        payload["steps"].append(
            {
                "index": step.index,
                "phase": step.phase,
                "round": step.round,
                "equation": step.equation,
                "names": list(step.names),
                "n_iter": step.n_iter,
                "failure": step.failure,
                "converged": step.converged,
                "epsilon": _hash(step.epsilon),
                "score": _hash(step.score),
                "before": {name: _hash(a) for name, a in step.before.as_dict().items()},
                "after": {name: _hash(a) for name, a in step.after.as_dict().items()},
            }
        )
    payload["curve"] = {name: _hash(values) for name, values in trace_.curve.items()}
    payload["corrections"] = {name: _hash(values) for name, values in trace_.corrections.items()}
    return payload


def _hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=float))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def write_trace(trace_: Trace, directory: Path) -> tuple[Path, Path]:
    """Arrays to ``.npz`` and everything else to ``.json``, one pair per order.

    Two files rather than one because they are read by different readers: the JSON is what a
    person and a diff look at, and the ``.npz`` is what F3's comparison loads.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "weights": trace_.weights,
        "treatment": trace_.treatment,
        "outcome": trace_.outcome,
        "observed": trace_.observed,
    }
    for step in trace_.steps:
        for side, state in (("before", step.before), ("after", step.after)):
            for name, values in state.as_dict().items():
                arrays[f"step{step.index:03d}.{side}.{name}"] = values
        arrays[f"step{step.index:03d}.epsilon"] = step.epsilon
        arrays[f"step{step.index:03d}.score"] = step.score
    for name, values in trace_.curve.items():
        arrays[f"curve.{name}"] = values
    for name, values in trace_.corrections.items():
        arrays[f"correction.{name}"] = values
    # The trace's own version, never the module global: `write_trace` read `FIXTURE_VERSION`
    # until F3-closeout, so a `v2` trace was written to a `v1` filename and every reader of it
    # was reading a label that had nothing to do with the run.
    stem = f"drtmle_trace_{trace_.fixture_version}_{trace_.order}"
    npz = directory / f"{stem}.npz"
    js = directory / f"{stem}.json"
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps(digest(trace_), indent=2) + "\n", encoding="utf-8")
    return npz, js


def compare(left: Trace, right: Trace) -> list[dict[str, Any]]:
    """The two update orders side by side, **reported and not asserted equal**.

    Whether ``"cleverly"`` and ``"paper"`` reach the same fixed point on real data is
    ``docs/roadmap.md``'s item 22 -- a measurement, and one this harness supplies an input to
    rather than settles.  The comparison is of the **estimates and the scores**, never of the
    fluctuation coefficients: the submodels a round passes through differ between the two
    orders, so an ``epsilon`` from one is not an ``epsilon`` from the other
    (``docs/drtmle/theorem-concordance.md`` §6).
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(set(left.estimates) & set(right.estimates)):
        rows.append(
            {
                "quantity": f"psi[{name}]",
                left.order: left.estimates[name]["psi"],
                right.order: right.estimates[name]["psi"],
                "difference": left.estimates[name]["psi"] - right.estimates[name]["psi"],
            }
        )
        rows.append(
            {
                "quantity": f"se[{name}]",
                left.order: left.estimates[name]["se"],
                right.order: right.estimates[name]["se"],
                "difference": left.estimates[name]["se"] - right.estimates[name]["se"],
            }
        )
    for name in sorted(set(left.curve) & set(right.curve)):
        rows.append(
            {
                "quantity": f"max|dD|[{name}]",
                left.order: float("nan"),
                right.order: float("nan"),
                "difference": float(np.max(np.abs(left.curve[name] - right.curve[name]))),
            }
        )
    for key in ("rounds", "closing", "ill_conditioned"):
        rows.append(
            {
                "quantity": key,
                left.order: left.exit[key],
                right.order: right.exit[key],
                "difference": left.exit[key] - right.exit[key],
            }
        )
    return rows


def _report(trace_: Trace) -> str:
    lines = [
        f"order = {trace_.order}; rounds = {trace_.exit['rounds']}; "
        f"exit = {trace_.exit['exit_reason']}; closing = {trace_.exit['closing']} "
        f"(capped={trace_.exit['closing_capped']}); ill-conditioned rounds = "
        f"{trace_.exit['ill_conditioned']}; clipped (row, arm) pairs = {trace_.clipped}",
        "",
        f"{'step':>4}  {'phase':<6} {'rnd':>3}  {'eq':<5} {'n_iter':>6}  {'|epsilon|':>11}  "
        f"{'max|score|':>11}",
    ]
    for step in trace_.steps:
        eps = float(np.max(np.abs(step.epsilon))) if step.epsilon.size else float("nan")
        score = float(np.max(np.abs(step.score))) if step.score.size else float("nan")
        lines.append(
            f"{step.index:>4}  {step.phase:<6} {step.round:>3}  {step.equation:<5} "
            f"{step.n_iter:>6}  {eps:>11.4g}  {score:>11.4g}"
        )
    rows = identities(trace_)
    worst = max(rows, key=lambda row: row.residual) if rows else None
    lines += ["", f"identities: {len(rows)} recomputed from the recorded state"]
    if worst is not None:
        lines.append(
            f"  worst residual {worst.residual:.3g} at step {worst.step} "
            f"({worst.label}, {worst.quantity})"
        )
    lines += ["", "degeneracy (how far from the regime this instrument is blind in):"]
    lines += [f"  {name:<12} {value:.4g}" for name, value in degeneracy(trace_).items()]
    lines += ["", "reduction vintages -- did the next equation use what this refit produced?"]
    lines += [
        f"  step {row['step']:>3} round {row['round']:>2}  "
        f"qr={row['qr']!s:<5} gr1={row['gr1']!s:<5} gr2={row['gr2']!s:<5}"
        for row in vintages(trace_)
    ]
    stalled = _stalled_closing(trace_)
    lines += [
        "",
        f"closing pass: {stalled} of its equation-(9) steps moved nothing "
        "(zero epsilon, unchanged score)",
    ]
    return "\n".join(lines)


def _stalled_closing(trace_: Trace) -> int:
    """Closing equation-(9) steps that returned a zero coefficient and left the score alone.

    The stage stops on ``max_steps`` or on ``spec.tol`` and has no test for *"this step did
    nothing"*, so a fixed point it does not recognise is spent as iterations rather than
    reported.  Counted here rather than judged: it is a fact about the loop, it is one of the
    things F4's pre-close/post-close column is for, and F2 may not change it.
    """
    return sum(
        1
        for step in trace_.steps
        if step.phase == "close"
        and step.equation == "9"
        and step.epsilon.size
        and not np.any(step.epsilon)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--order",
        default="cleverly",
        choices=["cleverly", "paper"],
        help="which update order to trace",
    )
    parser.add_argument("--both", action="store_true", help="trace both orders and compare them")
    parser.add_argument(
        # Until F3-closeout this CLI could only ever run `v1`, because it had no flag and
        # `read_fixture()` takes the module default -- so the second fixture was reachable
        # from `drtmle_r_compare` and from nowhere here.
        "--fixture-version",
        default=FIXTURE_VERSION,
        choices=sorted(FIXTURES),
        help="which frozen fixture: v1 (truncation slack) or v2 (truncation binds)",
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="regenerate the fixture named by --fixture-version and exit; every trace already "
        "taken is against the old bytes, so say in the commit message what moved",
    )
    parser.add_argument("--out", type=Path, default=None, help="directory to write the trace to")
    args = parser.parse_args()

    version = str(args.fixture_version)
    if args.write_fixture:
        fixture = write_fixture(version=version)
        print(
            f"wrote {fixture_path('csv', version=version)} ({fixture.n} rows), "
            f"sha256 {fixture.manifest['sha256']}"
        )
        return

    fixture = read_fixture(version=version)
    orders = ["cleverly", "paper"] if args.both else [args.order]
    traces = []
    for order in orders:
        traced = trace(fixture, order=order)
        traces.append(traced)
        print(_report(traced))
        print()
        if args.out is not None:
            npz, js = write_trace(traced, args.out)
            print(f"wrote {npz} and {js}\n")
    if len(traces) == 2:
        print("the two orders, reported and not asserted equal (roadmap item 22):")
        for row in compare(*traces):
            print(f"  {row['quantity']:<20} difference {row['difference']:>12.6g}")


if __name__ == "__main__":
    main()
