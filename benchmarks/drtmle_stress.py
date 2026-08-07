r"""F5's applied stress cell: real misspecification rather than engineered drift.

``docs/roadmap.md``'s F5 phase 2 carries F8's ten clauses, and **clause 3** asks for "a
reproduction of the published paper's relevant simulation setting *where feasible*, and a
recorded statement if it is not".

**It is not feasible here, and this module is what runs in its place.**  Three lines of the
repository's own record say why, and they are cited rather than summarised:

* ``docs/references.md`` marks Benkeser, Carone, van der Laan & Gilbert (2017), *Biometrika*
  104(4):863--880 -- the published paper -- **"Not read here"**;
* ``docs/drtmle/theorem-concordance.md`` marks the same source ``in repository: no``, and its
  coverage table lists what *was* read first-hand from the 2016 working paper: Theorem 1, the
  corrected influence function, appendix B's remainder terms, appendix A/B's rate conditions,
  the empirical-process conditions and the recursive algorithm.  **No simulation section
  appears in that list**;
* ``docs/drtmle/investigation-log.md`` records the reason: only the working paper's text --
  "Theorem 1 and appendices A to C" -- is obtainable without a Biometrika subscription.

So reproducing "the published paper's simulation setting" would mean inventing a
data-generating process and attributing it to a paper nobody here has read.  That is a
fabricated citation, and no amount of care in the rest of the study repairs one.  Clause 3's
own escape hatch is taken instead: the statement above **is** the recorded statement.

What runs instead, and why it is not a consolation prize
--------------------------------------------------------

**Gate 2 clause 4 is a committed clause C3c never read**: ``docs/drtmle/validation-plan.md``
asks that "the advantage persists in at least one applied stress setting", and
``docs/drtmle/coverage-study.md``'s gate readout records it **not read here**.  Discharging it
is a deliverable in its own right rather than a substitute for one.

The cell is :func:`base_law` -- ``nonlinear_dgp()`` -- with the ``"fast"`` learner preset on
**both** primary nuisances.  The contrast with the drift cells is the whole point of it:

============================  =========================================================
``q-drift`` / ``g-drift``     ``nonlinear`` (this cell)
============================  =========================================================
one nuisance correct, the     **both** nuisances misspecified for a GLM by construction
other wrong

misspecification injected at  misspecification is a property of the law, at whatever rate
a prescribed ``n^{-alpha}``   a flexible library happens to achieve

the regime Theorem 1's rate   a regime no rate condition is claimed for
conditions are stated in
============================  =========================================================

That is the situation a caller is actually in, and it is the one setting where "does the
correction still help?" is a question about practice rather than about a construction.

Three things this cell is **not**, each declared before the first fit
---------------------------------------------------------------------

**Its coverage is descriptive and is not a release number.**  The release criterion is read in
the drift cells, where the misspecification regime is the one the theorem is stated in.  A
coverage reading here says whether the advantage persists; it does not certify anything.

**Its remainder is not read against clause 4's vanishing trend.**  With both primaries
inconsistent no theorem predicts ``sqrt(n) R_remaining -> 0``, so a non-vanishing remainder
here is the expected reading rather than a failure.  **Item 13 closes on the drift cells and
only on them.**  Stating that in advance is what stops this cell's remainder being read later
as evidence it was never built to give.

**It carries no ceiling arm.**  None of F5's six branches reads a ceiling outside the drift
cells, and a fit spent on a branch nobody reads is the "nice to have" the terminal plan
forbids.

Why ``nonlinear_dgp`` and not ``weak_overlap_dgp``
--------------------------------------------------

``weak_overlap_dgp`` is the obvious alternative and it is **rejected on the scope decision, not
on cost**.  Its propensity is ``expit(3 W1 + 2.1 W2)``, whose linear predictor has a standard
deviation near ``3.7``, so its scores crowd hard against ``0`` and ``1`` and nearly every fit
would exit bound-active.  ``docs/drtmle/theorem-concordance.md`` section 7 scopes the guarantee
to fits where the truncation is **inactive** and covers neither truncation, so such a cell
could support no theorem-backed claim at all -- and a stress design whose every row is out of
scope spends fits and reads as evidence.  That is the same reasoning F4 recorded when it moved
its truncation contrast off a cohort and onto two frozen fixtures.

The law is admissible to the companion machinery, which is not incidental:
:func:`benchmarks.drtmle_remainder._refuse_unsupported` refuses a law that is not gaussian, or
that carries hidden latents, missingness, an intermediate node or clustering.
``nonlinear_dgp`` is four observed standard-normal latents with an additive-error gaussian
outcome, so the quadrature companion, ``truth_at`` and ``remainder_rows`` all work unchanged
and the remainder column is *available* -- it is simply not read against clause 4.

``tests/unit/test_drtmle_stress.py`` asserts the misspecification rather than describing it: a
GLM's held-out error on both ``g`` and ``Qbar`` must be materially above a flexible fit's, so a
cell that silently stopped being a stress fails rather than quietly reporting an easy law.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleverly.datasets import DGP, nonlinear_dgp
from cleverly.estimators.base import format_table
from cleverly.learners import fit_learner, predict_mean
from cleverly.learners.super_learner import resolve_learner
from cleverly.utils.bounds import OutcomeScaler

__all__ = [
    "CELLS",
    "LEARNER",
    "MISSPECIFICATION_FLOOR",
    "SIZES",
    "base_law",
    "misspecification_reading",
    "settings",
    "summary_rows",
]

#: The one cell.  A tuple rather than a bare name so this module presents the same surface as
#: :mod:`benchmarks.drtmle_tier2` and :mod:`benchmarks.drtmle_injection` -- the F5 harness
#: dispatches on a cell name and must not branch on which module supplied it.
CELLS = ("nonlinear",)

#: The two sizes this cell runs at.  Not three: the middle size exists in the drift cells for
#: clause 4's ``sqrt(n) R_remaining`` trend, and this cell's remainder is declared **not** to be
#: read against that trend, so a third size here would buy a rate nobody reads.
SIZES = (600, 2_400)

#: The primary-nuisance library, and it is the point of the cell rather than a setting on it.
#: ``"fast"`` is ``mean + glm + gam + boost`` -- a flexible library estimating both nuisances at
#: whatever rate it achieves on this law, which is the applied situation.  The drift cells hand
#: one nuisance a kernel with a **committed bandwidth sequence** so that "the intended asymptotic
#: regime was entered" is checkable; nothing of the sort is claimed here, and that is the
#: contrast the cell exists to draw.
LEARNER = "fast"

#: How much worse a GLM must be than the flexible library, in held-out mean squared error, for
#: this cell to be the stress it says it is.  A ratio rather than a difference, and on **both**
#: nuisances: the whole design is "neither nuisance is well specified", so a law on which a GLM
#: does nearly as well is not this cell however nonlinear its formula looks.
#:
#: ``1.05`` is deliberately a floor and not a target -- the measured ratios are far above it (see
#: :func:`misspecification_reading`), and a floor that a genuinely misspecified law clears by an
#: order of magnitude is a guard against the law silently changing, not a characterisation of it.
MISSPECIFICATION_FLOOR = 1.05

#: Folds, matching the drift cells so the two are comparable on everything except the law and
#: the library.  ``learner_folds`` is the inner split ``SuperLearner`` weights its candidates on.
N_FOLDS = 5
LEARNER_FOLDS = 3

#: The estimands every fit reports, in the drift cells' order.
ESTIMANDS = ("ate", "ey1", "ey0")


def base_law() -> DGP:
    """The stress law: nonlinear, heterogeneous and interacted in both nuisances.

    ``nonlinear_dgp`` rather than a law written here, because a DGP with a committed truth and
    an existing test tier is worth more than a bespoke one -- ``DGP.truth`` integrates by
    quasi-Monte Carlo on the law's own Sobol grid, which is the same rule
    :func:`benchmarks.drtmle_remainder.truth_at` reads, so the estimand and the companion's
    integral are one quantity rather than two that have to be argued equal.

    Its propensity is ``expit(0.6 W1 - 0.4 W2^2 + 0.5 W2 W3 + 0.3 * 1{W4 > 0})`` and its outcome
    mean carries a sine, a square, an interaction and an absolute value with a heterogeneous
    effect ``2.0 + 0.7 W1 - 0.5 * 1{W2 > 0}``.  A GLM is wrong for both, which
    :func:`misspecification_reading` measures rather than asserts.
    """
    return nonlinear_dgp()


def settings(cell: str, n: int) -> dict[str, Any]:
    """The shared estimator settings for one cell, at one size.

    Signature matched to :func:`benchmarks.drtmle_tier2.settings` so the F5 harness holds one
    call site rather than a branch: ``cell`` is validated and ``n`` is accepted and unused, since
    nothing here is size-dependent -- the drift cells' kernel bandwidth is, and the absence of
    any such sequence is exactly what makes this cell the applied one.

    The reduced learners are **not** set here.  Each F5 arm supplies its own pair, and a default
    left in this dictionary would silently win the argument the study is about.
    """
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
    del n
    return {
        "outcome_learner": LEARNER,
        "treatment_learner": LEARNER,
        "n_folds": N_FOLDS,
        "learner_folds": LEARNER_FOLDS,
        "simultaneous": False,
        "estimands": ESTIMANDS,
    }


def misspecification_reading(
    *, n: int = 4_000, seed: int = 20260204, folds: int = 5
) -> list[dict[str, Any]]:
    r"""How much worse a GLM is than the flexible library, on each primary nuisance.

    **The declared misspecification, measured rather than described.**  Both nuisances are
    supposed to be beyond a GLM on this law; a cell where that quietly stopped being true would
    still run, still report coverage, and would no longer be a stress setting -- and nothing in
    a coverage table would say so.

    Read as a **cross-fitted** held-out mean squared error, so the comparison is out of sample on
    both sides and a flexible library cannot win by fitting the noise.  ``Qbar`` is scored on the
    treated and control arms separately at the observed rows, since that is the object the
    recursion regresses; ``g`` is scored as a Brier score on the arm indicator.

    **The outcome is scaled to [0, 1] first, and that is not cosmetic.**
    :func:`~cleverly.learners.super_learner.resolve_learner` builds every ``SuperLearner`` with
    ``clip=(0.0, 1.0)``, which is correct inside the estimator because the outcome has already
    been mapped onto the unit interval by :class:`~cleverly.utils.bounds.OutcomeScaler`.  Scoring
    a raw-scale outcome through that same constructor clips every prediction into ``[0, 1]`` and
    reports the clipping as model error: the first run of this function read ``10.95`` against
    ``10.94`` on ``Qbar[A=1]`` -- two libraries indistinguishable because neither was allowed to
    predict the answer.  Scaling first is what makes the ratio a statement about the libraries.

    Returns one row per ``(nuisance, arm)`` with both errors and their ratio.  The ratio is what
    ``tests/unit/test_drtmle_stress.py`` asserts against :data:`MISSPECIFICATION_FLOOR`.

    This is a diagnostic and no fit reads it: it runs at its own size and its own seed, on a
    stream nothing else uses, precisely so that measuring the property cannot perturb the study
    that relies on it.
    """
    dgp = base_law()
    frame, _ = dgp.sample(n, seed=seed)
    covariates = np.column_stack(
        [np.asarray(frame[name], dtype=float) for name in ("W1", "W2", "W3", "W4")]
    )
    treatment = np.asarray(frame["A"], dtype=float)
    outcome = np.asarray(frame["Y"], dtype=float)

    scaler = OutcomeScaler.from_outcome(outcome)
    scaled = (outcome - scaler.lower) / scaler.range

    rng = np.random.default_rng(seed)
    assignment = rng.permutation(np.arange(n) % folds)

    truth_g = np.asarray(dgp.propensity(covariates), dtype=float)
    truth_q1 = (
        np.asarray(dgp.outcome_mean(covariates, 1.0, None), dtype=float) - scaler.lower
    ) / scaler.range
    truth_q0 = (
        np.asarray(dgp.outcome_mean(covariates, 0.0, None), dtype=float) - scaler.lower
    ) / scaler.range

    rows: list[dict[str, Any]] = []
    for label, task, target, oracle, mask in (
        ("g", "classification", treatment, truth_g, np.ones(n, dtype=bool)),
        ("Qbar[A=1]", "regression", scaled, truth_q1, treatment == 1.0),
        ("Qbar[A=0]", "regression", scaled, truth_q0, treatment == 0.0),
    ):
        risk: dict[str, float] = {}
        excess: dict[str, float] = {}
        for library in ("glm", LEARNER):
            predicted = _held_out(
                library, task, covariates[mask], target[mask], assignment[mask], folds
            )
            risk[library] = float(np.mean((target[mask] - predicted) ** 2))
            excess[library] = float(np.mean((oracle[mask] - predicted) ** 2))
        ratio = excess["glm"] / excess[LEARNER] if excess[LEARNER] > 0 else float("inf")
        rows.append(
            {
                "nuisance": label,
                "rows": int(mask.sum()),
                "glm_risk": risk["glm"],
                "flex_risk": risk[LEARNER],
                "glm_excess": excess["glm"],
                "flex_excess": excess[LEARNER],
                "ratio": ratio,
                "clears_floor": bool(ratio >= MISSPECIFICATION_FLOOR),
            }
        )
    return rows


def _held_out(
    library: str,
    task: Any,
    design: np.ndarray,
    target: np.ndarray,
    assignment: np.ndarray,
    folds: int,
) -> np.ndarray:
    """Out-of-fold predictions from one library, through the package's own fit/predict pair.

    A plain fold loop rather than
    :func:`~cleverly.estimators._nuisance.cross_fit_predictions`, which is built around a
    ``Folds`` object and an estimator's configuration and would drag the whole nuisance
    orchestration layer into a two-library comparison.
    :func:`~cleverly.learners.super_learner.resolve_learner`
    builds the same :class:`~cleverly.learners.SuperLearner` an estimator would, and
    :func:`~cleverly.learners.fit_learner` / :func:`~cleverly.learners.predict_mean` are the
    calls it would make -- so the comparison is of the libraries and not of two harnesses.
    """
    predicted = np.full(target.shape[0], np.nan, dtype=float)
    for fold in range(folds):
        held = assignment == fold
        train = ~held
        if not held.any() or not train.any():
            continue
        model = fit_learner(
            resolve_learner(library, task=task, n_folds=LEARNER_FOLDS, random_state=0),
            design[train],
            target[train],
        )
        predicted[held] = predict_mean(model, design[held], task)
    return predicted


MISSPECIFICATION_HEADERS = (
    "nuisance",
    "rows",
    "glm risk",
    f"{LEARNER} risk",
    "glm excess",
    f"{LEARNER} excess",
    "excess ratio",
    "clears",
)


def summary_rows() -> list[list[str]]:
    """One row per nuisance: the declared misspecification, printed before any fit."""
    return [
        [
            row["nuisance"],
            str(row["rows"]),
            f"{row['glm_risk']:.5f}",
            f"{row['flex_risk']:.5f}",
            f"{row['glm_excess']:.5f}",
            f"{row['flex_excess']:.5f}",
            f"{row['ratio']:.2f}x",
            "yes" if row["clears_floor"] else "NO",
        ]
        for row in misspecification_reading()
    ]


def _main() -> int:  # pragma: no cover - a convenience entry point, not a study phase
    print(f"F5 stress cell: {base_law().name}, library {LEARNER!r}, sizes {list(SIZES)}")
    print()
    print(format_table(MISSPECIFICATION_HEADERS, summary_rows()))
    print()
    print(
        "`risk` is the held-out error against the observed target and is context; `excess` is the\n"
        "error against the law's own function, which is the quantity Theorem 1's rate conditions\n"
        "are stated in. The verdict is on the excess ratio: every nuisance must clear "
        f"{MISSPECIFICATION_FLOOR:g}x\nfor this cell to be the stress setting it declares."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
