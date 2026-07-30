r"""A three-armed finite-support law that a finite sample realises *exactly*.

The sibling of :mod:`tests.discrete_law`, and it exists for the same reason: the
efficient influence function is a property of a distribution, not of a sample, so
checking that the library computes the right one needs a distribution the test can hold
in its hand.  What this one adds is a **third treatment arm**, which is the whole point
-- the two-armed law cannot distinguish an implementation that genuinely keys everything
by arm from one that happens to have two columns and calls them ``0`` and ``1``.

``W`` takes three values, ``A`` takes three, ``Y`` is binary, and every cell probability
is an exact multiple of ``1 / N``.  Laying out ``N`` rows in the cell proportions makes
the empirical distribution *equal* to the data-generating one, which buys the same three
things it does there:

* every population quantity is a closed-form function of the eighteen cell
  probabilities, computable without touching any library code;
* handed the oracle nuisances, the initial fit is exactly correct *in the sample*, so
  the targeting step's score is zero at :math:`\epsilon = 0` and the influence curve the
  estimator reports is the EIF at :math:`P_0` rather than an estimate of it;
* there is no sampling error anywhere, so the assertions can be exact.

Positivity holds comfortably by construction -- every arm has probability at least
``0.2`` at every covariate value, and every conditional mean sits in ``[0.15, 0.85]`` --
so no truncation or shrinkage is active and the estimator runs on the unmodified
nuisances.  That matters more here than in the binary case: with three arms the bounds
are applied arm by arm and *not* renormalised, so a law where they bind would make the
comparison against the oracle a comparison against a different estimand.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

#: Rows in the realised sample.  Every cell probability below is a multiple of ``1 / N``.
N = 2000

#: Number of treatment arms.  Three is the smallest number that is not two, and every
#: bug this module exists to catch shows up at three.
K = 3

#: ``P(W = w)`` for ``w in {0, 1, 2}``.
P_W = np.array([0.50, 0.30, 0.20])

#: ``g(a | w) = P(A = a | W = w)``, indexed ``[w, a]``.  Rows sum to one; every entry is
#: at least 0.2, so ``g_bounds`` never binds and the per-arm truncation stays inert.
G = np.array(
    [
        [0.50, 0.30, 0.20],
        [0.25, 0.50, 0.25],
        [0.20, 0.20, 0.60],
    ]
)

#: ``Qbar(a, w) = P(Y = 1 | A = a, W = w)``, indexed ``[w, a]``.  Deliberately *not*
#: monotone or additive in ``a``: a design that forced a linear dose-response -- a single
#: numeric treatment column, say -- cannot reproduce these and the oracle comparison
#: fails rather than passing on a coincidence.
Q = np.array(
    [
        [0.40, 0.75, 0.55],
        [0.20, 0.35, 0.80],
        [0.60, 0.30, 0.85],
    ]
)

#: The support, ordered ``(w, a, y)``.  Row blocks of the sample follow this order.
SUPPORT: tuple[tuple[int, int, int], ...] = tuple(itertools.product(range(3), range(K), range(2)))


def _cell_counts() -> np.ndarray:
    """``N * P(W = w, A = a, Y = y)`` as a ``(3, K, 2)`` integer array."""
    counts = np.empty((3, K, 2))
    for w, a, y in SUPPORT:
        outcome = Q[w, a] if y == 1 else 1.0 - Q[w, a]
        counts[w, a, y] = P_W[w] * G[w, a] * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- adjust N, P_W, G or Q"
        )
    return rounded.astype(int)


#: Cell counts in the realised sample.  Integral by construction -- checked above.
COUNTS = _cell_counts()

#: ``P(W, A, Y)``, shape ``(3, K, 2)``.  Taken from the counts rather than from the
#: constants above, so it is bit-for-bit the empirical law of :func:`frame`.
PROBS = COUNTS / N

#: The labels the treatment column carries.  Strings rather than numbers so the tests
#: also pin that reported parameter names use the analyst's levels, and so that a helper
#: which quietly assumed the codes *were* the levels shows up.
LABELS: tuple[str, ...] = ("low", "mid", "high")

#: Arm code -> label.  ``CausalData`` sorts levels, and these three sort to
#: ``("high", "low", "mid")``, so the code for "low" is 1 rather than 0 -- which is
#: exactly the confusion a test should be exposed to rather than shielded from.
SORTED_LABELS: tuple[str, ...] = tuple(sorted(LABELS))
#: Index into :data:`LABELS` (this module's arm axis) for each sorted arm code.
ARM_OF_CODE: tuple[int, ...] = tuple(LABELS.index(label) for label in SORTED_LABELS)


def frame(*, labelled: bool = True) -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law.

    Rows are laid out in :data:`SUPPORT` order, one contiguous block per support point.
    ``labelled=False`` leaves the treatment as the numeric arm index of this module,
    which is what a test comparing against :data:`Q` directly wants; the default labels
    it, which is what a test about reported names wants.
    """
    counts = [COUNTS[w, a, y] for w, a, y in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=float)[cells]
    treatment: Any = columns[:, 1]
    if labelled:
        treatment = np.array(LABELS)[columns[:, 1].astype(int)]
    return pd.DataFrame({"W": columns[:, 0], "A": treatment, "Y": columns[:, 2]})


def functional(probs: Any, estimand: str) -> Any:
    r"""The target parameter as a closed-form function of the cell probabilities.

    Written out longhand from the identification formula and sharing no code with the
    library, so comparing against it is a genuine check rather than a restatement.

    Every operation is arithmetic, so this stays analytic in the cell probabilities --
    which is what lets :func:`gateaux` differentiate it by a complex step.  Do not
    introduce ``clip``, ``abs`` or a comparison here.

    Names follow this module's own arm indices (``ey[0]`` is arm ``0`` = ``"low"``),
    not the codes ``CausalData`` assigns after sorting the labels.  :func:`reported_name`
    translates.
    """
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2))  # P(W = w)
    p_wa = p.sum(axis=2)  # P(W = w, A = a)
    q = p[:, :, 1] / p_wa  # E[Y | A = a, W = w]

    # E[Y(a)] = sum_w P(W = w) E[Y | A = a, W = w], one per arm.
    psi = [(p_w * q[:, a]).sum() for a in range(K)]

    if estimand.startswith("ey["):
        return psi[int(estimand[3:-1])]
    if estimand.startswith("ate["):
        arm, reference = (int(part) for part in estimand[4:-1].split(" vs "))
        return psi[arm] - psi[reference]
    if estimand.startswith("rr["):
        arm, reference = (int(part) for part in estimand[3:-1].split(" vs "))
        return np.log(psi[arm]) - np.log(psi[reference])
    if estimand.startswith("or["):
        arm, reference = (int(part) for part in estimand[3:-1].split(" vs "))
        return np.log(psi[arm] / (1.0 - psi[arm])) - np.log(psi[reference] / (1.0 - psi[reference]))
    raise ValueError(f"unknown estimand {estimand!r}")


def reported_name(estimand: str) -> str:
    """Translate an oracle name on *this module's* arm indices to the reported one.

    ``functional`` indexes arms as :data:`LABELS` orders them; the library reports them
    by the analyst's labels, which ``CausalData`` sorts.  So oracle ``"ate[2 vs 0]"`` --
    high against low -- is reported as ``"ate[high vs low]"``.  Doing the translation in
    one place keeps every test written in whichever vocabulary suits it.
    """
    stem, _, rest = estimand.partition("[")
    parts = rest[:-1].split(" vs ")
    labels = [LABELS[int(part)] for part in parts]
    return f"{stem}[{' vs '.join(labels)}]"


def gateaux(estimand: str, point: int, *, step: float = 1e-30) -> float:
    r"""The Gateaux derivative of ``estimand`` at support point ``point``.

    .. math::

        D^*(o) = \left.\frac{d}{dt}\, \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)
                 \right|_{t = 0}

    which for a pathwise-differentiable parameter in a nonparametric model *is* the
    efficient influence function, and here is derived from :func:`functional` alone --
    no clever covariate, no submodel, nothing the library supplies.

    Differentiation is by complex step, for the reasons
    :func:`tests.discrete_law.gateaux` sets out: the contamination path makes ``Psi`` a
    rational function of ``t``, hence analytic, and carrying the perturbation in the
    imaginary part avoids subtractive cancellation entirely.  The result is the
    derivative to full double precision.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand)) / step)


def eif(estimand: str) -> np.ndarray:
    """The EIF of ``estimand`` evaluated at every support point, in support order."""
    return np.array([gateaux(estimand, point) for point in range(len(SUPPORT))])


def cell_of_row() -> np.ndarray:
    """Support-point index for each row of :func:`frame`, in row order."""
    counts = [COUNTS[w, a, y] for w, a, y in SUPPORT]
    return np.repeat(np.arange(len(SUPPORT)), counts)


#: ``g(a | W = w)`` and ``E[Y | A = a, W = w]`` as the *realised sample* has them.
#: Equal to :data:`G` and :data:`Q` mathematically; derived from the counts so that the
#: oracle nuisances are exact in the sample down to the last bit.
G_EXACT = PROBS.sum(axis=2) / PROBS.sum(axis=(1, 2))[:, None]
Q_EXACT = PROBS[:, :, 1] / PROBS.sum(axis=2)


class OracleMultiTreatment(BaseEstimator):
    """A ``K``-class treatment model returning this law's own ``g(a | W)`` exactly.

    Duck-typed as a scikit-learn classifier, and deliberately reporting ``classes_`` in
    the library's *code* order rather than this module's arm order, so that a caller
    which lined the columns up by position instead of by class would be caught.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleMultiTreatment:
        self.classes_ = np.arange(float(K))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        w = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        # Column ``c`` is the arm with code ``c``, which is ``ARM_OF_CODE[c]`` here.
        return np.column_stack([G_EXACT[w, ARM_OF_CODE[code]] for code in range(K)])


class OracleMultiOutcome(BaseEstimator):
    """An outcome model returning this law's own ``E[Y | A, W]`` exactly.

    Reads the treatment off the design's indicator block rather than off a single
    column, because that is what the multi-arm design carries: ``K - 1`` drop-first
    indicators followed by the covariates.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleMultiOutcome:
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        indicators = design[:, : K - 1]
        # All-zero indicator row means the dropped first arm, i.e. code 0.
        code = np.where(indicators.any(axis=1), indicators.argmax(axis=1) + 1, 0)
        w = np.rint(design[:, K - 1]).astype(int)
        return Q_EXACT[w, np.array(ARM_OF_CODE)[code]]
