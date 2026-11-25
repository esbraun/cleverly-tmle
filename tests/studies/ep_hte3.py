r"""Declaration of the hte3 comparison of the EP curve (P1 plan, PR 4b, section 4).

**Question.**  On the same draws and the same nuisance estimates, is the EP curve of
:class:`cleverly.EPMethod` as accurate as the authors' implementation, hte3's
``Lrnr_cate_EP`` (``targeting_style="dr"``) and ``Lrnr_crr_EP`` at :data:`HTE3_COMMIT`?  The
comparison kind is ``different estimator``: hte3 debiases the CATE by EPL's linear Method 2
(glmnet ridge at lambda ``1e-8``) and this package by the logistic Method 1, and the two
cosine sieves are parameterized differently (hte3's ``Sieve::sieve_preprocess`` rescales each
modifier by its training 1% and 99% quantiles and clips to ``[0, 1]``; this package uses the
declared domain).  The paired statistic per replicate is

.. math::

   D = \log \mathrm{ISE}(\text{cleverly EP}) - \log \mathrm{ISE}(\text{hte3 EP}),

each ISE on the parent study's 20,000 fixed ISE points.

**Not a registered study.**  This is a comparison artifact on the precedent of
``tests/canonical/econml_dynamic_dml/`` (plan 4.1 (b)).  It has no ``StudyRecord``, no grid row
and no root seed of its own.  The parents ``ep-cate`` and ``ep-crr`` keep ``reference=None``,
"none claimed" in the grid and "No canonical implementation is compared." on their pages.

**Draws.**  Each cell re-draws its parent group's leading replicates from the parent's own
property stream (``stream_seed(parent.STUDY, "property_sample", group, r)``, the draw the
parent's properties phase made) and refits the parent's EP at ``random_state = r``.  The ISE
points are the parent's (``stream_seed(parent.STUDY, "ise_points", law)``).  The numpy pilot
draws on :data:`PILOT_STREAM` and the driver's smoke on :data:`SMOKE_STREAM`, both disjoint
from every draw a parent made.

**Nuisances.**  The Python phase exports ``W, A, Y, k`` and the fit's out-of-fold arm
regressions ``mu_n(0, W)``, ``mu_n(1, W)`` and propensity ``pi_n(a | W)`` after the fit's own
``g_bounds``.  hte3 receives them through ``make_hte3_Task_tx(pi.hat=, mu.hat=,
cross_fit_and_cv = FALSE)``.  hte3 then truncates each propensity column once more
(``resolve_contrast_nuisances`` calls ``truncate_propensity(truncation_method = "adaptive")``,
``R/utils.R:1156-1195`` at :data:`HTE3_COMMIT`): a cutoff chosen by Brent's method on a
Riesz-type loss, between ``1/n`` and ``0.5``.  That step is part of hte3's estimator, so the
comparison keeps it, and the emulation (:mod:`tests.studies.ep_hte3_pilot`) reproduces it.

**Cells, readings and sizing** are :data:`CELLS`, :func:`readings` and
:func:`design_numbers`.  The margins are fixed here: non-inferiority below
:data:`NONINFERIORITY_MARGIN` (``log 1.25``), similarity inside ``+-log 1.25``, reference validity
at the parent's margin, the targeting witness above :data:`WITNESS_MARGIN`, and the mutation
controls above their rescue margins.  The declaration commit pins :data:`REPLICATES` to
``design_numbers()["replicates"]`` from the committed pilot (R = 100 for every reading cell, 20
for M1 and M2; the gated family passes with probability above 0.9999, and each mutation control
misses with probability below ``1e-12``) and :data:`REPORTED` to its reported cells, none.

**A red reading** (plan 4.1).  A non-inferiority reading that fails is published red in
``comparison.csv`` and on the comparison page; no test asserts a pass.  The row thread
diagnoses it with the curve distance, the emulation and the sieve index matrix.  If the
diagnosis finds a ``cleverly`` defect, a fix row owns it (the only owner the roadmap allows),
and that fix reruns the parent study's affected cells.  If the difference is the two methods
or the two bases, the comparison page records the explanation, and no owner is needed.  A
curve distance above :data:`CURVE_DISTANCE_THRESHOLD` while both ISE readings pass points at a
runner defect that the ISE readings cannot see, and opens a diagnosis.  A mutation control that
does not fail voids the targeting witness.  A comparison artifact is not a registered study's
red cell, and it blocks nothing shipped.

**The emulation's agreement with hte3.**  The plan (4.4) asked the R curve and the numpy
emulation (:mod:`tests.studies.ep_hte3_pilot`) to agree to ``1e-6`` per point.  That cannot
hold: hte3 solves its ridge steps with ``glmnet(lambda = 1e-8)`` at glmnet's default
``thresh = 1e-7``, by coordinate descent on cosine columns that are strongly correlated (up to
101 of them at d = 20), so the two curves agree to the solver's tolerance, not to rounding.
This declaration loosens the plan's tolerance to two scale-free gates per replicate, declared
before any R run: :data:`EMULATION_LOG_ISE_GATE` on ``|log ISE(hte3 R) - log ISE(emulation)|``
and :data:`EMULATION_RMS_GATE` on the root mean square of ``theta_R - theta_emulation`` over
``sqrt(ISE(oracle))`` (:func:`emulation_agreement`).  The largest per-point gap is reported,
not gated.  hte3's truncation cutoffs are compared at :data:`CUTOFF_TOLERANCE`, because that
step is the same Brent search on both sides.  The smoke carries a mutation witness: the
emulation without the truncation must exceed a gate on C6, where the truncation binds.  If the
smoke uses more than :data:`GATE_USE_LIMIT` of a gate, the gate is re-declared before the
declaration commit, never after the run.  In the declared run the agreement is a reading in
``comparison.csv`` (``emulation_agrees``); an exceedance goes to the diagnosis below and does
not block reading the run.

**The gates re-declared after the R smoke, before any declared run.**  The smoke at
``005b1dad`` (committed as ``0112674c``) used 54% of the first log-ISE gate (``1e-3``), 31% of
the first RMS gate (``1e-2``) and exceeded the first cutoff gate (``1e-10``, worst
``1.06e-8``).  Under the rule above all three are re-declared before the declared run, and this
loosens the earlier gates 10 times (log-ISE), 5 times (RMS) and 2000 times (cutoff).  The RMS
and cutoff gates rest on the numerical precision of the two programs; the log-ISE gate is a
materiality bound taken from the margin, not a precision bound.  The smoke's gaps only confirm
the headroom (at most 5.4%, 6.1% and 5.3% of the new gates).  The runner's header comment
still says ``1e-10``; it is left as written because the runner's hash is gated, and this
docstring is the declaration.

- *RMS gate* :data:`EMULATION_RMS_GATE` ``= 0.05``, mostly from glmnet's precision.  glmnet
  stops coordinate descent when no coefficient update lowers the objective by more than
  ``thresh`` (``1e-7``) times the null deviance ``dev0``.  The objective gap at exit is then at
  most ``1/2 g' H^{-1} g``, about ``p thresh dev0 kappa`` for ``p`` columns and ``kappa`` the
  condition number of the weighted Gram matrix ``H``.  With near-orthogonal columns
  (``kappa`` near 1) that is a fitted-value error of RMS about ``sqrt(2 p thresh) * sigma0``,
  with ``sigma0^2 = dev0 / n``; divided by the oracle's root ISE this scale estimate is 0.011
  to 0.025 across the cells (E1 0.011, E5 0.013, C4 0.013, C6 0.015, E8 0.025, where ``sigma0``
  is 3 on the noise-2 law; one smoke-stream draw each), and the gate is twice the largest.
  ``kappa`` on the same draws is 2.3 (E1, E8), 3.0 (C1, C4), 5.3 (C6) and 27 (E5, d = 20), so
  the gate is a scale estimate, not a worst-case bound: at E5 the worst case
  ``sqrt(kappa)`` times the scale is 0.07.  The CRR cells' debiasing is binomial glmnet, whose
  outer reweighting loop has its own convergence rule on top of ``thresh``.  The gate is a
  curve distance of ``0.05^2 = 0.0025``, a quarter of a percent of
  :data:`CURVE_DISTANCE_THRESHOLD`.  The emulation solves exactly, and the runner-local final
  stages are ``glm.fit`` at ``epsilon = 1e-14``, so glmnet's stopping rules are the only solver
  tolerance between the two curves.
- *Log-ISE gate* :data:`EMULATION_LOG_ISE_GATE` ``= 0.01``, a materiality bound.  It bounds the
  shift an emulation difference can put into ``D`` directly, so it is taken from the smallest
  margin, ``log 1.25 = 0.223`` (non-inferiority and similarity): 0.01 is 4.5% of it, too small
  to change what a reading means.  It does not follow from glmnet's precision: a curve error of
  relative size ``r`` moves ``log ISE`` by up to about ``2 r`` to first order, and the smoke's
  C5 replicate 1 already shows half that alignment (log-ISE gap ``5.2e-4`` at RMS ratio
  ``5.0e-4``).
- *Cutoff gate* :data:`CUTOFF_TOLERANCE` ``= 2e-7``, from Brent's precision.  hte3 calls
  ``optim(method = "Brent")`` with no ``control`` (``R/utils.R:1175`` and ``:1184``), which
  calls ``optimize`` with ``tol = reltol = sqrt(.Machine$double.eps)``, about ``1.5e-8``, and the
  emulation ports the same ``Brent_fmin`` at the same ``tol``.  ``Brent_fmin`` stops when the
  minimizer is bracketed to ``t2 = 2 (sqrt(eps) |x| + tol / 3)``, at most about ``2.5e-8`` on
  ``[1/n, 0.5]``.  Two runs whose risk sums differ in the last bits (R's ``mean`` against
  numpy's pairwise sum) can stop up to about ``4 t2 = 1e-7`` apart, provided both converge in
  the same basin: the risk is piecewise smooth with kinks at the propensities, and a run that
  lands in another local minimum is not covered.  The gate is twice that worst case.  A cutoff
  moved by ``2e-7`` moves a weight ``1 / pi`` by at most ``5e-6`` of itself at the smallest
  truncated propensity.

**An agreement exceedance in the declared run** goes to the diagnosis of a red reading, and its
first step is pre-declared here: a replicate whose log-ISE gap exceeds its gate while its RMS
ratio is inside its gate (and, likewise, a replicate over the RMS gate) is refitted in R with
glmnet at ``thresh = 1e-12``.  If the refitted gap falls inside the gates, the exceedance was
solver tolerance and the comparison page says so; if it does not, it is an emulation or runner
fault and the diagnosis continues.

**Cost.**  The committed smoke ran about 2x oversubscribed on the local laptop and projects the
declared run at 73.4 CPU-h (Python 70.1, R 3.3; :func:`recost`).  The 35 to 40 CPU-h expected
for an uncontended local run assumes that oversubscription is the cause of the difference; the
guard is the in-run projection against :data:`LOCAL_BUDGET_CPU_HOURS`, not this figure.

**Lost replicates.**  A replicate whose Python fit or hte3 call fails is counted and published
(``failed`` and ``failed_share`` in ``comparison.csv``, the records in ``failures.csv``), never
refilled from later seeds.  A cell that loses more than :data:`FAILED_SHARE_LIMIT` of its
declared replicates becomes ``reported`` (:func:`readings`).

The driver is ``python -m tests.canonical.hte3_ep.regenerate``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cleverly.effect_curves import _pseudo_outcomes
from tests.studies import ep_cate, ep_cate_properties, ep_crr, ep_crr_properties
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import stream_seed

#: The pinned commits the image builds (``tests/canonical/hte3_ep/Dockerfile``).
HTE3_COMMIT = "eac64673f6fdf75f130275646657f08d1cb198e9"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
IMAGE = "cleverly-hte3:eac64673"

ARTIFACTS = ROOT / "tests" / "canonical" / "hte3_ep"
DOCKERFILE = ARTIFACTS / "Dockerfile"
RUNNER = ARTIFACTS / "hte3_ep_runner.R"
HARNESS = ROOT / "tests" / "canonical" / "study_harness.R"

#: The stream labels this comparison adds under each parent's record.  Neither is a root seed.
PILOT_STREAM = "hte3_pilot"
SMOKE_STREAM = "hte3_smoke"

#: The parents' own streams, which the declared run re-draws.
PROPERTY_STREAM = ep_cate_properties.STREAM
ISE_STREAM = "ise_points"

#: One-sided confidence of every bound, as in the parents.
CONFIDENCE = 0.99

#: The margins (plan 4.3), fixed before any reading.
NONINFERIORITY_MARGIN = math.log(1.25)
SIMILARITY_MARGIN = math.log(1.25)
WITNESS_MARGIN = math.log(1.5)
CURVE_DISTANCE_THRESHOLD = 1.0

#: The sizing rule (plan 4.4): candidates, the per-cell and family targets, the planning floor
#: of the mean and the SD factor.  The factor is the plan's 1.30; the exact upper 95% factor
#: ``sqrt(29 / chi2_{0.05, 29})`` is 1.28, so 1.30 is the conservative reading (review minor 4).
R_CANDIDATES = (100, 150, 200, 250, 300)
TARGET_PASS = 0.90
TARGET_FAMILY = 0.90
PLANNING_MEAN_FLOOR = math.log(1.1)
PLANNING_SD_FACTOR = 1.30

#: Mutation controls: their replicate count and the failure probability that gates each.
MUTATION_REPLICATES = 20
MUTATION_GATE_BELOW = 1e-3

#: The emulation's agreement with hte3 in R, per replicate (see the module docstring): the
#: log-ISE gap, the curve gap in units of the oracle's root ISE, and hte3's truncation cutoffs.
#: Re-declared after the R smoke and before any declared run (module docstring): 10, 5 and
#: 2000 times looser than the first declared values ``1e-3``, ``1e-2`` and ``1e-10``.
EMULATION_LOG_ISE_GATE = 1e-2
EMULATION_RMS_GATE = 5e-2
CUTOFF_TOLERANCE = 2e-7
#: The share of a gate the smoke may use before the gate is re-declared.
GATE_USE_LIMIT = 0.10

#: The share of a cell's declared replicates it may lose before it is reported, not gated.
FAILED_SHARE_LIMIT = 0.05

#: The local budget of the declared run: the smoke stops the local queue when its re-costing
#: projects more CPU-hours than this.  Python seconds against the pilot's are reported only.
LOCAL_BUDGET_CPU_HOURS = 45.0


GATED = "gated"
WITNESS = "witness"
MUTATION = "mutation"


@dataclass(frozen=True)
class Cell:
    """One comparison cell.

    Parameters
    ----------
    label : str
        The cell's name: ``E1`` to ``E8``, ``C1`` to ``C6``, ``M1`` and ``M2``.
    parent : {"ep-cate", "ep-crr"}
        The parent study.
    group : str
        The parent group the cell re-draws.
    learner : str
        The parent's outcome learner: ``ensemble``, ``misspecified`` or ``offset``.
    role : {"gated", "witness", "mutation"}
        A gated reading; a gated reading that also carries the targeting witness; or a
        mutation control that must fail.
    planning_replicates : int
        The plan's worst-case ``R``, which applies only without a pilot (rule step 5).
    validity_margin : float
        The parent's margin for hte3's own oracle comparison (reference validity), and for a
        mutation control the rescue margin it must exceed.
    mutation_of : str
        For a mutation control, the cell whose inputs it reuses with ``pi.hat`` set to 0.5.
    """

    label: str
    parent: str
    group: str
    learner: str
    role: str
    planning_replicates: int
    validity_margin: float
    mutation_of: str = ""

    @property
    def module(self) -> ModuleType:
        """The parent study module."""
        return ep_cate if self.parent == ep_cate.SLUG else ep_crr

    @property
    def scale(self) -> str:
        """``difference`` for the CATE, ``log_ratio`` for the log CRR."""
        return "difference" if self.parent == ep_cate.SLUG else "log_ratio"

    @property
    def parent_cell(self) -> str:
        """The parent's ``oracle_comparison`` cell of the same group and learner."""
        suffix = "ep" if self.learner == "ensemble" else f"ep_{self.learner}"
        return f"{self.group}__{suffix}"


_THM5 = ep_cate.THM5_MARGIN
_RESCUE_CATE = ep_cate.RESCUE_MARGIN
_RESCUE_CRR = ep_crr.RESCUE_MARGIN

CELLS: tuple[Cell, ...] = (
    Cell("E1", "ep-cate", "d3_simple_moderate_4k", "ensemble", GATED, 100, _THM5),
    Cell("E2", "ep-cate", "d3_simple_limited_4k", "ensemble", GATED, 100, _THM5),
    Cell("E3", "ep-cate", "d3_complex_moderate_4k", "ensemble", GATED, 100, _THM5),
    Cell("E4", "ep-cate", "d3_complex_limited_4k", "ensemble", GATED, 100, _THM5),
    Cell("E5", "ep-cate", "d20_simple_4k", "ensemble", GATED, 100, _THM5),
    Cell("E6", "ep-cate", "d20_complex_4k", "ensemble", GATED, 100, _THM5),
    Cell(
        "E7", "ep-cate", "d3_complex_moderate_noise2_4k", "misspecified", WITNESS, 300, _RESCUE_CATE
    ),
    Cell(
        "E8", "ep-cate", "d3_complex_limited_noise2_4k", "misspecified", WITNESS, 300, _RESCUE_CATE
    ),
    Cell("C1", "ep-crr", "d3_simple_moderate_4k", "ensemble", GATED, 100, _THM5),
    Cell("C2", "ep-crr", "d3_simple_limited_4k", "ensemble", GATED, 100, _THM5),
    Cell("C3", "ep-crr", "d3_complex_moderate_4k", "ensemble", GATED, 100, _THM5),
    Cell("C4", "ep-crr", "d3_complex_limited_4k", "ensemble", GATED, 300, _THM5),
    Cell("C5", "ep-crr", "d3_complex_limited_4k", "misspecified", WITNESS, 300, _RESCUE_CRR),
    Cell("C6", "ep-crr", "d3_complex_limited_4k", "offset", WITNESS, 300, _RESCUE_CRR),
    Cell(
        "M2",
        "ep-crr",
        "d3_complex_limited_4k",
        "offset",
        MUTATION,
        MUTATION_REPLICATES,
        _RESCUE_CRR,
        mutation_of="C6",
    ),
    Cell(
        "M1",
        "ep-cate",
        "d3_complex_limited_noise2_4k",
        "misspecified",
        MUTATION,
        MUTATION_REPLICATES,
        _RESCUE_CATE,
        mutation_of="E8",
    ),
)
CELL_BY_LABEL: Mapping[str, Cell] = {cell.label: cell for cell in CELLS}

#: The cells that read a non-inferiority verdict (every cell but the two mutation controls).
READING_CELLS: tuple[Cell, ...] = tuple(cell for cell in CELLS if cell.role != MUTATION)
MUTATION_CELLS: tuple[Cell, ...] = tuple(cell for cell in CELLS if cell.role == MUTATION)

#: ``R`` per cell and the cells declared reported, pinned by the declaration commit to
#: :func:`design_numbers` from the committed pilot (the design test checks the pin).
REPLICATES: Mapping[str, int] = {
    **{cell.label: 100 for cell in CELLS if cell.role != MUTATION},
    "M1": MUTATION_REPLICATES,
    "M2": MUTATION_REPLICATES,
}
REPORTED: frozenset[str] = frozenset()


# ---------------------------------------------------------------- draws and the Python phase


def properties_of(cell: Cell) -> ModuleType:
    """The parent's properties module, whose ``_draw`` made the parent's property draws."""
    return ep_cate_properties if cell.parent == ep_cate.SLUG else ep_crr_properties


def law_of(cell: Cell) -> Any:
    """The parent law of a cell."""
    module = cell.module
    return module.LAW_BY_LABEL[module.GROUP_BY_LABEL[cell.group].law]


def group_n(cell: Cell) -> int:
    return int(cell.module.GROUP_BY_LABEL[cell.group].n)


def max_replicates(cell: Cell) -> int:
    """The replicates the parent group holds: a cell reads a leading prefix of them."""
    return int(cell.module.GROUP_BY_LABEL[cell.group].replicates)


def draw_seed(cell: Cell, replicate: int, stream: str = PROPERTY_STREAM) -> int:
    """The seed of one draw: the parent's property draw, or a pilot or smoke draw."""
    return stream_seed(cell.module.STUDY, stream, cell.group, replicate)


def draw(cell: Cell, replicate: int, stream: str = PROPERTY_STREAM) -> pd.DataFrame:
    """One draw of a cell's parent group on ``stream``."""
    frame, _ = cell.module.draw_from_seed(
        cell.group, group_n(cell), draw_seed(cell, replicate, stream)
    )
    return frame


def truth(cell: Cell, points: np.ndarray) -> np.ndarray:
    """The parent's target at ``points``: the CATE or the log CRR."""
    law = law_of(cell)
    return ep_cate.cate(law, points) if cell.scale == "difference" else ep_crr.log_crr(law, points)


def ise_points(cell: Cell) -> np.ndarray:
    """The parent law's fixed ISE points."""
    return cell.module.ise_points(law_of(cell))


def columns(cell: Cell) -> list[str]:
    return list(law_of(cell).columns)


def sieve_size(cell: Cell) -> int:
    """Cosines per modifier: the parents' schedule at the group's ``n``."""
    return ep_cate.schedule(group_n(cell))


@dataclass
class Export:
    """What the Python phase hands the R phase and the emulation for one replicate.

    Parameters
    ----------
    cell : str
        The cell label.
    replicate : int
        The replicate index.
    w : numpy.ndarray
        The modifiers (every covariate), ``n`` by ``d``.
    a : numpy.ndarray
        The binary treatment.
    y : numpy.ndarray
        The outcome.
    k : int
        Cosines per modifier, so hte3's sieve holds ``k d + 1`` columns.
    mu0 : numpy.ndarray
        The out-of-fold arm regression at ``A = 0``.
    mu1 : numpy.ndarray
        The out-of-fold arm regression at ``A = 1``.
    pi0 : numpy.ndarray
        The propensity of ``A = 0`` after the fit's ``g_bounds``.
    pi1 : numpy.ndarray
        The propensity of ``A = 1`` after the fit's ``g_bounds``.
    theta : numpy.ndarray
        ``cleverly``'s EP curve at the ISE points.
    ise_cleverly : float
        ``cleverly``'s ISE.
    ise_oracle : float
        The oracle learner's ISE (the parent's ``theta_n0``).
    ise_t : float
        The T-learner's ISE on the same nuisances and ``cleverly``'s sieve.
    seconds : float
        CPU seconds of the refit and the curves.
    """

    cell: str
    replicate: int
    w: np.ndarray
    a: np.ndarray
    y: np.ndarray
    k: int
    mu0: np.ndarray
    mu1: np.ndarray
    pi0: np.ndarray
    pi1: np.ndarray
    theta: np.ndarray
    ise_cleverly: float
    ise_oracle: float
    ise_t: float
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def mutated(self, label: str) -> Export:
        """The same inputs with ``pi.hat`` 0.5 for both arms: a mutation control's inputs."""
        half = np.full_like(self.pi1, 0.5)
        return Export(
            label,
            self.replicate,
            self.w,
            self.a,
            self.y,
            self.k,
            self.mu0,
            self.mu1,
            half,
            half.copy(),
            self.theta,
            self.ise_cleverly,
            self.ise_oracle,
            self.ise_t,
            self.seconds,
        )


def nuisances(fit: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(mu0, mu1, pi0, pi1)`` of one EP fit: its out-of-fold arm regressions and bounded ``g``.

    The arm regressions are the ones ``_pseudo_outcomes`` reads, on the outcome's own scale;
    the propensity is truncated into the fit's resolved ``g_bounds``, as the EP fit reads it.
    """
    nuisance_fit = fit.fit.nuisance_fit
    nuisance = nuisance_fit.nuisance
    treated, control = (float(level) for level in nuisance.effect_modification.pair)
    pair = (treated, control)
    if pair != (1.0, 0.0):
        raise ValueError(f"the comparison reads the contrast 1 against 0, not {pair}")
    mu1, mu0, _, _ = _pseudo_outcomes(
        nuisance,
        fit.fit.data,
        pair,
        g_bounds=tuple(nuisance_fit.config.g_bounds),
        missingness_bound=float(nuisance_fit.config.missingness_bound),
    )
    bounded = np.asarray(nuisance.propensity.bounded(tuple(nuisance_fit.config.g_bounds)))
    pi1 = bounded[:, nuisance.propensity.column_for(1.0)]
    pi0 = bounded[:, nuisance.propensity.column_for(0.0)]
    as_array = [np.asarray(values, dtype=float).copy() for values in (mu0, mu1, pi0, pi1)]
    return as_array[0], as_array[1], as_array[2], as_array[3]


def python_phase(cell: Cell, frame: pd.DataFrame, replicate: int) -> Export:
    """Refit ``cleverly``'s EP on one draw as the parent did, and export its nuisances.

    The parent's ``fit_ep`` at ``random_state = replicate`` and its ``curves`` give the EP,
    oracle and T-learner ISEs in the same process, so the pairing does not depend on the
    parent's committed rows.
    """
    module = cell.module
    law = law_of(cell)
    start = time.process_time()
    fit = module.fit_ep(law, frame, cell.learner, replicate, size=ep_cate.schedule)
    measured = module.curves(law, frame, fit)
    names = columns(cell)
    points = pd.DataFrame(ise_points(cell), columns=names)
    theta = np.asarray(fit.predict(points), dtype=float)
    mu0, mu1, pi0, pi1 = nuisances(fit)
    if int(fit.sieve.chosen) != sieve_size(cell):
        raise RuntimeError(f"{cell.label}: the EP fit chose {fit.sieve.chosen} cosines")
    return Export(
        cell=cell.label,
        replicate=replicate,
        w=frame.loc[:, names].to_numpy(dtype=float),
        a=frame["A"].to_numpy(dtype=float),
        y=frame["Y"].to_numpy(dtype=float),
        k=sieve_size(cell),
        mu0=mu0,
        mu1=mu1,
        pi0=pi0,
        pi1=pi1,
        theta=theta,
        ise_cleverly=float(measured["ep"]),
        ise_oracle=float(measured["oracle"]),
        ise_t=float(measured["t_learner"]),
        seconds=time.process_time() - start,
    )


def export_key(key: tuple[str, int, str], drawn: Any) -> Export:
    """The worker of the declared run and the smoke: ``key = (cell, replicate, stream)``."""
    label, replicate, _ = key
    return python_phase(CELL_BY_LABEL[label], drawn, replicate)


def draw_key(key: tuple[str, int, str]) -> pd.DataFrame:
    label, replicate, stream = key
    return draw(CELL_BY_LABEL[label], replicate, stream)


# ---------------------------------------------------------------- the R phase's files


def input_frame(cell: Cell, exports: Sequence[Export]) -> pd.DataFrame:
    """The rows of one inputs archive: one block per replicate, in replicate order.

    ``scenario`` carries the cell label, so the harness's failure records and systematic
    prefix name the cell.
    """
    names = columns(cell)
    blocks = []
    for export in exports:
        block = pd.DataFrame(export.w, columns=names)
        block.insert(0, "scale", cell.scale)
        block.insert(0, "k", export.k)
        block.insert(0, "replicate", export.replicate)
        block.insert(0, "scenario", export.cell)
        block["A"] = export.a.astype(int)
        block["Y"] = export.y
        block["mu0"] = export.mu0
        block["mu1"] = export.mu1
        block["pi0"] = export.pi0
        block["pi1"] = export.pi1
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def points_frame(cell: Cell, label: str | None = None) -> pd.DataFrame:
    """The ISE points of a cell, as the runner reads them."""
    frame = pd.DataFrame(ise_points(cell), columns=columns(cell))
    frame.insert(0, "point", np.arange(1, len(frame) + 1))
    frame.insert(0, "scenario", label or cell.label)
    return frame


#: Every float the R phase reads is written with 17 significant digits, so it parses back to
#: the same double.
FLOAT_FORMAT = "%.17g"


def write_inputs(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format=FLOAT_FORMAT, compression={"method": "gzip"})


def prediction_digest(theta: np.ndarray) -> str:
    """The SHA-256 of one replicate's predictions as little-endian float64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(theta, dtype="<f8").tobytes()).hexdigest()


def cell_digest(digests: Sequence[str]) -> str:
    """A cell's prediction digest: the SHA-256 of its replicates' digests, in replicate order.

    The manifest records one per cell (plan M6); the predictions themselves are not committed.
    """
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


# ---------------------------------------------------------------- the reducer


#: The columns of ``replicates.csv.gz``.
REPLICATE_COLUMNS = (
    "cell",
    "replicate",
    "log_ise_cleverly",
    "log_ise_hte3",
    "log_ise_oracle",
    "log_ise_t",
    "curve_distance",
    "drift",
    "emulation_log_ise_gap",
    "emulation_rms_ratio",
    "emulation_max_gap",
)


@dataclass(frozen=True)
class Committed:
    """The parents' committed per-replicate oracle comparisons, for the drift diagnostic."""

    values: Mapping[tuple[str, str, int], float]

    @classmethod
    def load(cls) -> Committed:
        values: dict[tuple[str, str, int], float] = {}
        for module in (ep_cate, ep_crr):
            rows = pd.read_parquet(
                module.STUDY.artifacts / "property-replicates.parquet",
                columns=["property", "cell", "replicate", "estimate"],
            )
            rows = rows.loc[rows["property"] == module.ORACLE]
            for cell, replicate, estimate in zip(
                rows["cell"], rows["replicate"], rows["estimate"], strict=True
            ):
                values[(module.SLUG, str(cell), int(replicate))] = float(estimate)
        return cls(values)

    def drift(self, cell: Cell, replicate: int, log_ratio: float) -> float:
        """The recomputed oracle comparison minus the committed one (library versions float)."""
        committed = self.values.get((cell.parent, cell.parent_cell, replicate))
        return math.nan if committed is None else log_ratio - committed


def emulation_agreement(
    target: np.ndarray, theta_r: np.ndarray, emulated: np.ndarray, ise_oracle: float
) -> dict[str, float]:
    """The two agreement gates of one replicate, and the largest per-point gap (reported)."""
    theta_r = np.asarray(theta_r, dtype=float)
    emulated = np.asarray(emulated, dtype=float)
    log_r = math.log(float(np.mean((theta_r - target) ** 2)))
    log_e = math.log(float(np.mean((emulated - target) ** 2)))
    gap = theta_r - emulated
    return {
        "log_ise_gap": abs(log_r - log_e),
        "rms_ratio": math.sqrt(float(np.mean(gap**2))) / math.sqrt(ise_oracle),
        "max_gap": float(np.max(np.abs(gap))),
    }


def agrees(log_ise_gap: float, rms_ratio: float) -> bool:
    """Both gates hold."""
    return bool(log_ise_gap <= EMULATION_LOG_ISE_GATE and rms_ratio <= EMULATION_RMS_GATE)


def reduce(
    cell: Cell,
    export: Export,
    theta_hte3: np.ndarray,
    *,
    committed: Committed | None = None,
    emulated: np.ndarray | None = None,
) -> dict[str, Any]:
    """One row of ``replicates.csv.gz`` from one replicate's two curves."""
    target = truth(cell, ise_points(cell))
    hte3_ise = float(np.mean((np.asarray(theta_hte3) - target) ** 2))
    log_cleverly = math.log(export.ise_cleverly)
    log_oracle = math.log(export.ise_oracle)
    distance = float(np.mean((export.theta - np.asarray(theta_hte3)) ** 2)) / export.ise_oracle
    drift = math.nan
    if committed is not None and cell.role != MUTATION:
        drift = committed.drift(cell, export.replicate, log_cleverly - log_oracle)
    agreement = dict.fromkeys(("log_ise_gap", "rms_ratio", "max_gap"), math.nan)
    if emulated is not None:
        agreement = emulation_agreement(target, theta_hte3, emulated, export.ise_oracle)
    return {
        "cell": cell.label,
        "replicate": export.replicate,
        "log_ise_cleverly": log_cleverly,
        "log_ise_hte3": math.log(hte3_ise),
        "log_ise_oracle": log_oracle,
        "log_ise_t": math.log(export.ise_t),
        "curve_distance": distance,
        "drift": drift,
        "emulation_log_ise_gap": agreement["log_ise_gap"],
        "emulation_rms_ratio": agreement["rms_ratio"],
        "emulation_max_gap": agreement["max_gap"],
    }


# ---------------------------------------------------------------- the readings


def _t(replicates: int, confidence: float = CONFIDENCE) -> float:
    return float(stats.t.ppf(confidence, replicates - 1))


def bounds(values: np.ndarray) -> tuple[float, float, float, float]:
    """``(mean, sd, lower, upper)``: one-sided 99% Student bounds of a mean."""
    values = np.asarray(values, dtype=float)
    m = values.size
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = _t(m) * sd / math.sqrt(m)
    return mean, sd, mean - half, mean + half


def roles(reported: frozenset[str] = REPORTED) -> dict[str, str]:
    """Each cell's verdict role: ``gated``, ``reported`` or ``mutation``."""
    out = {}
    for cell in CELLS:
        if cell.role == MUTATION:
            out[cell.label] = "reported" if cell.label in reported else MUTATION
        else:
            out[cell.label] = "reported" if cell.label in reported else GATED
    return out


READING_COLUMNS = (
    "cell",
    "role",
    "declared",
    "replicates",
    "failed",
    "failed_share",
    "d_mean",
    "d_sd",
    "d_upper",
    "noninferior",
    "similarity_lower",
    "similarity_upper",
    "similar",
    "hte3_oracle_upper",
    "cleverly_oracle_upper",
    "reference_valid",
    "witness_lower",
    "witness",
    "mutation_lower",
    "mutation_failed",
    "curve_distance_mean",
    "curve_distance_flag",
    "drift_max",
    "emulation_log_ise_max",
    "emulation_rms_max",
    "emulation_max_gap",
    "emulation_agrees",
)


def readings(
    rows: pd.DataFrame,
    reported: frozenset[str] = REPORTED,
    declared: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Every reading of plan 4.3, recomputed from ``replicates.csv.gz`` rows.

    A failed non-inferiority reading is published as ``False``; nothing here or in the tests
    asserts that one passes.  ``declared`` is the replicates each cell was run at
    (:data:`REPLICATES` by default): a replicate absent from ``rows`` was lost, and a cell that
    lost more than :data:`FAILED_SHARE_LIMIT` of them is ``reported``.  Without a declared count
    a cell's rows are its declared replicates.  The emulation's agreement is a reading too, and
    an exceedance blocks nothing.
    """
    role_of = roles(reported)
    declared = REPLICATES if declared is None else declared
    out = []
    for cell in CELLS:
        frame = rows.loc[rows["cell"] == cell.label]
        planned = int(declared.get(cell.label, len(frame)))
        if frame.empty and planned == 0:
            continue
        failed = planned - len(frame)
        if failed < 0:
            raise ValueError(f"{cell.label}: {len(frame)} rows for {planned} declared replicates")
        share = failed / planned
        role = "reported" if share > FAILED_SHARE_LIMIT else role_of[cell.label]
        row: dict[str, Any] = dict.fromkeys(READING_COLUMNS, math.nan)
        row.update(
            cell=cell.label,
            role=role,
            declared=planned,
            replicates=len(frame),
            failed=failed,
            failed_share=share,
        )
        if len(frame) < 2:
            out.append(row)
            continue
        log_gap = frame["emulation_log_ise_gap"].to_numpy(dtype=float)
        rms = frame["emulation_rms_ratio"].to_numpy(dtype=float)
        if np.isfinite(log_gap).any():
            row.update(
                emulation_log_ise_max=float(np.nanmax(log_gap)),
                emulation_rms_max=float(np.nanmax(rms)),
                emulation_max_gap=float(np.nanmax(frame["emulation_max_gap"])),
                emulation_agrees=bool(
                    np.isfinite(log_gap).all()
                    and agrees(float(np.max(log_gap)), float(np.max(rms)))
                ),
            )
        hte3_ratio = (frame["log_ise_hte3"] - frame["log_ise_oracle"]).to_numpy()
        cleverly_ratio = (frame["log_ise_cleverly"] - frame["log_ise_oracle"]).to_numpy()
        if cell.role == MUTATION:
            _, _, lower, _ = bounds(hte3_ratio)
            row["mutation_lower"] = lower
            row["mutation_failed"] = bool(lower > cell.validity_margin)
            out.append(row)
            continue
        d = (frame["log_ise_cleverly"] - frame["log_ise_hte3"]).to_numpy()
        mean, sd, _, upper = bounds(d)
        half = _t(len(d)) * sd / math.sqrt(len(d))
        hte3_upper = bounds(hte3_ratio)[3]
        cleverly_upper = bounds(cleverly_ratio)[3]
        distance = float(frame["curve_distance"].mean())
        valid = bool(hte3_upper < cell.validity_margin)
        row.update(
            d_mean=mean,
            d_sd=sd,
            d_upper=upper,
            noninferior=bool(upper < NONINFERIORITY_MARGIN),
            similarity_lower=mean - half,
            similarity_upper=mean + half,
            similar=bool(mean - half > -SIMILARITY_MARGIN and mean + half < SIMILARITY_MARGIN),
            hte3_oracle_upper=hte3_upper,
            cleverly_oracle_upper=cleverly_upper,
            reference_valid=valid,
            curve_distance_mean=distance,
            curve_distance_flag=bool(
                distance > CURVE_DISTANCE_THRESHOLD
                and valid
                and cleverly_upper < cell.validity_margin
            ),
            drift_max=float(np.nanmax(np.abs(frame["drift"])))
            if frame["drift"].notna().any()
            else math.nan,
        )
        if cell.role == WITNESS:
            witness = (frame["log_ise_t"] - frame["log_ise_hte3"]).to_numpy()
            lower = bounds(witness)[2]
            row.update(witness_lower=lower, witness=bool(lower > WITNESS_MARGIN))
        out.append(row)
    return pd.DataFrame(out, columns=list(READING_COLUMNS))


# ---------------------------------------------------------------- the sizing rule


PILOT_CSV = ROOT / "tests" / "studies" / "ep_hte3_pilot.csv"


def pilot_rows(path: Path = PILOT_CSV) -> dict[tuple[str, str], dict[str, float]] | None:
    """``(cell, statistic) -> {mean, sd, replicates}`` from the committed pilot, or ``None``."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return {
        (str(row["cell"]), str(row["statistic"])): {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "replicates": int(row["replicates"]),
        }
        for _, row in frame.iterrows()
    }


def pass_probability(mean: float, sd: float, replicates: int) -> float:
    """P(the upper bound of mean ``D`` lies below ``log 1.25``) at a true mean and SD (plan 4.4)."""
    return ep_cate.gate_pass(mean, sd, replicates, NONINFERIORITY_MARGIN, below=True)


def mutation_miss(mean: float, sd: float, replicates: int, margin: float) -> float:
    """P(a mutation control's lower bound stays at or below its margin): it would not fail."""
    return 1.0 - ep_cate.gate_pass(mean, sd, replicates, margin, below=False)


def _committed_m2() -> dict[str, float]:
    """The committed unit-weight mutation of ``ep-crr``'s offset pilot (1.72, SD 0.34).

    Only the fallback reads it; with a committed pilot, M2 is sized from its own pilot row.
    """
    row = ep_crr.pilot_rows()[("d3_complex_limited_4k", "ep_offset_unit_weight")]
    return {"mean": row["mean"], "sd": row["sd"], "replicates": row["replicates"]}


def size_cells(
    planning: Mapping[str, tuple[float, float]], caps: Mapping[str, int]
) -> tuple[dict[str, int], set[str]]:
    """Rule steps 2 and 3: per-cell ``R`` and the cells no candidate sizes.

    Each cell takes the smallest candidate, up to the replicates its parent group holds
    (``caps``), with pass probability at least :data:`TARGET_PASS`; a cell with none is
    reported.  While the family product of the gated cells is below :data:`TARGET_FAMILY`, the
    gated cell with the lowest pass probability moves to its next candidate; a cell already at
    its largest candidate that is still the lowest is reported.
    """

    def candidates(label: str) -> tuple[int, ...]:
        return tuple(r for r in R_CANDIDATES if r <= caps[label])

    sized: dict[str, int] = {}
    reported: set[str] = set()
    for label, (mean, sd) in planning.items():
        fit = [r for r in candidates(label) if pass_probability(mean, sd, r) >= TARGET_PASS]
        if fit:
            sized[label] = fit[0]
        else:
            reported.add(label)
    while sized:
        passes = {label: pass_probability(*planning[label], r) for label, r in sized.items()}
        if float(np.prod(list(passes.values()))) >= TARGET_FAMILY:
            break
        lowest = min(passes, key=lambda label: passes[label])
        allowed = candidates(lowest)
        position = allowed.index(sized[lowest])
        if position + 1 < len(allowed):
            sized[lowest] = allowed[position + 1]
        else:
            del sized[lowest]
            reported.add(lowest)
    return sized, reported


def design_numbers(path: Path = PILOT_CSV) -> dict[str, Any]:
    """Every computed number the declaration fixes before the run (plan 4.4).

    Returns
    -------
    dict
        ``source``: ``"pilot"`` or ``"fallback"`` (no committed pilot, rule step 5);
        ``replicates``: ``R`` per cell, the mutation controls included; ``reported``: the cells
        declared reported; ``cells``: per reading cell its planning mean and SD and its pass
        probability at its ``R``; ``family``: the product over the gated cells; ``mutations``:
        each control's probability of not failing at 20 replicates; ``cost``: CPU-hours of the
        Python phase from the pilot's seconds.
    """
    pilot = pilot_rows(path)
    cells: dict[str, dict[str, float]] = {}
    if pilot is None:
        m2 = _committed_m2()
        mutations = {
            "M2": mutation_miss(m2["mean"], m2["sd"], MUTATION_REPLICATES, _RESCUE_CRR),
        }
        replicates = {cell.label: cell.planning_replicates for cell in CELLS}
        reported = {"E7", "E8", "C6", "M1"}
        for cell in READING_CELLS:
            cells[cell.label] = {
                "planning_mean": math.nan,
                "planning_sd": math.nan,
                "pass": math.nan,
            }
        return {
            "source": "fallback",
            "replicates": replicates,
            "reported": frozenset(reported),
            "cells": cells,
            "family": math.nan,
            "mutations": {**mutations, "M1": math.nan},
            "cost": {"cpu_hours": math.nan},
        }
    planning = {
        cell.label: (
            max(pilot[(cell.label, "d")]["mean"], PLANNING_MEAN_FLOOR),
            pilot[(cell.label, "d")]["sd"] * PLANNING_SD_FACTOR,
        )
        for cell in READING_CELLS
    }
    caps = {cell.label: max_replicates(cell) for cell in READING_CELLS}
    sized, reported = size_cells(planning, caps)
    for cell in READING_CELLS:
        mean, sd = planning[cell.label]
        r = sized.get(cell.label, caps[cell.label])
        cells[cell.label] = {
            "planning_mean": mean,
            "planning_sd": sd,
            "pass": pass_probability(mean, sd, r),
        }
    # Each mutation control from its own pilot row, the SD inflated as the reading cells' are.
    mutations = {}
    for cell in MUTATION_CELLS:
        row = pilot[(cell.label, "log_ratio_hte3")]
        mutations[cell.label] = mutation_miss(
            row["mean"], row["sd"] * PLANNING_SD_FACTOR, MUTATION_REPLICATES, cell.validity_margin
        )
        if mutations[cell.label] >= MUTATION_GATE_BELOW:
            reported.add(cell.label)
    replicates = {label: sized.get(label, caps[label]) for label in planning}
    replicates.update({cell.label: MUTATION_REPLICATES for cell in MUTATION_CELLS})
    family = float(np.prod([cells[label]["pass"] for label in sized]))
    seconds = 0.0
    for cell in READING_CELLS:
        key = (cell.label, "seconds")
        if key in pilot:
            seconds += pilot[key]["mean"] * replicates[cell.label]
    return {
        "source": "pilot",
        "replicates": replicates,
        "reported": frozenset(reported),
        "cells": cells,
        "family": family,
        "mutations": mutations,
        "cost": {"cpu_hours": seconds / 3600.0},
    }


#: The files a declared run writes into ``tests/canonical/hte3_ep/``.
RUN_FILES = ("replicates.csv.gz", "comparison.csv", "manifest.json", "run.log")

#: The files a smoke writes, which the local runner commits under ``tests/canonical/hte3_ep/smoke``.
SMOKE_FILES = (
    "agreement.csv",
    "points.csv",
    "sieve-index.csv",
    "timings.csv",
    "recost.csv",
    "session.txt",
)
SMOKE_DIR = ARTIFACTS / "smoke"

#: Points per replicate that ``smoke/points.csv`` keeps (the agreement reads all 20,000).
SMOKE_POINTS = 200


def recost(timings: pd.DataFrame, path: Path = PILOT_CSV) -> pd.DataFrame:
    """The declared run's CPU-hours projected from the smoke's seconds, per cell.

    ``cpu_hours`` is the cell's Python and R seconds per replicate at the smoke times the
    replicates ``design_numbers`` gives; a mutation control reuses its parent's Python fits, so
    only its R seconds count.  Their sum is checked against :data:`LOCAL_BUDGET_CPU_HOURS`
    (:func:`projected_cpu_hours`).  ``ratio`` (Python seconds against the pilot's) is reported
    only: the smoke may run on a slower or busier machine than the pilot did.
    """
    pilot = pilot_rows(path) or {}
    replicates = design_numbers(path)["replicates"]
    means = timings.groupby("cell")[["python_seconds", "r_seconds"]].mean()
    out = []
    for cell in CELLS:
        if cell.label not in means.index:
            continue
        python, r = (float(value) for value in means.loc[cell.label])
        if cell.role == MUTATION:
            python = 0.0
        piloted = pilot.get((cell.label, "seconds"), {}).get("mean", math.nan)
        out.append(
            {
                "cell": cell.label,
                "python_seconds": python,
                "pilot_seconds": piloted,
                "ratio": math.nan if math.isnan(piloted) else python / piloted,
                "r_seconds": r,
                "cpu_hours": (python + r) * replicates[cell.label] / 3600.0,
            }
        )
    return pd.DataFrame(out)


def projected_cpu_hours(recosted: pd.DataFrame) -> float:
    """The declared run's projected CPU-hours: the sum of :func:`recost`'s cells."""
    return float(recosted["cpu_hours"].sum())
