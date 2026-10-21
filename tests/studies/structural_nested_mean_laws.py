r"""The laws of the registered study ``structural-nested-mean`` (X44), with exact truths.

Every law is a finite table: each variable has a conditional distribution over a few values, and
the outcome is a function of the path plus a symmetric three-point noise.  Enumerating the paths
gives the exact probability of each, so a truth is a finite sum
(:mod:`tests.discrete_law_structural`, an independent g-formula that reads nothing from the
estimator), and a sample is a multinomial draw of rows.  The module imports nothing from
``cleverly``.

======== ===========================================================================================
setup    structure
======== ===========================================================================================
``S2``   two binary nodes, ``phi = 1``; blips 1.28 and 0.5
``S3``   three binary nodes with an :math:`A_t \to A_j` path; blips 1.15, 0.95, 0.6
``K``    two nodes of three levels; the effects of the levels are not monotone
``C``    two doses on the support :math:`\{-1, 0, 1\}`; every conditional mean is linear in the
         history, so main-terms linear nuisances are exact; blips 0.97 and 0.6 per unit dose
``H``    two binary nodes; :math:`\phi_1 = (1, W)`, :math:`\phi_2 = (1, W, A_1)`
``P``    the node-2 blip :math:`A_2 (0.5 + 1.5 A_1)`, declared with :math:`\phi_2 = (1, A_1)`
``M``    the second treatment and the outcome read the first covariate
``L``    the node-2 blip :math:`A_2 (0.5 + 0.8 S_2)`, declared with :math:`\phi_2 = (1, S_2)`
``N``    two binary nodes on a nine-point baseline; the outcome mean, the covariate and both
         treatment probabilities are nonlinear in the baseline; ``phi = 1``
``S2W``  ``S2`` with the known weight :math:`w = 0.5 + 1.5 W`; the target is the law with the
         baseline reweighted by :math:`w`
``S2C``  ``S2`` in clusters of ten that share the baseline and an outcome intercept
``LU``   law ``L`` with the node-2 blip declared as :math:`\phi_2 = 1`: the negative control
``VB``   law ``P`` with a baseline-only :math:`\phi_2`: the second negative control
``SP``   the model-check law with the treatment logits multiplied by three: weak positivity;
         blips :math:`A_1 (1.15 + 0.47 W)` and :math:`A_2 (0.5 + 0.4 W + 0.6 A_1)`
======== ===========================================================================================

**Correct models.**  On each positive setup the structural model holds: a blip is
:math:`c(a)^\top \Psi \phi(h)` for every history, and
:func:`tests.discrete_law_structural.structural_blip_errors` verifies it by the g-formula (the
design test does).  The outcome means and the treatment probabilities are linear probability
models of binary parents.  A main-terms or low-order-interaction ``GLM`` is therefore exact on
the binary setups, and the design test checks the declared learners at population weights.
``N`` is the exception by design: its nuisances are nonlinear in the baseline.

**Truths.**  A blip coefficient is declared by hand (``PSI``) and checked against the g-formula.
``ey_reference`` is the mean with every treatment at its reference, a regimen level is the
g-formula mean under the static regimen, and a contrast is a difference of two levels.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tests import discrete_law_structural as finite
from tests.discrete_law_structural import Law, Var, bern

#: The known observation weight of ``S2W``: ``w = 0.5 + 1.5 W``.  The weights are 0.5 and 2.
WEIGHT_INTERCEPT = 0.5
WEIGHT_SLOPE = 1.5
#: The cluster size of ``S2C``, the standard deviation of its outcome intercept, and the
#: probability of its shared baseline.
CLUSTER_SIZE = 10
CLUSTER_OUTCOME_SD = 1.0
#: The treatment-logit multiplier of ``SP``.
SPARSE_SCALE = 3.0

Path = dict[str, float]


def expit(x: Any) -> Any:
    """The logistic function."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ---------------------------------------------------------------------------- finite laws


def _var(
    name: str, role: str, node: int, law: Callable[[Path], Any], reference: float = 0.0
) -> Var:
    return Var(name, role, node, law, reference)


def _with_noise(name: str, variables: tuple[Var, ...], mean: Callable[[Path], float]) -> Law:
    """A law whose outcome is ``mean`` of the path plus the symmetric noise ``E``."""
    noise = Var(
        "E",
        "noise",
        0,
        lambda p: [(-finite.NOISE, 0.25), (0.0, 0.5), (finite.NOISE, 0.25)],
    )
    return Law(name, (*variables, noise), lambda p: mean(p) + p["E"])


def law_c_linear() -> Law:
    """Two doses on :math:`\\{-1, 0, 1\\}` whose conditional means are linear in the history.

    :math:`E[A_1 \\mid S_1, W] = 0.1 + 0.1 S_1 + 0.05 W` and
    :math:`E[A_2 \\mid S_2, A_1] = 0.1 + 0.1 S_2 + 0.1 A_1`, so a main-terms Gaussian regression
    is exact for every nuisance.  The blips are 0.97 and 0.6 per unit dose.
    """

    def dose(plus: float, minus: float) -> list[tuple[float, float]]:
        return [(-1.0, minus), (0.0, 1.0 - plus - minus), (1.0, plus)]

    variables = (
        _var("W", "baseline", 0, lambda p: bern(0.4)),
        _var("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _var(
            "A1",
            "treatment",
            1,
            lambda p: dose(0.35 + 0.10 * p["S1"], 0.25 - 0.05 * p["W"]),
        ),
        _var("S2", "covariate", 2, lambda p: bern(0.3 + 0.3 * p["S1"] + 0.1 * p["A1"])),
        _var(
            "A2",
            "treatment",
            2,
            lambda p: dose(0.35 + 0.10 * p["S2"] + 0.05 * p["A1"], 0.25 - 0.05 * p["A1"]),
        ),
    )

    def mean(p: Path) -> float:
        return 0.9 * p["A1"] + 0.6 * p["A2"] + 0.7 * p["S2"] + 0.3 * p["W"] + 0.2 * p["S1"]

    law = _with_noise("C", variables, mean)
    return Law(law.name, law.variables, law.outcome, doses=("A1", "A2"))


#: The baseline grid of law ``N``: nine equally likely points on [-1, 1].
N_GRID = tuple(float(x) for x in np.linspace(-1.0, 1.0, 9))


def law_n() -> Law:
    """Two binary nodes on a nine-point baseline; every nuisance is nonlinear in it.

    The outcome mean is :math:`A_1 + 0.5 A_2 + 0.7 S_2 + g(W, S_1)` with
    :math:`g = 1.2 \\sin(\\pi W) + 0.8 W^2 + 0.5 S_1 \\cos(\\pi W)`.  The second covariate
    moves with :math:`A_1` by a constant 0.2, so the blips stay constant: 1.14 and 0.5.
    """

    def p_s1(p: Path) -> float:
        return float(expit(1.4 * np.sin(np.pi * p["W"])))

    def p_a1(p: Path) -> float:
        return float(expit(0.2 + 1.3 * np.cos(np.pi * p["W"]) + 0.9 * (p["S1"] - 0.5)))

    def p_s2(p: Path) -> float:
        base = 0.4 + 0.28 * np.sin(np.pi * p["W"]) * (2.0 * p["S1"] - 1.0)
        return float(base + 0.2 * p["A1"])

    def p_a2(p: Path) -> float:
        return float(
            expit(
                -0.3
                + 1.1 * np.sin(2.0 * np.pi * p["W"])
                + 0.8 * (p["S2"] - 0.5)
                + 0.5 * (p["A1"] - 0.5)
            )
        )

    variables = (
        _var("W", "baseline", 0, lambda p: [(x, 1.0 / len(N_GRID)) for x in N_GRID]),
        _var("S1", "covariate", 1, lambda p: bern(p_s1(p))),
        _var("A1", "treatment", 1, lambda p: bern(p_a1(p))),
        _var("S2", "covariate", 2, lambda p: bern(p_s2(p))),
        _var("A2", "treatment", 2, lambda p: bern(p_a2(p))),
    )

    def mean(p: Path) -> float:
        wiggle = (
            1.2 * np.sin(np.pi * p["W"])
            + 0.8 * p["W"] ** 2
            + 0.5 * p["S1"] * np.cos(np.pi * p["W"])
        )
        return p["A1"] + 0.5 * p["A2"] + 0.7 * p["S2"] + float(wiggle)

    return _with_noise("N", variables, mean)


def law_s2_reweighted() -> Law:
    """``S2`` with the baseline reweighted by the known weight: the target of ``S2W``."""
    q = 0.4 * weight(1.0) / (0.4 * weight(1.0) + 0.6 * weight(0.0))
    base = finite.law_s2()
    _, *rest = base.variables
    tilted = Var("W", "baseline", 0, lambda p: bern(q))
    return Law("S2W", (tilted, *rest), base.outcome)


def law_s2_cluster() -> Law:
    """``S2`` for the cluster design: the same marginal law, so the same truths."""
    base = finite.law_s2()
    return Law("S2C", base.variables, base.outcome)


def law_sparse() -> Law:
    """The model-check law of the size tests with the treatment logits multiplied by three.

    Binary :math:`W, A_1, L_2, A_2`; the blips are :math:`A_1 (1 + 0.5 W)` and
    :math:`A_2 (0.5 + 0.4 W + 0.6 A_1)`.  The treatment probabilities reach 0.05 and 0.94, so
    some history cells hold few rows at :math:`n = 500`.
    """
    k = SPARSE_SCALE
    variables = (
        _var("W", "baseline", 0, lambda p: bern(0.5)),
        _var("A1", "treatment", 1, lambda p: bern(float(expit(k * (-0.3 + 0.8 * p["W"]))))),
        _var(
            "L2", "covariate", 2, lambda p: bern(float(expit(-0.2 + 0.7 * p["W"] + 0.9 * p["A1"])))
        ),
        _var(
            "A2",
            "treatment",
            2,
            lambda p: bern(float(expit(k * (-0.4 + 0.8 * p["L2"] + 0.5 * p["A1"] - 0.6 * p["W"])))),
        ),
    )

    def mean(p: Path) -> float:
        return (
            p["A1"] * (1.0 + 0.5 * p["W"])
            + p["A2"] * (0.5 + 0.4 * p["W"] + 0.6 * p["A1"])
            + 0.7 * p["L2"]
            + 0.3 * p["W"]
        )

    return _with_noise("SP", variables, mean)


def weight(w: Any) -> Any:
    """The known observation weight :math:`0.5 + 1.5 W` of ``S2W``."""
    return WEIGHT_INTERCEPT + WEIGHT_SLOPE * np.asarray(w, dtype=float)


# ------------------------------------------------------------------------------ setups


@dataclass(frozen=True)
class Setup:
    """One law, the blip features declared on it, the regimens, and its declared coefficients.

    Parameters
    ----------
    name : str
        The setup's name.
    law : Law
        The finite law the sample is drawn from.
    target : Law
        The finite law the truths are computed on.  It differs from ``law`` only for a weighted
        setup, whose target is the reweighted population.
    features : dict or None
        The ``blip_features`` declaration.
    regimens : dict
        Static regimens, by label.
    reference : str
        The regimen every contrast is taken against.
    psi : dict
        The declared coefficient of each blip, by reported name.
    coefficients : dict
        The declared matrix :math:`\\Psi_t` per node, shape ``(p_t, d_t)`` as
        ``result.blip(node)`` returns it, for the verification against the g-formula.
    estimands : tuple of str
        The names the setup reports and the study scores.
    cluster : bool
        Whether the sample is clustered.
    weighted : bool
        Whether the draw carries the known weight column ``w``.
    nuisance : str
        ``"glm"`` for the declared GLM nuisances, ``"booster"`` for the adaptive cell.
    """

    name: str
    law: Law
    target: Law
    features: dict[int, list[str]] | None
    regimens: dict[str, list[float]]
    reference: str
    psi: dict[str, float]
    coefficients: dict[int, list[list[float]]]
    estimands: tuple[str, ...] = ()
    cluster: bool = False
    weighted: bool = False
    nuisance: str = "glm"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def doses(self) -> list[str]:
        """The treatment columns that hold a dose."""
        return list(self.law.doses)


def _blip(treatment: str, level: str, feature: str = "intercept") -> str:
    return f"blip[{treatment}:{level}:{feature}]"


def _pair(label: str, reference: str) -> str:
    return f"ate_regimen[{label} vs {reference}]"


def _level(label: str) -> str:
    return f"ey_regimen[{label}]"


BINARY_REGIMENS = {"r00": [0.0, 0.0], "r11": [1.0, 1.0], "r10": [1.0, 0.0], "r01": [0.0, 1.0]}
TRIPLE_REGIMENS = {"r000": [0.0, 0.0, 0.0], "r111": [1.0, 1.0, 1.0], "r101": [1.0, 0.0, 1.0]}
THREE_LEVEL_REGIMENS = {"r00": [0.0, 0.0], "r12": [1.0, 2.0], "r21": [2.0, 1.0]}
DOSE_REGIMENS = {"dose0": [0.0, 0.0], "dose_mid": [0.5, 0.25], "dose_high": [1.0, -0.5]}

#: The node-2 slope of ``L`` on its second covariate.
L_SLOPE = finite.L_SLOPE


def _setup(
    name: str,
    law: Law,
    regimens: dict[str, list[float]],
    reference: str,
    psi: dict[str, float],
    coefficients: dict[int, list[list[float]]],
    estimands: Sequence[str],
    features: dict[int, list[str]] | None = None,
    **options: Any,
) -> Setup:
    target = options.pop("target", law)
    return Setup(
        name=name,
        law=law,
        target=target,
        features=features,
        regimens=regimens,
        reference=reference,
        psi=psi,
        coefficients=coefficients,
        estimands=tuple(estimands),
        **options,
    )


def _build() -> dict[str, Setup]:
    out: dict[str, Setup] = {}

    s2 = finite.law_s2()
    out["S2"] = _setup(
        "S2",
        s2,
        BINARY_REGIMENS,
        "r00",
        {_blip("A1", "1.0"): 1.28, _blip("A2", "1.0"): 0.5},
        {1: [[1.28]], 2: [[0.5]]},
        (
            _blip("A1", "1.0"),
            _blip("A2", "1.0"),
            "ey_reference",
            _level("r11"),
            _pair("r11", "r00"),
            # With ``phi = 1`` the contrast of ``r10`` is the first blip and that of ``r01`` is
            # the second, so the level of ``r10`` is the distinct third regimen quantity.
            _level("r10"),
        ),
    )

    s3 = finite.law_s3()
    out["S3"] = _setup(
        "S3",
        s3,
        TRIPLE_REGIMENS,
        "r000",
        {_blip("A1", "1.0"): 1.15, _blip("A2", "1.0"): 0.95, _blip("A3", "1.0"): 0.6},
        {1: [[1.15]], 2: [[0.95]], 3: [[0.6]]},
        (
            _blip("A1", "1.0"),
            _blip("A2", "1.0"),
            _blip("A3", "1.0"),
            "ey_reference",
            _pair("r111", "r000"),
            _pair("r101", "r000"),
        ),
    )

    k = finite.law_k()
    shift = finite.K_SHIFT
    e1, e2 = finite.K_EFFECT_1, finite.K_EFFECT_2
    out["K"] = _setup(
        "K",
        k,
        THREE_LEVEL_REGIMENS,
        "r00",
        {
            _blip("A1", "1.0"): e1[1] + 0.7 * shift[1],
            _blip("A1", "2.0"): e1[2] + 0.7 * shift[2],
            _blip("A2", "1.0"): e2[1],
            _blip("A2", "2.0"): e2[2],
        },
        {
            1: [[e1[1] + 0.7 * shift[1], e1[2] + 0.7 * shift[2]]],
            2: [[e2[1], e2[2]]],
        },
        (
            _blip("A1", "1.0"),
            _blip("A1", "2.0"),
            _blip("A2", "1.0"),
            _blip("A2", "2.0"),
            "ey_reference",
            _level("r12"),
            _pair("r12", "r00"),
            _pair("r21", "r00"),
        ),
    )

    c = law_c_linear()
    out["C"] = _setup(
        "C",
        c,
        DOSE_REGIMENS,
        "dose0",
        {_blip("A1", "dose"): 0.97, _blip("A2", "dose"): 0.6},
        {1: [[0.97]], 2: [[0.6]]},
        (
            _blip("A1", "dose"),
            _blip("A2", "dose"),
            "ey_reference",
            _level("dose_mid"),
            _pair("dose_mid", "dose0"),
            _pair("dose_high", "dose0"),
        ),
    )

    h = finite.law_h()
    out["H"] = _setup(
        "H",
        h,
        BINARY_REGIMENS,
        "r00",
        {
            _blip("A1", "1.0", "intercept"): 1.14,
            _blip("A1", "1.0", "W"): 0.5,
            _blip("A2", "1.0", "intercept"): 0.5,
            _blip("A2", "1.0", "W"): 0.4,
            _blip("A2", "1.0", "A1"): 0.6,
        },
        {1: [[1.14], [0.5]], 2: [[0.5], [0.4], [0.6]]},
        (
            _blip("A1", "1.0", "intercept"),
            _blip("A1", "1.0", "W"),
            _blip("A2", "1.0", "intercept"),
            _blip("A2", "1.0", "W"),
            _blip("A2", "1.0", "A1"),
            "ey_reference",
            _level("r11"),
            _pair("r11", "r00"),
            _pair("r10", "r00"),
        ),
        features={1: ["W"], 2: ["W", "A1"]},
    )

    p = finite.law_p()
    out["P"] = _setup(
        "P",
        p,
        BINARY_REGIMENS,
        "r00",
        {
            _blip("A1", "1.0"): 1.28,
            _blip("A2", "1.0", "intercept"): 0.5,
            _blip("A2", "1.0", "A1"): 1.5,
        },
        {1: [[1.28]], 2: [[0.5], [1.5]]},
        (
            _blip("A1", "1.0"),
            _blip("A2", "1.0", "intercept"),
            _blip("A2", "1.0", "A1"),
            "ey_reference",
            _pair("r11", "r00"),
            # The contrast of ``r01`` is the second blip at ``A1 = 0``, so the level is read.
            _level("r01"),
        ),
        features={2: ["A1"]},
    )

    m = finite.law_m()
    out["M"] = _setup(
        "M",
        m,
        BINARY_REGIMENS,
        "r00",
        {_blip("A1", "1.0"): 1.28, _blip("A2", "1.0"): 0.5},
        {1: [[1.28]], 2: [[0.5]]},
        (
            _blip("A1", "1.0"),
            _blip("A2", "1.0"),
            "ey_reference",
            _pair("r11", "r00"),
        ),
    )

    # The feature ``S2`` reads a time-varying covariate, so the fit reports the blips and
    # ``ey_reference`` and declares no static regimen.
    law_l = finite.law_l()
    out["L"] = _setup(
        "L",
        law_l,
        {},
        "",
        {
            _blip("A1", "1.0"): 1.28,
            _blip("A2", "1.0", "intercept"): 0.5,
            _blip("A2", "1.0", "S2"): L_SLOPE,
        },
        {1: [[1.28]], 2: [[0.5], [L_SLOPE]]},
        (
            _blip("A1", "1.0"),
            _blip("A2", "1.0", "intercept"),
            _blip("A2", "1.0", "S2"),
            "ey_reference",
        ),
        features={2: ["S2"]},
    )

    n = law_n()
    out["N"] = _setup(
        "N",
        n,
        BINARY_REGIMENS,
        "r00",
        {_blip("A1", "1.0"): 1.14, _blip("A2", "1.0"): 0.5},
        {1: [[1.14]], 2: [[0.5]]},
        (
            _blip("A1", "1.0"),
            _blip("A2", "1.0"),
            "ey_reference",
            _pair("r11", "r00"),
        ),
        nuisance="booster",
    )

    out["S2W"] = _setup(
        "S2W",
        s2,
        BINARY_REGIMENS,
        "r00",
        {_blip("A1", "1.0"): 1.28, _blip("A2", "1.0"): 0.5},
        {1: [[1.28]], 2: [[0.5]]},
        (_blip("A1", "1.0"), "ey_reference", _level("r11"), _pair("r11", "r00")),
        target=law_s2_reweighted(),
        weighted=True,
    )

    out["S2C"] = _setup(
        "S2C",
        law_s2_cluster(),
        BINARY_REGIMENS,
        "r00",
        {_blip("A1", "1.0"): 1.28, _blip("A2", "1.0"): 0.5},
        {1: [[1.28]], 2: [[0.5]]},
        (_blip("A1", "1.0"), "ey_reference", _level("r11"), _pair("r11", "r00")),
        cluster=True,
    )

    # The negative controls fit the structural estimator with too few features.  Their
    # coefficients are not blips of the law, so ``psi`` is empty and the estimands are the
    # regimen quantities whose truths the g-formula states.
    out["LU"] = _setup(
        "LU",
        law_l,
        BINARY_REGIMENS,
        "r00",
        {},
        {},
        # ``ate_regimen[r11 vs r00]`` is left out: its limit misses the truth by -0.017, 0.4
        # sampling SD at n = 2,000, so no cell could fail on it with power.
        (_level("r11"), _pair("r10", "r00"), _pair("r01", "r00")),
    )
    out["VB"] = _setup(
        "VB",
        p,
        BINARY_REGIMENS,
        "r00",
        {},
        {},
        (_level("r11"), _pair("r11", "r00"), _pair("r10", "r00"), _pair("r01", "r00")),
        features={2: ["W"]},
    )

    # The node-1 blip carries the effect of ``A1`` on the covariate ``L2``, which is a logistic
    # step in ``W``.  With a binary baseline it is still linear in ``(1, W)``.
    step0 = float(expit(-0.2 + 0.9) - expit(-0.2))
    step1 = float(expit(-0.2 + 0.7 + 0.9) - expit(-0.2 + 0.7))
    sp_1 = 1.0 + 0.7 * step0
    sp_1_w = 0.5 + 0.7 * (step1 - step0)
    out["SP"] = _setup(
        "SP",
        law_sparse(),
        BINARY_REGIMENS,
        "r00",
        {
            _blip("A1", "1.0", "intercept"): sp_1,
            _blip("A1", "1.0", "W"): sp_1_w,
            _blip("A2", "1.0", "intercept"): 0.5,
            _blip("A2", "1.0", "W"): 0.4,
            _blip("A2", "1.0", "A1"): 0.6,
        },
        {1: [[sp_1], [sp_1_w]], 2: [[0.5], [0.4], [0.6]]},
        (
            _blip("A1", "1.0", "intercept"),
            _blip("A1", "1.0", "W"),
            _blip("A2", "1.0", "intercept"),
            _blip("A2", "1.0", "W"),
            _blip("A2", "1.0", "A1"),
            "ey_reference",
            _level("r11"),
            _pair("r11", "r00"),
            _pair("r10", "r00"),
            _pair("r01", "r00"),
        ),
        features={1: ["W"], 2: ["W", "A1"]},
    )
    return out


@functools.cache
def setups() -> dict[str, Setup]:
    """Every setup of the study, built once."""
    return _build()


def setup(name: str) -> Setup:
    """One setup by name."""
    return setups()[name]


# ------------------------------------------------------------------------------- truths


@functools.cache
def _truth_cache(name: str) -> dict[str, float]:
    chosen = setup(name)
    law = chosen.target
    out = dict(chosen.psi)
    out["ey_reference"] = finite.reference_mean(law)
    levels = {
        label: finite.regimen_mean(law, regimen) for label, regimen in chosen.regimens.items()
    }
    for label, value in levels.items():
        out[_level(label)] = value
        if label != chosen.reference:
            out[_pair(label, chosen.reference)] = value - levels[chosen.reference]
    return out


def truths(name: str) -> dict[str, float]:
    """The truth of every estimand a setup can report: its blips, the reference mean, the levels
    and the contrasts, each by the g-formula on the target law."""
    return dict(_truth_cache(name))


def truth(name: str, estimand: str) -> float:
    """The truth of one estimand of a setup."""
    return truths(name)[estimand]


def outcome_support(name: str) -> tuple[float, float]:
    """The exact support of the outcome: the least and the greatest value over the paths."""
    values = finite.frame(setup(name).law)["Y"]
    return float(values.min()), float(values.max())


# --------------------------------------------------------------------- the comparator limit


def _coding_columns(law: Law, node: int) -> tuple[list[float], bool]:
    """The coded levels of a node (every level but the reference), and whether it is a dose."""
    name = law.treatments[node - 1]
    if name in law.doses:
        return [], True
    levels = sorted({value for path, _ in finite.paths(law) for value in [path[name]]})
    return [level for level in levels if level != law.reference_of(node)], False


def _state_names(law: Law, node: int, state: str) -> list[str]:
    """The state of period ``node``: the history, or the Markov state ``(W, S_t, A_{t-1})``."""
    if state == "history":
        return law.history_names(node)
    if state != "markov":
        raise ValueError(f"unknown state {state!r}")
    covariate = next(v.name for v in law.variables if v.role == "covariate" and v.node == node)
    names = [*law.baseline, covariate]
    if node > 1:
        names.append(law.treatments[node - 2])
    return names


@functools.cache
def _dynamic_dml_coefficients(
    name: str, state: str, features: tuple[str, ...]
) -> tuple[np.ndarray, ...]:
    table = finite.frame(setup(name).law)
    law = setup(name).law
    weight = table["p"].to_numpy()
    phi = np.column_stack(
        [np.ones(len(table)), *(table[column].to_numpy(dtype=float) for column in features)]
    )
    codings = []
    for node in range(1, law.n_nodes + 1):
        levels, dose = _coding_columns(law, node)
        values = table[law.treatments[node - 1]].to_numpy(dtype=float)
        if dose:
            codings.append(values.reshape(-1, 1))
        else:
            codings.append(np.column_stack([(values == level).astype(float) for level in levels]))

    def given(names: list[str], values: np.ndarray) -> np.ndarray:
        frame = table[names].copy()
        keys = list(names)
        sums = (
            pd.DataFrame(values * weight[:, None], index=table.index)
            .groupby([frame[key] for key in keys])
            .transform("sum")
        )
        mass = (
            pd.Series(weight, index=table.index)
            .groupby([frame[key] for key in keys])
            .transform("sum")
        )
        return sums.to_numpy() / mass.to_numpy()[:, None]

    def kron(residual: np.ndarray) -> np.ndarray:
        # DynamicDML's final stage reads the treatment residual times the features ``(1, X)``,
        # coding column first and feature second.
        return np.einsum("il,ik->ilk", residual, phi).reshape(len(table), -1)

    y = table["Y"].to_numpy(dtype=float).reshape(-1, 1)
    nodes = law.n_nodes
    residual_y: dict[int, np.ndarray] = {}
    residual_z: dict[tuple[int, int], np.ndarray] = {}
    for t in range(1, nodes + 1):
        names = _state_names(law, t, state)
        residual_y[t] = (y - given(names, y)).reshape(-1)
        for j in range(t, nodes + 1):
            residual_z[(j, t)] = kron(codings[j - 1] - given(names, codings[j - 1]))
    theta: dict[int, np.ndarray] = {}
    for t in range(nodes, 0, -1):
        target = residual_y[t] - sum(
            (residual_z[(j, t)] @ theta[j] for j in range(t + 1, nodes + 1)),
            np.zeros(len(table)),
        )
        own = residual_z[(t, t)]
        theta[t] = np.linalg.solve(own.T @ (own * weight[:, None]), own.T @ (target * weight))
    return tuple(theta[t] for t in range(1, nodes + 1))


def dynamic_dml_limit(
    name: str, estimand: str, *, state: str = "markov", features: Sequence[str] = ()
) -> float:
    r"""The population limit of EconML's ``DynamicDML`` fit of one estimand of a setup.

    The fit reads the period ``state`` (``"markov"``: ``(W, S_t, A_{t-1})``; ``"history"``:
    :math:`H_t`), saturated nuisances, and the blip features ``(1, X)`` with ``X`` the
    baseline ``features``.  Each nuisance is then the exact conditional mean of its target
    given the state, and the final stage solves the backward g-estimating equations of
    ``DynamicDML`` at the law's probabilities.  A blip estimand is one coefficient, and a
    contrast is :math:`E_X \sum_t (c(a_t) - c(a'_t))^\top \theta_t (1, X)`.  The limit equals
    the truth exactly when the state removes the confounding and the blips read only ``X``.
    """
    chosen = setup(name)
    law = chosen.law
    theta = _dynamic_dml_coefficients(name, state, tuple(features))
    width = 1 + len(features)
    match = re.fullmatch(r"blip\[A(\d+):([\w.]+):(\w+)\]", estimand)
    if match:
        node, level, feature = int(match.group(1)), match.group(2), match.group(3)
        levels, dose = _coding_columns(law, node)
        column = 0 if dose else levels.index(float(level))
        offset = 0 if feature == "intercept" else 1 + list(features).index(feature)
        return float(theta[node - 1][column * width + offset])
    match = re.fullmatch(r"ate_regimen\[(\w+) vs (\w+)\]", estimand)
    if match is None:
        raise KeyError(f"DynamicDML reports no {estimand}")
    treated, control = (chosen.regimens[label] for label in match.groups())
    table = finite.frame(law)
    weight = table["p"].to_numpy()
    phi = np.column_stack(
        [np.ones(len(table)), *(table[column].to_numpy(dtype=float) for column in features)]
    )
    effect = np.zeros(len(table))
    for node in range(1, law.n_nodes + 1):
        levels, dose = _coding_columns(law, node)

        def code(value: float, levels: list[float] = levels, dose: bool = dose) -> np.ndarray:
            if dose:
                return np.array([value])
            return np.array([float(value == level) for level in levels])

        change = code(treated[node - 1]) - code(control[node - 1])
        coefficients = theta[node - 1].reshape(len(change), width)
        effect += phi @ (change @ coefficients)
    return float(effect @ weight)


# ------------------------------------------------------------------------------ samplers


@functools.cache
def _table(name: str) -> pd.DataFrame:
    table = finite.frame(setup(name).law)
    return table.reset_index(drop=True)


@functools.cache
def _by_baseline(name: str) -> dict[float, tuple[pd.DataFrame, np.ndarray]]:
    table = _table(name)
    out = {}
    for value, group in table.groupby("W"):
        probability = group["p"].to_numpy()
        out[float(value)] = (group.reset_index(drop=True), probability / probability.sum())
    return out


def _rows(table: pd.DataFrame, probability: np.ndarray, n: int, rng: np.random.Generator) -> Any:
    index = rng.choice(len(table), size=n, p=probability)
    return table.iloc[index].reset_index(drop=True)


def sample(name: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of a setup's law.

    A weighted setup carries the known weight ``w``.  A clustered setup draws clusters of
    :data:`CLUSTER_SIZE` that share the baseline, adds one outcome intercept per cluster with
    mean zero (so no truth moves) and carries the cluster column ``id``.
    """
    chosen = setup(name)
    rng = np.random.default_rng(seed)
    if chosen.cluster:
        count = -(-n // CLUSTER_SIZE)
        baselines = rng.binomial(1, 0.4, size=count).astype(float)
        frames = []
        effect = rng.normal(0.0, CLUSTER_OUTCOME_SD, size=count)
        for index, value in enumerate(baselines):
            table, probability = _by_baseline(name)[value]
            rows = _rows(table, probability, CLUSTER_SIZE, rng)
            rows["Y"] = rows["Y"] + effect[index]
            rows["id"] = index
            frames.append(rows)
        frame = pd.concat(frames, ignore_index=True).iloc[:n]
    else:
        table = _table(name)
        frame = _rows(table, table["p"].to_numpy(), n, rng)
    frame = frame.drop(columns=["p", "E"])
    if chosen.weighted:
        frame["w"] = weight(frame["W"])
    return frame


__all__ = [
    "BINARY_REGIMENS",
    "CLUSTER_SIZE",
    "Setup",
    "dynamic_dml_limit",
    "outcome_support",
    "sample",
    "setup",
    "setups",
    "truth",
    "truths",
]
