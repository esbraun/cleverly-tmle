"""Small modified-treatment-policy fits, one per policy shape and density-ratio route.

Each builder fits in sample with explicit linear learners, in seconds. The names are
``mtp/<shape>/<route>``: ``shape`` is a capped shift (``shift``), a banded
:class:`~cleverly.interventions.ModifiedPolicy` on a dose with a declared row-specific
:class:`~cleverly.interventions.Support` (``band``, which the band-rank placebo reads), a
two-node longitudinal fit with continuous nodes (``panel``), and a capped shift on a dose whose
declared atoms hold every observed and shifted dose (``atoms``, where the omitted-variable rows'
policy axis answers). ``route`` is the ``ratio=`` setting: ``default`` (``None``),
``density`` or ``classifier``.

:data:`MTP_ROUTE_KINDS` holds the builders as zero-argument functions with the ``must_run``
set of rows each one answered when the snapshot was measured, in the shape of
:class:`tests.unit._capability_sweep_support.Kind`, so the capability sweep registers them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.interventions import Shift, Support
from cleverly.longitudinal import LTMLE
from tests import dose_declarations
from tests.studies import longitudinal_mtp_common as panel_law

#: The ``ratio=`` setting of each route name.
ROUTES: dict[str, str | None] = {"default": None, "density": "density", "classifier": "classifier"}

#: The declared atoms of the ``atoms`` dose: every observed dose and every shifted one.
ATOMS = (0.0, 1.0, 2.0, 3.0, 4.0)
#: ``a + 1`` held at the top atom, so every shifted dose is an atom too.
ATOM_SHIFT = Shift(1.0, cap=4.0, name="up")
#: The banded map of :mod:`tests.unit.test_mtp_declared_support`, inside ``u(h) = 1 + W``.
BAND_SUPPORT = Support(0.0, dose_declarations.upper)


def _quiet(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def _learners(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        **overrides,
    }


def shift_frame() -> pd.DataFrame:
    """The continuous-dose law of ``make_shift_dose(300, 0)``."""
    from cleverly.datasets import make_shift_dose

    frame, _ = make_shift_dose(n=300, seed=0)
    return frame


def fit_shift(ratio: str | None) -> Any:
    """``Shift(0.5, cap=5.0)`` of the continuous dose of :func:`shift_frame`."""
    frame = shift_frame()
    bounds = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
    return _quiet(
        lambda: (
            TMLE(**_learners(policies=[Shift(0.5, cap=5.0)], ratio=ratio, q_bounds=bounds))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
    )


def fit_band(ratio: str | None) -> Any:
    """The banded map on ``A ~ U(0, 1 + W)`` with its declared support ``[0, 1 + W]``."""
    from tests.unit.test_mtp_declared_support import INSIDE, NATURAL

    frame = dose_declarations.point_frame(300, 0)
    return _quiet(
        lambda: (
            TMLE(
                **_learners(
                    policies=[NATURAL, INSIDE],
                    ratio=ratio,
                    outcome_learner=LogisticRegression(max_iter=1000),
                )
            )
            .fit(
                frame, outcome="Y", treatment="A", covariates=["W"], treatment_support=BAND_SUPPORT
            )
            .single()
        )
    )


def atoms_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A dose on the five atoms whose probabilities read ``W``, and a linear outcome."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    logits = np.column_stack([0.4 * k * w for k in range(len(ATOMS))])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    a = np.array([rng.choice(ATOMS, p=row) for row in p])
    y = 0.5 * a + w + rng.normal(size=n)
    return pd.DataFrame({"W": w, "A": a, "Y": y})


def fit_atoms(ratio: str | None) -> Any:
    """``a + 1`` held at 4 on a dose whose declared atoms hold every observed and shifted dose."""
    frame = atoms_frame()
    bounds = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
    return _quiet(
        lambda: (
            TMLE(**_learners(policies=[ATOM_SHIFT], ratio=ratio, q_bounds=bounds))
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_kind="continuous",
                dose_atoms=ATOMS,
            )
            .single()
        )
    )


def fit_panel(ratio: str | None) -> Any:
    """One-fold ``LTMLE`` of ``up`` against the natural course, at both continuous nodes."""
    frame = panel_law.sample_continuous(300, 0)
    return _quiet(
        lambda: LTMLE(
            {"natural": panel_law.NATURAL, "up": panel_law.UP},
            reference="natural",
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=1,
            ratio=ratio,
            simultaneous=False,
            random_state=0,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["A1", "A2"],
        )
    )


#: Each shape's builder, which takes the ``ratio=`` setting.
SHAPES: dict[str, Callable[[str | None], Any]] = {
    "shift": fit_shift,
    "band": fit_band,
    "panel": fit_panel,
    "atoms": fit_atoms,
}


def builder(shape: str, route: str) -> Callable[[], Any]:
    """The zero-argument fit of ``mtp/<shape>/<route>``."""

    def fit() -> Any:
        return SHAPES[shape](ROUTES[route])

    fit.__name__ = f"fit_mtp_{shape}_{route}"
    return fit


#: Every kind's name, ``mtp/<shape>/<route>``, and its builder.
FITS: dict[str, Callable[[], Any]] = {
    f"mtp/{shape}/{route}": builder(shape, route) for shape in SHAPES for route in ROUTES
}
