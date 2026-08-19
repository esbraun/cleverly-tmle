"""Does cleverly work with exactly *one* dataframe backend installed?

``pytest -m "not slow"`` cannot answer that.  The test suite imports pandas and polars
at module scope in dozens of files -- reasonably, since it compares them against each
other -- so it can only ever run where both are present, and ``.[dev]`` installs both.
That left the single-backend branches of :mod:`cleverly.utils.frames` unexecuted
anywhere, and it is how :func:`cleverly.datasets.make_longitudinal_weighted` came to
force ``backend="pandas"`` and call ``.loc``: unusable on a polars-only install, and
invisible to a suite that always had the other one to fall back on.

So this is a script rather than a test, run by the ``minimal-install`` job in
``.github/workflows/ci.yml`` once per backend.  It asserts three things, which are the
three the failure above would have tripped:

1. the *other* backend really is absent, so a pass means something;
2. a fit end to end returns frames in the backend that is present, from the estimate and
   from the diagnostics alike;
3. every dataset generator produces one, including the two that build a frame, subset it
   and rebuild it.

Usage::

    python scripts/smoke_backend.py pandas
    python scripts/smoke_backend.py polars
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate, make_longitudinal, make_longitudinal_weighted
from cleverly.utils.frames import available_backends

OTHER = {"pandas": "polars", "polars": "pandas"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"  ok: {message}")


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "pandas"
    if backend not in OTHER:
        raise SystemExit(f"usage: {sys.argv[0]} [pandas|polars]")
    frame_type = importlib.import_module(backend).DataFrame

    print(f"backend under test: {backend}")
    check(
        OTHER[backend] not in available_backends(),
        f"{OTHER[backend]} is absent, so this exercises the single-backend path",
    )
    check(available_backends()[:1] == (backend,), f"{backend} is the only backend installed")

    frame, _ = make_linear_ate(n=400, seed=0, backend=backend)
    check(isinstance(frame, frame_type), f"make_linear_ate returns a {backend} frame")

    result = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
        ),
    ).estimate(
        ATE(),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=4,
        learner_folds=3,
        random_state=0,
    )
    check(np.isfinite(result.psi("ate")), "the fit produced a finite estimate")
    check(isinstance(result.to_frame(), frame_type), "to_frame() follows the backend")
    check(
        isinstance(result.validation.score_check().to_frame(), frame_type),
        "the score check follows the backend with nothing threaded in",
    )
    check(
        isinstance(result.sensitivity.positivity().to_frame(), frame_type),
        "the positivity report follows the backend with nothing threaded in",
    )

    # The two generators that build a frame, subset it and rebuild it -- the shape that
    # reached for `.loc` and so pinned itself to pandas.
    for name, generator in (
        ("make_longitudinal", make_longitudinal),
        ("make_longitudinal_weighted", make_longitudinal_weighted),
    ):
        built, _ = generator(n=200, seed=1, backend=backend)
        check(isinstance(built, frame_type), f"{name} returns a {backend} frame")

    print(f"\n{backend}-only install: all checks passed")


if __name__ == "__main__":
    main()
