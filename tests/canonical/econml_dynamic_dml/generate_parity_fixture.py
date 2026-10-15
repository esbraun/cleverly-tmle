"""Write the EconML ``DynamicDML`` parity fixture for ``StructuralNestedMean``.

Run this file inside the pinned environment of ``requirements.txt`` (see ``README.md``).  It
draws each parity case from a seeded generator, fits EconML 0.17.0 on the case with the setup
below, and writes the inputs, the folds, and EconML's coefficients and covariance to
``parity_fixture.json``.  The fast test ``tests/unit/test_structural_nested_mean.py`` reads that
file.  No test and no library module imports EconML.

Parity setup.  The two sides must share every choice that is not the algorithm.

* Folds.  A numpy permutation assigns each unit to one of two folds.  EconML receives the
  folds as a ``cv`` iterable over the long rows.  ``StructuralNestedMean`` receives the same
  assignment.
* Nuisances.  Both sides fit ordinary least squares for the outcome.  A binary or categorical
  treatment uses :class:`LinearProbability` (least squares on the indicators, not clipped).
  One case uses a logistic fit with a tight tolerance on both sides.  A dose uses least squares.
* State.  EconML reads one period row per node.  The row holds the baseline features ``X_0``
  repeated on every period, and the history as ``W``, zero-padded on the early rows.  A
  ``VarianceThreshold`` step drops the padding columns.  ``StructuralNestedMean`` reads the same
  history.
* Variance.  EconML fills the off-diagonal Jacobian blocks of a vector coding in the
  transposed order.  The generator recomputes both sandwiches from EconML's own residuals for
  the cases without features and writes both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

HERE = Path(__file__).resolve().parent
ECONML_VERSION = "0.17.0"
FIXTURE_VERSION = 1
N_PERIODS = 2
BOOTSTRAP_DRAWS = 20000


class LinearProbability(BaseEstimator, ClassifierMixin):
    """Least squares on the class indicators, with the fitted values as probabilities.

    The fit is not clipped to the unit interval.  ``cleverly`` and EconML both read
    ``predict_proba``, so the two sides share the nuisance exactly.
    """

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> LinearProbability:
        """Fit one least-squares regression per class indicator."""
        y = np.asarray(y)
        x = np.asarray(x, dtype=float)
        self.classes_ = np.unique(y)
        design = np.column_stack([np.ones(len(x)), x])
        target = (y[:, None] == self.classes_[None, :]).astype(float)
        self.beta_ = np.linalg.lstsq(design, target, rcond=None)[0]
        return self

    def predict_proba(self, x: Any) -> Any:
        """Return the fitted indicator means, one column per class."""
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones(len(x)), x]) @ self.beta_

    def predict(self, x: Any) -> Any:
        """Return the class with the largest fitted indicator mean."""
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


#: One entry per parity case.  ``kind`` is the treatment type of every node.  ``features`` puts
#: the baseline covariate ``W0`` in the blip of each node.  ``nuisance`` is the treatment model.
CASES: dict[str, dict[str, Any]] = {
    "binary_scalar": {
        "kind": "binary",
        "features": False,
        "nuisance": "linear_probability",
        "n": 500,
        "seed": 11,
        "contrast": [1.0, 1.0],
    },
    "binary_features": {
        "kind": "binary",
        "features": True,
        "nuisance": "linear_probability",
        "n": 500,
        "seed": 12,
        "contrast": None,
    },
    "binary_logistic": {
        "kind": "binary",
        "features": False,
        "nuisance": "logistic",
        "n": 600,
        "seed": 13,
        "contrast": [1.0, 1.0],
    },
    "categorical": {
        "kind": "categorical",
        "features": False,
        "nuisance": "linear_probability",
        "n": 800,
        "seed": 14,
        "contrast": [1.0, 2.0],
    },
    "categorical_features": {
        "kind": "categorical",
        "features": True,
        "nuisance": "linear_probability",
        "n": 800,
        "seed": 15,
        "contrast": None,
    },
    "continuous": {
        "kind": "continuous",
        "features": False,
        "nuisance": "linear",
        "n": 500,
        "seed": 16,
        "contrast": [1.0, 1.0],
    },
    "continuous_features": {
        "kind": "continuous",
        "features": True,
        "nuisance": "linear",
        "n": 500,
        "seed": 17,
        "contrast": None,
    },
}


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _draw_treatment(kind: str, linear: Any, rng: np.random.Generator) -> Any:
    n = len(linear)
    if kind == "binary":
        return (rng.random(n) < _expit(linear)).astype(float)
    if kind == "categorical":
        logits = np.column_stack([np.zeros(n), linear, -0.8 * linear + 0.2])
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        return (rng.random(n)[:, None] > np.cumsum(prob, axis=1)).sum(axis=1).astype(float)
    return linear + rng.normal(size=n)


def _code(kind: str, a: Any) -> Any:
    """The coding of a treatment value: the indicator, the one-hot columns, or the dose."""
    if kind == "categorical":
        return np.column_stack([(a == 1).astype(float), (a == 2).astype(float)])
    return np.asarray(a, dtype=float).reshape(-1, 1)


def draw_case(spec: dict[str, Any]) -> dict[str, Any]:
    """Draw the columns of one case: ``W0, L1, A1, L2, A2, Y`` and the fold assignment."""
    rng = np.random.default_rng(spec["seed"])
    n, kind = spec["n"], spec["kind"]
    d = 2 if kind == "categorical" else 1
    w0 = rng.normal(size=n)
    l1 = 0.6 * w0 + rng.normal(size=n)
    a1 = _draw_treatment(kind, 0.5 * l1 + 0.3 * w0, rng)
    c1 = _code(kind, a1)
    l2 = 0.5 * l1 + 0.8 * c1.sum(axis=1) + rng.normal(size=n)
    a2 = _draw_treatment(kind, 0.4 * l2 + 0.3 * c1.sum(axis=1) - 0.2 * w0, rng)
    c2 = _code(kind, a2)
    b1 = np.array([1.0, -0.7])[:d]
    b2 = np.array([0.7, 0.9])[:d]
    slope1 = np.array([0.5, 0.3])[:d] if spec["features"] else np.zeros(d)
    slope2 = np.array([-0.4, 0.6])[:d] if spec["features"] else np.zeros(d)
    y = (
        c1 @ b1
        + (c1 * slope1).sum(axis=1) * w0
        + c2 @ b2
        + (c2 * slope2).sum(axis=1) * w0
        + 0.7 * l2
        + 0.3 * l2**2
        + 0.5 * w0
        + rng.normal(size=n)
    )
    folds = (rng.permutation(n) % 2).astype(int)
    columns = {"W0": w0, "L1": l1, "A1": a1, "L2": l2, "A2": a2, "Y": y}
    return {
        "columns": {key: value.tolist() for key, value in columns.items()},
        "folds": folds.tolist(),
    }


def _long_layout(spec: dict[str, Any], columns: dict[str, Any]) -> dict[str, Any]:
    """Arrange the case as EconML's long panel: row ``2 * i + t`` is unit ``i`` at period ``t``."""
    kind = spec["kind"]
    w0, l1, l2 = (np.asarray(columns[name]) for name in ("W0", "L1", "L2"))
    a1, a2, y = (np.asarray(columns[name]) for name in ("A1", "A2", "Y"))
    n = len(y)
    c1 = _code(kind, a1)
    history = [np.zeros((n, 0)), c1]
    padded = np.zeros((n * N_PERIODS, 1 + c1.shape[1] + 1))
    padded[0::2, 0] = l1
    padded[1::2, 0] = l1
    padded[1::2, 1 : 1 + c1.shape[1]] = history[1]
    padded[1::2, -1] = l2
    base = w0.reshape(-1, 1)
    long_x = np.repeat(base, N_PERIODS, axis=0)
    if spec["features"]:
        x, w = long_x, padded
    else:
        x = None
        w = np.column_stack([long_x, padded])
    treatment = np.column_stack([a1, a2]).reshape(-1)
    return {
        "y": np.repeat(y, N_PERIODS),
        "t": treatment,
        "x": x,
        "w": w,
        "groups": np.repeat(np.arange(n), N_PERIODS),
    }


def _splits(folds: Any) -> list[tuple[Any, Any]]:
    folds = np.asarray(folds)
    out = []
    for k in (0, 1):
        test_units = np.flatnonzero(folds == k)
        train_units = np.flatnonzero(folds != k)
        rows = lambda units: np.sort(  # noqa: E731
            np.concatenate([N_PERIODS * units + t for t in range(N_PERIODS)])
        )
        out.append((rows(train_units), rows(test_units)))
    return out


def _residual_pieces(est: Any, n: int, d: int) -> dict[str, Any]:
    """EconML's cached residuals for two periods, as ``(n, d)`` blocks.

    ``z(t, j)`` is the residual of the coding of treatment ``j`` given the state of period
    ``t``.  EconML stores it on the period-``j`` rows with ``t`` on the last axis.
    """
    y_res, t_res = est._cached_values.nuisances
    n_rows = n * N_PERIODS
    pf = [np.arange(t, n_rows, N_PERIODS) for t in range(N_PERIODS)]

    def z(t: int, j: int) -> Any:
        return np.asarray(t_res[pf[j]][..., t]).reshape(n, d)

    return {
        "y0": np.asarray(y_res[pf[0]]).reshape(n),
        "y1": np.asarray(y_res[pf[1]]).reshape(n),
        "z00": z(0, 0),
        "z01": z(0, 1),
        "z11": z(1, 1),
    }


def _sandwiches(est: Any, n: int, d: int) -> dict[str, Any]:
    """Recompute the covariance from EconML's residuals, with both Jacobian orientations.

    A case without features has one coefficient block of ``d`` entries per period.  Equation
    ``t`` is differentiated in ``theta_j``: its derivative is ``E[z(t, t) z(t, j)']``, the
    ``correct`` block.  EconML fills the block with the transpose.
    """
    p = _residual_pieces(est, n, d)
    coef = np.asarray(est._ortho_learner_model_final._get_coef_()).reshape(-1)
    theta = [coef[t * d : (t + 1) * d] for t in range(N_PERIODS)]
    eps1 = p["y1"] - p["z11"] @ theta[1]
    eps0 = p["y0"] - p["z00"] @ theta[0] - p["z01"] @ theta[1]
    score = np.hstack([p["z00"] * eps0[:, None], p["z11"] * eps1[:, None]])
    correct = np.zeros((2 * d, 2 * d))
    correct[:d, :d] = p["z00"].T @ p["z00"] / n
    correct[:d, d:] = p["z00"].T @ p["z01"] / n
    correct[d:, d:] = p["z11"].T @ p["z11"] / n
    transposed = correct.copy()
    transposed[:d, d:] = p["z01"].T @ p["z00"] / n
    sigma = score.T @ score / n
    out = {}
    for name, jac in (("correct", correct), ("block_transposed", transposed)):
        inv = np.linalg.inv(jac)
        out[name] = (inv @ sigma @ inv.T / n).tolist()
    return out


def _bootstrap_contrast_se(est: Any, n: int, d: int, levels: list[float]) -> float:
    """Standard error of the contrast by resampling units, with the residuals held fixed.

    This is an estimate of the final-stage sampling variance that uses neither Jacobian
    orientation.  ``levels`` are the treatment levels of the contrast against the reference.
    """
    p = _residual_pieces(est, n, d)
    index = [int(level) - 1 if d > 1 else 0 for level in levels]
    rng = np.random.default_rng(0)
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        rows = rng.integers(0, n, size=n)
        z11, z00, z01 = p["z11"][rows], p["z00"][rows], p["z01"][rows]
        theta1 = np.linalg.solve(z11.T @ z11, z11.T @ p["y1"][rows])
        theta0 = np.linalg.solve(z00.T @ z00, z00.T @ (p["y0"][rows] - z01 @ theta1))
        values.append(theta0[index[0]] + theta1[index[1]])
    return float(np.std(values))


def fit_econml(spec: dict[str, Any], draw: dict[str, Any]) -> dict[str, Any]:
    """Fit ``DynamicDML`` on one drawn case and return what the fixture records."""
    import warnings

    from econml.panel.dml import DynamicDML
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline

    layout = _long_layout(spec, draw["columns"])
    discrete = spec["kind"] != "continuous"
    if spec["nuisance"] == "linear_probability":
        model_t: Any = make_pipeline(VarianceThreshold(0.0), LinearProbability())
    elif spec["nuisance"] == "logistic":
        model_t = make_pipeline(
            VarianceThreshold(0.0),
            LogisticRegression(penalty=None, solver="newton-cholesky", tol=1e-14, max_iter=500),
        )
    else:
        model_t = make_pipeline(VarianceThreshold(0.0), LinearRegression())
    model_y = make_pipeline(VarianceThreshold(0.0), LinearRegression())
    est = DynamicDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=discrete,
        cv=_splits(draw["folds"]),
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(
            layout["y"],
            layout["t"],
            X=layout["x"],
            W=layout["w"],
            groups=layout["groups"],
            cache_values=True,
        )
    final = est._ortho_learner_model_final
    coef = np.asarray(final._get_coef_())
    cov = np.asarray(final.model_final_._param_var)
    n = len(draw["folds"])
    record: dict[str, Any] = {
        "coef": coef.tolist(),
        "coef_shape": list(coef.shape),
        "cov": cov.tolist(),
        "ate": None,
        "sandwiches": None,
        "bootstrap_contrast_se": None,
    }
    contrast = spec["contrast"]
    if contrast is not None:
        t1 = np.asarray(contrast, dtype=float).reshape(1, -1)
        t0 = np.zeros_like(t1)
        inference = est.ate_inference(T0=t0, T1=t1)
        record["ate"] = {
            "t0": t0.reshape(-1).tolist(),
            "t1": t1.reshape(-1).tolist(),
            "point": float(np.asarray(inference.mean_point).reshape(-1)[0]),
            "stderr": float(np.asarray(inference.stderr_mean).reshape(-1)[0]),
        }
        d = 2 if spec["kind"] == "categorical" else 1
        record["sandwiches"] = _sandwiches(est, n, d)
        record["bootstrap_contrast_se"] = _bootstrap_contrast_se(est, n, d, contrast)
    return record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned_wheel_hash() -> str:
    text = (HERE / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(
        r"^econml==" + re.escape(ECONML_VERSION) + r" \\\n\s+--hash=sha256:(\w+)", text, re.M
    )
    if match is None:
        raise SystemExit("requirements.txt does not pin econml with one wheel hash")
    return match.group(1)


def build_fixture() -> dict[str, Any]:
    """Run every case and assemble the fixture."""
    import econml

    if econml.__version__ != ECONML_VERSION:
        raise SystemExit(f"need econml {ECONML_VERSION}; found {econml.__version__}")
    cases = {}
    for name, spec in CASES.items():
        draw = draw_case(spec)
        cases[name] = {
            "spec": spec,
            "columns": draw["columns"],
            "folds": draw["folds"],
            "econml": fit_econml(spec, draw),
        }
    return {
        "fixture_version": FIXTURE_VERSION,
        "econml_version": ECONML_VERSION,
        "wheel_sha256": _pinned_wheel_hash(),
        "generator_sha256": _sha256(Path(__file__)),
        "dockerfile_sha256": _sha256(HERE / "Dockerfile"),
        "requirements_sha256": _sha256(HERE / "requirements.txt"),
        "cases": cases,
    }


def main() -> None:
    """Write the fixture next to this file, or to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "parity_fixture.json")
    args = parser.parse_args()
    fixture = build_fixture()
    text = json.dumps(fixture, indent=1, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(text) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
