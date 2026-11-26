"""The reviewed semantic callback for ``docs/examples/learner-choice``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals are
also compared against its stored outputs.  The repeated-draw numbers its readings quote come from
the page's probe, ``reviews/notebook-review/probes/learner-choice``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import navigation_protocol
from cleverly.learners import screen_by_outcome
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refutation",)

NOTEBOOK = EXAMPLES / "learner-choice.ipynb"

_PROBE = ROOT / "reviews/notebook-review/probes/learner-choice"

_SUMMARY = "reviews/notebook-review/probes/learner-choice/summary.log"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.0003": f"mean screened-minus-unscreened estimate, 200 draws: {_SUMMARY}",
    "0.0025": f"mean absolute screened-minus-unscreened estimate, 200 draws: {_SUMMARY}",
    "0.0032": f"SD of the screened-minus-unscreened estimate, 200 draws: {_SUMMARY}",
    "1.02": f"median screened SE over unscreened SE, 200 draws: {_SUMMARY}",
    "0.0016": f"mean screened estimate minus the truth, 200 draws: {_SUMMARY}",
    "0.0018": f"mean unscreened estimate minus the truth, 200 draws: {_SUMMARY}",
}

#: The lines of the probe summary that the readings quote, by the words the summary prints.
_QUOTED = (
    "screened minus unscreened: mean 0.0003",
    "SD 0.0032, mean absolute 0.0025",
    "screened SE / unscreened SE: median 1.018",
    "screened: bias -0.0016",
    "unscreened: bias -0.0018",
    "draws where the screen drops medication_burden: 147",
    "draws where the screen drops age: 162",
    "draws where the declared interval contains the truth: 190",
    "draws where the screened interval contains the truth: 192",
    "coverage 0.965",
)


def _probe_config() -> dict[str, Any]:
    """The settings the probe ran, from its ``sweep.py``."""
    spec = importlib.util.spec_from_file_location("learner_choice_sweep", _PROBE / "sweep.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config: dict[str, Any] = module.CONFIG
    return config


def _weights(table: pd.DataFrame, model: str) -> pd.Series:
    rows = table[table["model"] == model]
    return pd.Series(rows["weight"].to_numpy(), index=rows["candidate"].to_numpy())


#: The candidates the probe's ``sweep.py`` builds, by name and estimator class.
_CANDIDATES = {
    "outcome_ensemble": {
        "linear": "LinearRegression",
        "splines": "Pipeline",
        "boosting": "HistGradientBoostingRegressor",
    },
    "treatment_ensemble": {
        "logistic": "LogisticRegression",
        "splines": "Pipeline",
        "boosting": "HistGradientBoostingClassifier",
    },
}


def _check_settings(namespace: dict[str, Any]) -> None:
    """The page's learners are the ones the probe repeated."""
    config = _probe_config()
    assert len(namespace["frame"]) == config["n"]
    for key, expected in _CANDIDATES.items():
        library = namespace[key].library
        assert {name: type(candidate).__name__ for name, candidate in library} == expected
        splines = dict(library)["splines"]
        assert [type(step).__name__ for _, step in splines.steps][1] == (
            "RidgeCV" if key == "outcome_ensemble" else "LogisticRegression"
        )
    assert tuple(namespace["outcome_ensemble"].clip) == (0.0, 1.0)
    assert namespace["treatment_ensemble"].clip is None
    for ensemble in (namespace["outcome_ensemble"], namespace["treatment_ensemble"]):
        assert ensemble.n_folds == config["inner_folds"]
        for name, candidate in ensemble.library:
            if name == "boosting":
                params = candidate.get_params()
                assert {key: params[key] for key in config["booster"]} == config["booster"]
            if name == "splines":
                assert candidate.steps[0][1].n_knots == config["n_knots"]
    method = namespace["declared_method"]
    assert method.cross_fitting.n_folds == config["outer_folds"]
    assert tuple(method.targeting.q_bounds) == tuple(config["q_bounds"])
    assert namespace["noise"].shape[1] == config["noise_columns"]
    assert namespace["screened_models"].min_retain == config["min_retain"]
    plain = namespace["plain"].models
    for learner in (plain.outcome_learner, plain.treatment_learner):
        params = learner.get_params()
        assert {key: params[key] for key in config["booster"]} == config["booster"]


def check(namespace: dict[str, Any]) -> None:
    """Check the page's claims."""
    truth = namespace["truth"]
    protocol = namespace["protocol"]
    declared = namespace["declared"]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, declared)
    assert changed_fields(protocol, navigation_protocol()) == set()
    _check_settings(namespace)

    # Count the work: 2 models x 5 folds, each a 3-candidate ensemble on 5 inner folds plus a
    # refit (3 x 6 = 18 per fit); the declared ensembles use 3 inner folds (3 x 4 = 12 per fit).
    planned = namespace["planned"]
    assert planned.total_fits == 10
    assert planned.total_candidate_fits == 10 * 3 * (5 + 1) == 180
    declared_plan = namespace["effect"].planned_fits(method=namespace["declared_method"])
    assert declared_plan.total_candidate_fits == 10 * 3 * (3 + 1) == 120

    # The blended fit's interval covers the truth.
    estimate = namespace["estimate"]
    assert covers(estimate, truth["ate"])

    weights = namespace["weights"]
    for model in ("treatment model", "outcome model"):
        assert np.isclose(_weights(weights, model).sum(), 1.0)
    outcome = _weights(weights, "outcome model")
    treatment = _weights(weights, "treatment model")
    assert outcome.idxmax() == "boosting"
    assert outcome["linear"] == 0.0
    assert treatment.idxmax() == "splines"
    assert (treatment > 0).all()

    # The default ensembles: weights on the simplex, the forest carries the treatment model,
    # boosting the outcome model, and the default booster has the worst treatment-model risk.
    default_treatment = namespace["default_treatment"].diagnostics_
    default_outcome = namespace["default_outcome"].diagnostics_
    for found in (default_treatment, default_outcome):
        assert np.isclose(found.weights.sum(), 1.0)
    assert default_treatment.names[int(np.argmax(default_treatment.weights))] == "random_forest"
    assert default_outcome.names[int(np.argmax(default_outcome.weights))] == (
        "hist_gradient_boosting"
    )
    assert default_treatment.names[int(np.argmax(default_treatment.cv_risk))] == (
        "hist_gradient_boosting"
    )

    # Screening: the printed set is the screen's, the lasso alone keeps only the two columns with
    # a straight-line link to the score, and the floor of 4 adds two noise columns.  A no-op screen
    # would keep all 24, and so does the default floor.
    wide = namespace["wide"]
    columns = list(namespace["wide_covariates"])
    arrays = (
        wide[columns].to_numpy(),
        wide["transition_navigation"].to_numpy(),
        wide["transition_score"].to_numpy(),
    )
    kept = [name for name, keep in zip(columns, namespace["kept"], strict=True) if keep]
    assert kept == ["discharge_risk", "prior_utilization", "record_09", "record_17"]
    lasso_only = screen_by_outcome(*arrays, min_retain=1)
    assert [name for name, keep in zip(columns, lasso_only, strict=True) if keep] == [
        "discharge_risk",
        "prior_utilization",
    ]
    assert len(kept) < len(columns) == 24
    assert namespace["floor"] == 24

    # One candidate per fold: fold-averaged weights are multiples of 1/5.
    discrete = namespace["discrete"]
    for model in discrete.diagnostics.nuisance_models().models:
        shares = np.array(list(model.learner_weights.values())) * 5
        assert np.allclose(shares, np.round(shares))
    picks = {
        model.name: model.learner_weights for model in discrete.diagnostics.nuisance_models().models
    }
    assert picks["outcome"]["boosting"] == 1.0
    assert np.isclose(picks["propensity"]["splines"], 0.8)
    assert np.isclose(picks["propensity"]["boosting"], 0.2)

    # Check the fit: nothing needs attention.
    assert not namespace["assessment"].attention

    # The repeated-draw numbers the readings quote are the probe's.
    summary = (ROOT / _SUMMARY).read_text()
    for quoted in _QUOTED:
        assert quoted in summary, quoted


def check_refits(namespace: dict[str, Any]) -> None:
    """Check the refutation cell."""
    frame = namespace["checks"].to_frame()
    assert list(frame["test"]) == ["placebo", "random_common_cause", "subset"]
    assert set(frame["verdict"]) == {"descriptive"}
    placebo = frame.set_index("test").loc["placebo", "refuted_mean"]
    assert abs(placebo) < 0.1 * namespace["estimate"].psi
