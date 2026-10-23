"""The TWINS cohort and fit of the notebook, for review probes outside the notebook.

The cohort and the method repeat the notebook's load-data and att-fit cells. ``TWINS_DATA`` names a
local directory with the three pinned CEVAE files; the default reads them from the pinned commit.
"""

import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from cleverly import (
    ATT,
    CausalStudy,
    CrossFitting,
    Inference,
    ModelSpec,
    PointTreatment,
    Runtime,
    SuperLearner,
    Targeting,
    TMLEMethod,
)
from cleverly.learners import GLM

SEED = 2026

COMMIT = "9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd"
D = os.environ.get(
    "TWINS_DATA",
    f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/{COMMIT}/datasets/TWINS/",
)
X = pd.read_csv(D + "twin_pairs_X_3years_samesex.csv")
T = pd.read_csv(D + "twin_pairs_T_3years_samesex.csv")
state_year = [X["brstate"], X["data_year"]]
missing_share = X["tobacco"].isna().groupby(state_year).transform("mean")
reporting_state = missing_share <= 0.5
eligible = reporting_state & X["tobacco"].notna()


def category(column, merge=None):
    values = X.loc[eligible, column].map(lambda v: merge.get(v, v) if merge else v)
    return values.map(lambda v: "not recorded" if pd.isna(v) else str(int(v))).to_numpy()


source = X.loc[eligible]
risk_items = ["diabetes", "chyper", "lung", "cardiac", "renal", "preterm", "pre4000"]
risk_missing = source[risk_items].isna()
risk = source[risk_items].fillna(0).astype(float)
pairs = pd.DataFrame(
    {
        "smoked": source["tobacco"].astype(int).to_numpy(),
        "birth_weight_g": (T.loc[eligible, "dbirwt_0"] + T.loc[eligible, "dbirwt_1"]).to_numpy()
        / 2,
        "mother_age": category("mager8", {1: 2, 8: 7}),
        "mother_education": category("meduc6"),
        "father_education": category("feduc6"),
        "father_age": category("dfageq"),
        "mother_race": category("mrace", dict.fromkeys(range(4, 10), 3)),
        "mother_hispanic": category("ormoth", dict.fromkeys(range(2, 6), 1)),
        "mother_birth_region": category("mplbir_reg"),
        "residence_region": category("brstate_reg"),
        "birth_year": category("data_year"),
        "birth_order": category("dtotord_min", dict.fromkeys(range(6, 40), 5)),
        "married": source["dmar"].astype(float).to_numpy(),
        "alcohol": category("alcohol"),
        "risk_factors_recorded": 1.0 - risk_missing["diabetes"].astype(float).to_numpy(),
        "diabetes": risk["diabetes"].to_numpy(),
        "chronic_hypertension": risk["chyper"].to_numpy(),
        "lung_heart_kidney_disease": risk[["lung", "cardiac", "renal"]].max(axis=1).to_numpy(),
        "previous_preterm_or_small": risk["preterm"].to_numpy(),
        "previous_infant_over_4000g": risk["pre4000"].to_numpy(),
    }
)
ADJUSTMENT = tuple(c for c in pairs.columns if c not in ("smoked", "birth_weight_g"))
BOUNDS = (227.0, 8165.0)


def outcome_library():
    return SuperLearner(
        library=[
            ("linear model", GLM(family="gaussian")),
            (
                "boosting",
                HistGradientBoostingRegressor(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=300,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ],
        task="regression",
        n_folds=3,
        random_state=SEED,
        n_jobs=1,
    )


def treatment_library():
    return SuperLearner(
        library=[
            ("logistic", LogisticRegression(max_iter=3_000, random_state=SEED)),
            (
                "boosting",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=2_000,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=SEED,
                ),
            ),
        ],
        task="classification",
        n_folds=3,
        random_state=SEED,
        n_jobs=1,
    )


def method():
    return TMLEMethod(
        models=ModelSpec(outcome_learner=outcome_library(), treatment_learner=treatment_library()),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
        targeting=Targeting(q_bounds=BOUNDS),
        inference=Inference(alpha=0.05),
        runtime=Runtime(random_state=SEED, n_jobs=1),
    )


def fit():
    study = CausalStudy(
        pairs,
        design=PointTreatment(outcome="birth_weight_g", treatment="smoked", adjustment=ADJUSTMENT),
    )
    return study.identify(ATT(reference=0)).estimate(method=method())
