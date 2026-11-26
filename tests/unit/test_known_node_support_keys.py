"""The support key of a declared node mechanism names each factor by role and node.

A declared treatment or censoring factor enters the key as ``known_treatment@t`` or
``known_censoring@t``, ``t`` the 1-based node, and never as the user's column name, so the
label of a declared-mechanism fit does not depend on what the columns are called. The rename
witnesses refit a study's own case with every column renamed. Each has a control that keys by
column name and must fail it, and the partial and moved declarations must leave the row.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import _support
from cleverly.longitudinal import LTMLE, LongitudinalData, SurvivalDRTMLE
from cleverly.longitudinal import estimator as estimator_module
from tests.studies import canonical_known_node_mechanisms as smart
from tests.studies import survival_dr_tmle as survival

SMART_CONTRAST = "ate_regimen[always vs never]"
SURVIVAL_CONTRAST = f"ate_regimen[a1 vs a0 @ t={survival.law.HORIZON}]"
#: The declared factors of ``G51.declared``: the baseline treatment and retention at nodes 2 to 5.
SURVIVAL_DECLARED = frozenset(
    {"known_treatment@1", *(f"known_censoring@{node}" for node in range(2, 6))}
)

#: Every column of the SMART frame, the time-varying ones and the declared ones included.
SMART_RENAMES = {
    "L0": "severe",
    "A1": "first",
    "C1": "stayed",
    "L1": "responded",
    "A2": "second",
    "C2": "stayed_2",
    "Y": "remitted",
    "g1_0": "first_0",
    "g1_1": "first_1",
    "g2_0": "second_0",
    "g2_1": "second_1",
    "r1": "stay_chance",
    "r2": "stay_2",
}

#: Every column of the time-to-event frame. The censoring nodes are named from ``time``.
SURVIVAL_RENAMES = {
    "W1": "age",
    "W2": "score",
    "A": "arm",
    "time": "followup",
    "event": "died",
    "p1": "chance_1",
    "p0": "chance_0",
    "r1": "keep_1",
    "r2": "keep_2",
    "r3": "keep_3",
    "r4": "keep_4",
}


def _quiet(fit: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit()


def _smart_ltmle(**settings: Any) -> LTMLE:
    """The ``known-node-mechanisms`` study's estimator (``fit_cleverly``)."""
    learner = smart.ColumnQuasiBinomial(drop_last=True)
    base: dict[str, Any] = {
        "reference": smart.REGIMEN_REFERENCE,
        "outcome_learner": learner,
        "pseudo_learner": learner,
        "n_folds": 1,
        "g_bounds": smart.G_BOUNDS,
        "simultaneous": False,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    base.update(settings)
    return LTMLE({label: tuple(plan) for label, plan in smart.PLANS.items()}, **base)


def _smart_renamed(censoring: dict[str, str] | None = None, **settings: Any) -> Any:
    """The study's G21 case on a frame whose every column is renamed."""
    frame = smart.law.smart_sample(400, 0).rename(columns=SMART_RENAMES)
    return _quiet(
        lambda: _smart_ltmle(**settings).fit(
            frame,
            outcome="remitted",
            treatment=["first", "second"],
            baseline=["severe"],
            time_varying=[[], ["responded"]],
            censoring=["stayed", "stayed_2"],
            treatment_probabilities={
                "first": {0.0: "first_0", 1.0: "first_1"},
                "second": {0.0: "second_0", 1.0: "second_1"},
            },
            censoring_probabilities=(
                {"stayed": "stay_chance", "stayed_2": "stay_2"} if censoring is None else censoring
            ),
        )
    )


def _smart_original() -> Any:
    return _quiet(lambda: smart.fit_cleverly(smart.law.smart_sample(400, 0)))


def _keyed_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: key each declared factor by its column name, as before."""
    monkeypatch.setattr(estimator_module, "_known_nodes", estimator_module._known_factors)


class TestSmart:
    """``G21``: the two-node SMART with both treatments and both retentions declared."""

    def test_renaming_every_column_keeps_the_row_and_the_estimate(self) -> None:
        original, renamed = _smart_original(), _smart_renamed()
        assert original[SMART_CONTRAST].support.row == "G21"
        assert renamed[SMART_CONTRAST].support.row == "G21"
        assert renamed[SMART_CONTRAST].support.label == "supported"
        assert renamed[SMART_CONTRAST].psi == pytest.approx(original[SMART_CONTRAST].psi, abs=1e-12)
        assert renamed.config.support_key.qualifiers == original.config.support_key.qualifiers
        # The summary still prints the user's column names.
        assert renamed.config.known_factors == ("first", "stayed", "second", "stayed_2")

    def test_a_key_by_column_name_fails_the_rename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _keyed_by_name(monkeypatch)
        renamed = _smart_renamed()
        assert renamed[SMART_CONTRAST].support.row != "G21"
        assert renamed[SMART_CONTRAST].support.label == "experimental"

    def test_retention_declared_at_one_node_only_leaves_the_row(self) -> None:
        partial = _smart_renamed(
            censoring={"stayed": "stay_chance"},
            censoring_learner=LogisticRegression(max_iter=1000),
        )
        assert partial.config.support_key.qualifiers >= {
            "known_treatment@1",
            "known_treatment@2",
            "known_censoring@1",
        }
        assert "known_censoring@2" not in partial.config.support_key.qualifiers
        assert partial[SMART_CONTRAST].support.row != "G21"
        assert partial[SMART_CONTRAST].support.label == "experimental"

    def test_a_long_panel_declaration_reaches_the_row(self) -> None:
        """A long panel names its nodes ``"first@t"``, which no name-keyed row could match."""
        wide = smart.law.smart_sample(400, 0).rename(columns=SMART_RENAMES)
        unit = np.arange(len(wide))
        node_one = pd.DataFrame(
            {
                "unit": unit,
                "visit": 1,
                "severe": wide["severe"],
                "responded": np.nan,
                "treated": wide["first"],
                "stayed": wide["stayed"],
                "remitted": wide["remitted"],
                "chance_0": wide["first_0"],
                "chance_1": wide["first_1"],
                "stay_chance": wide["stay_chance"],
            }
        )
        kept = wide["stayed"] == 1.0
        node_two = pd.DataFrame(
            {
                "unit": unit[kept],
                "visit": 2,
                "severe": wide.loc[kept, "severe"],
                "responded": wide.loc[kept, "responded"],
                "treated": wide.loc[kept, "second"],
                "stayed": wide.loc[kept, "stayed_2"],
                "remitted": wide.loc[kept, "remitted"],
                "chance_0": wide.loc[kept, "second_0"],
                "chance_1": wide.loc[kept, "second_1"],
                "stay_chance": wide.loc[kept, "stay_2"],
            }
        )
        data = LongitudinalData.from_long_panel(
            pd.concat([node_one, node_two], ignore_index=True),
            unit="unit",
            time="visit",
            treatment="treated",
            baseline=["severe"],
            time_varying=["responded"],
            censoring="stayed",
            outcome="remitted",
            treatment_probabilities={0.0: "chance_0", 1.0: "chance_1"},
            censoring_probabilities="stay_chance",
        )
        panel = _quiet(lambda: _smart_ltmle().fit(data))
        original = _smart_original()
        assert panel[SMART_CONTRAST].support.row == "G21"
        assert panel[SMART_CONTRAST].psi == pytest.approx(original[SMART_CONTRAST].psi, abs=1e-9)


def _survival_fit(frame: pd.DataFrame, renamed: bool, scale: float = 1.0) -> Any:
    """The ``survival-dr-tmle`` study's ``known_1k`` fit, on the frame as named.

    ``scale`` multiplies the times, the grid, the declaration's grid keys and the horizon, so
    grid value and node index differ while the fit is otherwise the same.
    """
    names = SURVIVAL_RENAMES if renamed else {name: name for name in SURVIVAL_RENAMES}
    frame = frame.assign(time=frame["time"] * scale)
    data = LongitudinalData.from_time_to_event(
        frame.rename(columns=names),
        time=names["time"],
        event=names["event"],
        treatment=names["A"],
        baseline=[names["W1"], names["W2"]],
        grid=[scale * g for g in survival.law.GRID],
        treatment_probabilities={1: names["p1"], 0: names["p0"]},
        censoring_probabilities={
            scale * float(g): names[f"r{g}"] for g in range(1, survival.law.K)
        },
    )
    settings = {**survival.settings("known"), "horizons": [scale * survival.law.HORIZON]}
    return _quiet(lambda: SurvivalDRTMLE(survival.law.ARMS, **settings).fit(data))


class TestTimeToEvent:
    """``G51.declared``: a declared baseline treatment and the retention at nodes 2 to 5."""

    @staticmethod
    def frame() -> pd.DataFrame:
        return survival.law.sample(1_000, 0, trial=True)

    def test_renaming_every_column_keeps_the_row(self) -> None:
        frame = self.frame()
        original, renamed = _survival_fit(frame, False), _survival_fit(frame, True)
        assert original[SURVIVAL_CONTRAST].support.row == "G51.declared"
        assert renamed[SURVIVAL_CONTRAST].support.row == "G51.declared"
        assert renamed[SURVIVAL_CONTRAST].support.label == "supported"
        assert renamed[SURVIVAL_CONTRAST].psi == pytest.approx(
            original[SURVIVAL_CONTRAST].psi, abs=1e-12
        )
        assert renamed.config.support_key.qualifiers >= SURVIVAL_DECLARED
        assert renamed.config.known_factors[0] == "arm"

    def test_a_grid_that_is_not_the_node_index_keys_the_node(self) -> None:
        """On grid 2, 4, ..., 10 the node index and the grid value differ.

        Grid time ``g_j`` declares node ``j + 1``'s retention, so the declared nodes are 2 to 5
        while the generated names read ``followup>=4`` to ``followup>=10``. A key parsed from
        those names would hold ``known_censoring@4`` to ``@10`` and leave the row.
        """
        scaled = _survival_fit(self.frame(), True, scale=2.0)
        assert scaled.config.known_factors == (
            "arm",
            "followup>=4",
            "followup>=6",
            "followup>=8",
            "followup>=10",
        )
        declared = {
            name for name in scaled.config.support_key.qualifiers if name.startswith("known_")
        }
        assert declared == SURVIVAL_DECLARED
        # The horizon 10 is grid time 10, node 5, and the estimate is labelled by the node.
        assert scaled[SURVIVAL_CONTRAST].support.row == "G51.declared"
        assert scaled[SURVIVAL_CONTRAST].support.label == "supported"

    def test_a_key_by_column_name_fails_the_rename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _keyed_by_name(monkeypatch)
        renamed = _survival_fit(self.frame(), True)
        assert renamed[SURVIVAL_CONTRAST].support.row != "G51.declared"
        assert renamed[SURVIVAL_CONTRAST].support.label == "experimental"


class TestNuisanceLearners:
    """The learners a declared fit uses are chosen by column name, apart from the key."""

    @staticmethod
    def case(known: bool) -> tuple[LTMLE, LongitudinalData]:
        frame = smart.law.smart_sample(400, 0)
        # Twelve clusters put the fit on the ``t`` tier.
        frame["clinic"] = np.arange(len(frame)) % 12
        declared: dict[str, Any] = (
            {
                "treatment_probabilities": smart.TREATMENT,
                "censoring_probabilities": smart.CENSORING,
            }
            if known
            else {}
        )
        data = LongitudinalData.from_frame(frame, **smart.COLUMNS, id="clinic", **declared)
        estimator = LTMLE(
            {"always": 1, "never": 0},
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            # Adaptive, and unused when every factor is declared.
            treatment_learner=RandomForestClassifier(n_estimators=5, random_state=0),
            censoring_learner=LogisticRegression(max_iter=1000),
            n_folds=1,
            g_bounds=smart.G_BOUNDS,
            simultaneous=False,
            random_state=0,
        )
        return estimator, data

    @classmethod
    def key(cls, known: bool) -> _support.SupportKey:
        estimator, data = cls.case(known)
        return _support.longitudinal_key(
            estimator,
            data,
            outcome_kind="end_of_study",
            cross_fitted=False,
            known_factors=estimator_module._known_factors(data),
            known_nodes=estimator_module._known_nodes(data),
            has_msm=False,
            regimen_kinds={"static"},
        )[0]

    def test_an_unused_adaptive_treatment_learner_adds_no_qualifier(self) -> None:
        declared = self.key(known=True)
        assert declared.dependence == "t"
        assert _support.ADAPTIVE_NUISANCE not in declared.qualifiers
        # The nonzero witness: the same learner, estimated, is adaptive on this tier.
        assert _support.ADAPTIVE_NUISANCE in self.key(known=False).qualifiers

    def test_a_declared_fit_keys_no_adaptive_learner_it_never_fits(self) -> None:
        """The same check through ``fit``: a swap of the two arguments at the call site fails it."""
        estimator, data = self.case(known=True)
        key = _quiet(lambda: estimator.fit(data)).config.support_key
        assert key is not None and key.dependence == "t"
        assert _support.ADAPTIVE_NUISANCE not in key.qualifiers
        assert key.qualifiers >= {"known_treatment@1", "known_censoring@2"}
