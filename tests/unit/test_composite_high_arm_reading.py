"""The ``composite-high-arm`` reading and the composite study's exit counts, from committed rows.

The roadmap owner ``composite-high-arm`` and the study page
``observational-missing-data-dr-tmle.md`` quote every number below.  Each one is rebuilt here
from ``tests/canonical/drtmle_composite`` and the study's law, so an edited number fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.studies import composite_drtmle_properties as design
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.evidence.registry import ROOT

ARTIFACTS = ROOT / "tests" / "canonical" / "drtmle_composite"
PACKAGE = "cleverly-composite-drtmle"
REFERENCE = "drtmle-r-composite"
NAME = "ey[high]"


def _rows() -> pd.DataFrame:
    rows = pd.read_csv(ARTIFACTS / "replicates.csv.gz")
    return rows.loc[(rows["scenario"] == design.THREE_ARM) & (rows["estimand"] == NAME)]


def _exits() -> pd.DataFrame:
    """One row per primary DR-TMLE fit: the composite TMLE rows are excluded."""
    exits = pd.read_csv(ARTIFACTS / "fit-exits.csv")
    exits = exits.loc[~exits["estimand"].str.startswith("tmle_")]
    return exits.drop_duplicates(["scenario", "replicate"])


def test_the_primary_exit_counts_are_the_published_ones() -> None:
    """48 of 2,400 DR-TMLE fits reached the cap and 9 stalled; 47 and 8 on three arms."""
    exits = _exits()
    assert len(exits) == 2_400
    counts = exits["exit_reason"].value_counts()
    assert counts.to_dict() == {"tolerance": 2_343, "cap": 48, "stall": 9}
    three = exits.loc[exits["scenario"] == design.THREE_ARM, "exit_reason"].value_counts()
    assert (int(three["cap"]), int(three["stall"])) == (47, 8)


def test_the_red_pair_is_the_only_red_truth_row() -> None:
    tests = pd.read_csv(ARTIFACTS / "performance-tests.csv")
    red = tests.loc[~tests["passed"].astype(bool)]
    assert sorted(zip(red["implementation"], red["estimand"], strict=True)) == [
        (PACKAGE, NAME),
        (REFERENCE, NAME),
    ]
    assert len(tests) == 42
    lower = red.set_index("implementation")["coverage_ci_lower"]
    assert lower[PACKAGE] == pytest.approx(0.8949, abs=5e-5)
    assert lower[REFERENCE] == pytest.approx(0.8963, abs=5e-5)
    assert (lower < red["coverage_floor"].iloc[0]).all()


def test_the_composite_high_arm_reading() -> None:
    """Finite-sample and shared: the SE is right, and the capped fits cover better."""
    rows = _rows()
    by = rows.groupby("implementation")
    coverage = by["covered"].mean()
    assert coverage[PACKAGE] == pytest.approx(0.9225, abs=5e-5)
    assert coverage[REFERENCE] == pytest.approx(0.92375, abs=5e-5)

    paired = rows.pivot(index="replicate", columns="implementation", values="estimate")
    assert abs(float((paired[PACKAGE] - paired[REFERENCE]).mean())) < 1e-5

    # The variance estimator is right: the mean reported SE sits on the efficiency bound, and
    # the shortfall is the empirical spread, 1.066 times the bound.
    bound = laws.efficiency_sd(design._keyed(design.THREE_ARM), NAME) / np.sqrt(2_000)
    assert bound == pytest.approx(0.03437, abs=5e-6)
    mean_se = by["std_error"].mean()
    assert mean_se[PACKAGE] == pytest.approx(0.03424, abs=5e-6)
    assert mean_se[REFERENCE] == pytest.approx(0.03425, abs=5e-6)
    spread = by["estimate"].std()
    assert spread[PACKAGE] == pytest.approx(0.0367, abs=5e-5)
    assert spread[PACKAGE] / bound == pytest.approx(1.066, abs=5e-4)

    # The capped fits are not the cause: they cover more often than the rest.
    exits = _exits().loc[lambda frame: frame["scenario"] == design.THREE_ARM]
    joined = rows.merge(exits[["replicate", "exit_reason"]], on="replicate", validate="m:1")
    package = joined.loc[joined["implementation"] == PACKAGE].groupby("exit_reason")["covered"]
    assert package.mean()["cap"] == pytest.approx(0.957, abs=5e-4)
    assert package.mean()["tolerance"] == pytest.approx(0.921, abs=5e-4)
    assert package.size()["cap"] == 47

    # `high` has the law's smallest composite mechanism, where W = 0 holds half the mass.
    law = design.LAWS[design.THREE_ARM]
    column = law.labels.index("high")
    assert law.p_w[0] == pytest.approx(0.5)
    assert law.composite[0, column] == pytest.approx(0.070)
    assert law.composite[:, column].min() == law.composite.min()
