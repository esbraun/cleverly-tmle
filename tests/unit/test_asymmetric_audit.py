"""The PR 3 interval screen (``tests/studies/evidence/asymmetric_audit.py``).

The screen recomputes each committed difference whose arms its replicate records under the
package's MOVER interval, and reads every other row under the symmetric widening that contains
the skew-widened t.  These tests tie its MOVER arithmetic to the package's, check that the
pairing finds the arms of a known difference and refuses a broken identity, and recompute the
committed rows of the screen's MOVER and refit studies from the committed artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.inference import implied_correlation, mover_ci
from tests.studies.evidence import asymmetric_audit as screen_module
from tests.studies.evidence import interval_audit
from tests.studies.evidence.registry import registered


def _logit_interval(p: float, se: float) -> tuple[float, float]:
    low, high = interval_audit.logit_interval(np.array([p]), np.array([se]))
    return float(low[0]), float(high[0])


def test_the_screen_mover_is_the_package_mover() -> None:
    """Over random arms and correlations the vector rule equals ``mover_ci`` on the arms' logit
    intervals with ``implied_correlation``, to 1e-12."""
    rng = np.random.default_rng(20261113)
    for _ in range(200):
        p1, p0 = rng.uniform(0.02, 0.98, size=2)
        s1, s0 = rng.uniform(0.005, 0.08, size=2)
        r = rng.uniform(-0.6, 0.9)
        sd = float(np.sqrt(s1**2 + s0**2 - 2 * r * s1 * s0))
        low, high = screen_module.mover_interval(p1 - p0, p1, s1, p0, s0, sd)
        expected = mover_ci(
            p1,
            _logit_interval(p1, s1),
            p0,
            _logit_interval(p0, s0),
            correlation=implied_correlation(sd**2, s1**2, s0**2),
        )
        assert float(low) == pytest.approx(expected[0], abs=1e-12)
        assert float(high) == pytest.approx(expected[1], abs=1e-12)


def test_dropping_the_correlation_term_moves_the_limits() -> None:
    """Mutation control: with ``r = 0.6`` the uncorrelated rule gives other limits."""
    s1, s0, r = 0.04, 0.03, 0.6
    sd = float(np.sqrt(s1**2 + s0**2 - 2 * r * s1 * s0))
    kept = screen_module.mover_interval(0.4, 0.7, s1, 0.3, s0, sd)
    dropped = screen_module.mover_interval(0.4, 0.7, s1, 0.3, s0, sd, correlated=False)
    assert abs(float(kept[0]) - float(dropped[0])) > 0.01
    assert abs(float(kept[1]) - float(dropped[1])) > 0.01


def test_an_arm_on_its_bound_has_no_mover_interval() -> None:
    low, high = screen_module.mover_interval(1.0, 1.0, 0.01, 0.0, 0.01, 0.01)
    assert np.isnan(low) and np.isnan(high)


def _groups(slug: str) -> list[screen_module.Group]:
    study = screen_module._load(interval_audit.record_for(slug))
    return [group for group in study.groups if group.kind == "truth"]


def test_the_pairing_finds_the_arms_of_a_stratum_effect() -> None:
    groups = _groups("canonical-stratified-tmle")
    pairs = screen_module.pair(groups)
    assert pairs["stratified_binary/ate[V=2]"] == (
        "stratified_binary/ey[1][V=2]",
        "stratified_binary/ey[0][V=2]",
    )
    assert not any(key.split("/", 1)[1].startswith("ey[") for key in pairs)


def test_the_pairing_refuses_a_broken_identity() -> None:
    """Nonzero witness: moving one replicate's arm estimate by 1e-6 unpairs the difference."""
    groups = _groups("canonical-stratified-tmle")
    broken = []
    for group in groups:
        if group.key == "stratified_binary/ey[1][V=2]":
            rows = group.rows.copy()
            rows.iloc[5, rows.columns.get_loc("estimate")] += 1e-6
            group = screen_module.Group(group.key, group.kind, group.pool, rows, group.index)
        broken.append(group)
    assert "stratified_binary/ate[V=2]" not in screen_module.pair(broken)


def test_the_pairing_refuses_an_impossible_correlation() -> None:
    """A difference whose standard error exceeds the arms' sum implies ``|r| > 1``."""
    rows = pd.DataFrame(
        {
            "replicate": [0, 1],
            "estimate": [0.6, 0.5],
            "std_error": [0.01, 0.01],
            "truth": [0.5, 0.5],
            "covered": [1, 1],
        }
    ).set_index("replicate", drop=False)
    arm1 = screen_module.Group("s/ey[1]", "truth", "s", rows, rows.index)
    arm0_rows = rows.assign(estimate=[0.3, 0.2])
    arm0 = screen_module.Group("s/ey[0]", "truth", "s", arm0_rows, rows.index)
    good = rows.assign(estimate=[0.3, 0.3], std_error=[0.012, 0.012])
    bad = rows.assign(estimate=[0.3, 0.3], std_error=[0.05, 0.05])
    paired = screen_module.pair(
        [arm1, arm0, screen_module.Group("s/ate", "truth", "s", good, rows.index)]
    )
    assert paired == {"s/ate": ("s/ey[1]", "s/ey[0]")}
    unpaired = screen_module.pair(
        [arm1, arm0, screen_module.Group("s/ate", "truth", "s", bad, rows.index)]
    )
    assert unpaired == {}


# ------------------------------------------------------------------ the committed screen


@pytest.fixture(scope="module")
def screen() -> pd.DataFrame:
    return screen_module.read_screen()


def test_the_screen_lists_every_registered_study_once(screen: pd.DataFrame) -> None:
    assert list(screen.columns) == list(screen_module.SCREEN_COLUMNS)
    assert sorted(screen["study"]) == sorted(record.slug for record in registered())
    assert screen["study"].is_unique


def test_every_mover_flip_is_explained_or_moved(screen: pd.DataFrame) -> None:
    flipped = screen.loc[screen["mover_flips"] != ""]
    for row in flipped.itertuples():
        if row.study in screen_module.SCREEN_REASONS:
            assert row.disposition == "declared" and row.reason
        else:
            assert row.disposition == "moved", row.study
    assert set(screen_module.SCREEN_REASONS) <= set(flipped["study"])
    assert set(screen.loc[screen["disposition"] == "current", "study"]) == set(
        screen_module.CURRENT
    )


#: Studies whose screen row the fast tier recomputes: each study with a MOVER flip, and a
#: binary-outcome study whose differences all pair.
_RECOMPUTED = ("canonical-stratified-tmle",)


@pytest.mark.parametrize("slug", _RECOMPUTED)
def test_each_recomputed_screen_row_matches_the_committed_rows(
    screen: pd.DataFrame, slug: str
) -> None:
    """A new cell or a regenerated study changes its row; refresh with the module's command."""
    committed_row = screen.loc[screen["study"] == slug].iloc[0].to_dict()
    again = screen_module.screen_row(interval_audit.record_for(slug))
    assert committed_row.pop("max_coverage_change") == pytest.approx(
        again.pop("max_coverage_change"), abs=1e-6
    )
    assert committed_row == again


# ------------------------------------------------------------------ the refit audit (B1, M1)


def test_no_study_awaits_its_refits(screen: pd.DataFrame) -> None:
    """Plan 9.8: every study the shipped intervals may move is settled by its refit records
    before the default ships.  Run the runner brief's commands, commit
    ``tests/canonical/asymmetric-refit/``, and refresh the screen."""
    waiting = screen.loc[screen["disposition"] == "refits", "study"].tolist()
    assert waiting == [], f"{len(waiting)} studies await their refit audit: {waiting}"


def test_an_unpaired_difference_is_never_neutral_on_the_screen_alone(
    screen: pd.DataFrame,
) -> None:
    """MOVER can narrow one side, so no widening settles an unpaired difference (M1)."""
    neutral = screen.loc[(screen["unpaired"] != "") & (screen["disposition"] == "neutral")]
    assert (neutral["scope"] == "ok").all(), neutral["study"].tolist()
    absent = screen_module.RefitReading("absent", [], [])
    assert screen_module._disposition("x", False, True, absent)[0] == "refits"
    settled = screen_module.RefitReading("ok", [], [])
    assert screen_module._disposition("x", False, True, settled)[0] == "neutral"
    moved = screen_module.RefitReading("ok", ["s/ate"], ["s/ate@lo-lo"])
    assert screen_module._disposition("x", False, True, moved)[0] == "moved"


def _truth_group(covered_at: float, n: int = 400) -> screen_module.Group:
    """A truth group whose estimates sit at ``covered_at`` normal half-widths from the truth
    on every replicate but a share, so the coverage floor reads near its edge."""
    rng = np.random.default_rng(5)
    se = np.full(n, 0.1)
    distance = np.where(rng.uniform(size=n) < 0.94, 0.5, covered_at) * 1.959964 * se
    estimate = 0.5 + distance
    rows = pd.DataFrame(
        {
            "replicate": np.arange(n),
            "estimate": estimate,
            "std_error": se,
            "truth": np.full(n, 0.5),
            "covered": (distance <= 1.959964 * se).astype(int),
        }
    ).set_index("replicate", drop=False)
    return screen_module.Group("s/ate", "truth", "s", rows, rows.index)


def test_the_bound_judges_a_narrowing_side() -> None:
    """A MOVER side that narrows can push coverage under the floor; the bound's narrow corner
    finds it, and judging the widening corners alone (the symmetric screen) would not."""
    record = interval_audit.record_for("canonical-stratified-tmle")
    group = _truth_group(covered_at=0.97)
    study = screen_module._Study(record, None, [group])
    groups = {group.key: group}
    z = float(stats.norm.ppf(0.975))
    narrowing = {"s/ate": (0.9, 1.0, 1.0, 1.0)}
    flips = screen_module.bound_flips(study, groups, narrowing, z)
    assert any(flip.startswith("s/ate@lo") for flip in flips), flips
    widening_only = {"s/ate": (1.0, 1.1, 1.0, 1.1)}
    assert screen_module.bound_flips(study, groups, widening_only, z) == []


def test_the_side_bounds_inflate_the_spread_and_hold_one() -> None:
    from tests.studies.evidence import asymmetric_refit as refit

    psi = np.zeros(refit.REFIT_REPLICATES)
    lower_ratio = np.linspace(0.9, 1.0, refit.REFIT_REPLICATES)
    upper_ratio = np.linspace(1.0, 1.2, refit.REFIT_REPLICATES)
    records = pd.DataFrame(
        {
            "estimate": psi,
            "shipped_lower": -lower_ratio,
            "normal_lower": -np.ones_like(psi),
            "shipped_upper": upper_ratio,
            "normal_upper": np.ones_like(psi),
        }
    )
    bounds = screen_module.side_bounds(records)
    assert bounds is not None
    low_q = float(np.quantile(lower_ratio, 1 - refit.QUANTILE))
    high_q = float(np.quantile(upper_ratio, refit.QUANTILE))
    assert bounds[0] == pytest.approx(1 - refit.INFLATION * (1 - low_q))
    assert bounds[1] == 1.0
    assert bounds[2] == 1.0
    assert bounds[3] == pytest.approx(1 + refit.INFLATION * (high_q - 1))
    assert screen_module.side_bounds(records.iloc[:-1]) is None


def test_the_recorder_names_each_interval_a_fit_reads() -> None:
    """The scope refit's recorder logs the estimates whose interval a study reads and matches a
    written row to its scale: the skew-widened t on a continuous outcome, MOVER on a binary
    difference, and ``own`` for a row whose interval the study formed itself."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import ATE, CausalStudy, PointTreatment
    from tests.studies.evidence import asymmetric_refit as refit

    rng = np.random.default_rng(3)
    n = 400
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    for binary, scale in ((False, "skew_t"), (True, "mover")):
        y = (
            rng.binomial(1, 1 / (1 + np.exp(-(0.5 * a + w - 1)))).astype(float)
            if binary
            else 0.5 * a + w + rng.gamma(2.0, 1.0, n)
        )
        frame = pd.DataFrame({"W": w, "A": a, "Y": y})
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        fit = study.identify(ATE()).estimate(
            outcome_learner=LogisticRegression() if binary else LinearRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )
        estimate = fit["ate"]
        with refit.logging() as log:
            low, high = estimate.ci
        row = {"estimate": estimate.psi, "std_error": estimate.std_error}
        entry = refit.match(row, log)
        assert entry["scale"] == scale
        assert (entry["shipped_lower"], entry["shipped_upper"]) == pytest.approx((low, high))
        assert refit.match(row, []) == {"scale": "own"}


def test_every_committed_refit_record_replays() -> None:
    """Each committed scope or refit record held its estimate and standard error to the
    committed row (the replay rule), and names a scale."""
    from tests.studies.evidence import asymmetric_refit as refit

    if not refit.AUDIT_DIR.exists():
        return
    for directory in sorted(path for path in refit.AUDIT_DIR.iterdir() if path.is_dir()):
        for name in (refit.SCOPE_FILE, refit.REFITS_FILE):
            records = refit.read_records(directory, name)
            if records is None:
                continue
            assert list(records.columns) == list(refit.RECORD_COLUMNS), directory.name
            assert (records["replay"] != "mismatch").all(), (directory.name, name)
            assert records["scale"].astype(str).str.len().gt(0).all(), (directory.name, name)
