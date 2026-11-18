"""The PR 3 interval screen (``tests/studies/evidence/asymmetric_audit.py``).

The screen recomputes each committed difference whose arms its replicate records under the
package's MOVER interval, and reads every other row under the symmetric widening that contains
the skew-widened t.  These tests tie its MOVER arithmetic to the package's, check that the
pairing finds the arms of a known difference and refuses a broken identity, and recompute the
committed rows of the screen's MOVER and refit studies from the committed artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _bound_fit(binary: bool, *, cross_fit: bool = False) -> Any:
    """A small two-arm fit whose sensitivity bounds the recorder reads."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import ATE, CausalStudy, PointTreatment

    rng = np.random.default_rng(11)
    n = 400
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    y = (
        rng.binomial(1, 1 / (1 + np.exp(-(0.5 * a + w - 1)))).astype(float)
        if binary
        else 0.5 * a + w + rng.gamma(2.0, 1.0, n)
    )
    frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)))
    return study.identify(ATE()).estimate(
        outcome_learner=LogisticRegression() if binary else LinearRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=cross_fit,
        random_state=0,
    )


@pytest.mark.parametrize(("binary", "scale"), [(False, "skew_t"), (True, "mover")])
def test_the_recorder_reads_an_omitted_variable_bound_at_the_row_alpha(
    binary: bool, scale: str
) -> None:
    """A row at a bound end and its own standard error matches the bound's read at the row's
    alpha, whose shipped limits are the two-sided interval about the end; the bounds' own
    one-sided reads are at alpha 0.10.  Mutation control: the standard error read back off the
    limit with a normal quantile, which the harnesses reported before, matches nothing."""
    from cleverly.inference import interval_reference
    from cleverly.inference.cluster import influence_variance
    from cleverly.sensitivity import omitted_variable as ov
    from tests.studies.evidence import asymmetric_refit as refit

    fit = _bound_fit(binary)
    with refit.logging() as log:
        bounds = ov.omitted_variable_bounds(fit, "ate", cf_y=0.3, cf_d=0.2, rho=1.0)
    assert {round(float(entry["alpha"]), 12) for entry in log} == {0.05, 0.1}
    estimate = fit["ate"]
    assert estimate.interval_scale == scale
    elements = ov.sensitivity_elements(fit, "ate")
    curve = np.asarray(estimate.influence_curve) - bounds.confounding_strength * np.asarray(
        elements.psi_max_bias
    )
    se = float(np.sqrt(influence_variance(curve, None)))
    entry = refit.match({"estimate": bounds.lower, "std_error": se}, log)
    assert entry["scale"] == scale
    assert entry["alpha"] == pytest.approx(refit.ROW_ALPHA)
    shipped = interval_reference(
        estimate, level=0.95, psi=bounds.lower, std_error=se, influence_curve=curve
    )
    assert (entry["shipped_lower"], entry["shipped_upper"]) == pytest.approx(
        (shipped.lower, shipped.upper), rel=1e-12
    )
    z = stats.norm.ppf(0.975)
    assert (entry["normal_lower"], entry["normal_upper"]) == pytest.approx(
        (bounds.lower - z * se, bounds.lower + z * se), rel=1e-12
    )
    # Not vacuous: the shipped read is not the normal one on this sample.
    assert abs(entry["shipped_lower"] - entry["normal_lower"]) > 1e-6
    backed_out = (bounds.lower - bounds.ci_lower) / stats.norm.ppf(0.95)
    assert abs(backed_out - se) > 1e-8
    assert refit.match({"estimate": bounds.lower, "std_error": backed_out}, log) == {"scale": "own"}


def test_the_recorder_reads_a_marginal_bound_at_the_row_alpha() -> None:
    """A marginal-sensitivity row at an end and its reported standard error matches the
    bound's MOVER read; the bound's own read is already at the row's alpha."""
    from cleverly.sensitivity import marginal_sensitivity_bounds
    from tests.studies.evidence import asymmetric_refit as refit

    fit = _bound_fit(True, cross_fit=True)
    with refit.logging() as log:
        result = marginal_sensitivity_bounds(fit, (1.5,))
    (row,) = result.rows
    entry = refit.match({"estimate": row.lower, "std_error": row.lower_std_error}, log)
    assert entry["scale"] == "mover"
    assert entry["alpha"] == pytest.approx(refit.ROW_ALPHA)
    assert entry["shipped_lower"] == pytest.approx(row.lower_limit, rel=1e-12)
    assert abs(entry["shipped_lower"] - entry["normal_lower"]) > 1e-6


def test_a_bound_row_reads_only_its_own_alpha() -> None:
    """``match`` keeps a bound read at the row's alpha over one at another level, raises on a
    bound row with no read at its alpha, and matches every other entry as before."""
    from tests.studies.evidence import asymmetric_refit as refit

    def entry(alpha: float, scale: str, lower: float, source: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {"psi": 1.0, "std_error": 0.1, "scale": scale, "alpha": alpha,
                               "shipped_lower": lower}  # fmt: skip
        if source is not None:
            out["source"] = source
        return out

    row = {"estimate": 1.0, "std_error": 0.1}
    one_sided = entry(0.10, "skew_t", 0.8, "bound")
    two_sided = entry(0.05, "skew_t", 0.7, "bound")
    assert refit.match(row, [one_sided, two_sided])["shipped_lower"] == 0.7
    assert refit.match(row, [two_sided, one_sided])["shipped_lower"] == 0.7
    with pytest.raises(ValueError, match="alpha"):
        refit.match(row, [one_sided])
    # An estimate's own entry at another alpha is matched as it always was.
    own = entry(0.10, "identity", 0.6, None)
    assert refit.match(row, [own])["shipped_lower"] == 0.6
    assert refit.match(row, [own, one_sided, two_sided])["scale"] == "identity+skew_t"


def _replay_records(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """Mismatched scope records: ``(group, scale, estimate shift, standard-error shift)``
    against replicate 0 of :func:`_truth_group`."""
    from tests.studies.evidence import asymmetric_refit as refit

    committed = _truth_group(covered_at=0.97).rows.iloc[0]
    out = []
    for group, scale, shift, se_shift in rows:
        record: dict[str, Any] = dict.fromkeys(refit.RECORD_COLUMNS, np.nan)
        record.update(
            stage="scope",
            phase="primary",
            group=group,
            replicate=0,
            estimate=committed["estimate"] + shift,
            std_error=committed["std_error"] + se_shift,
            committed_estimate=committed["estimate"],
            committed_std_error=committed["std_error"],
            replay="mismatch",
            scale=scale,
        )
        out.append(record)
    return pd.DataFrame(out, columns=list(refit.RECORD_COLUMNS))


def test_a_replay_shift_inside_the_limit_is_accepted_and_one_outside_is_not() -> None:
    """Nonzero witness for the replay rule: the committed group's estimates spread by ``sd``,
    so a refit ``0.2 sd`` away is accepted and one ``0.3 sd`` away is unresolved, on either the
    estimate or the standard error.  An ``own`` row and a group the screen never judges are
    accepted at any shift, and a failed refit never is."""
    group = _truth_group(covered_at=0.97)
    groups = {group.key: group}
    sd = float(group.rows["estimate"].std(ddof=1))
    se_sd = float(group.rows["std_error"].std(ddof=1))
    assert sd > 0.0 and se_sd == 0.0  # a constant standard error: only an exact SE replays
    limit = screen_module.SHIFT_LIMIT
    inside = _replay_records([("s/ate", "skew_t", (limit - 0.05) * sd, 0.0)])
    assert screen_module.replay_shifts(inside, groups) == (1, [])
    outside = _replay_records([("s/ate", "mover", (limit + 0.05) * sd, 0.0)])
    assert screen_module.replay_shifts(outside, groups) == (0, ["s/ate@r0"])
    se_moved = _replay_records([("s/ate", "skew_t", 0.0, 1e-6)])
    assert screen_module.replay_shifts(se_moved, groups) == (0, ["s/ate@r0"])
    exempt = _replay_records([("s/ate", "own", 5.0 * sd, 0.1), ("s/other", "skew_t", 9.0, 0.0)])
    assert screen_module.replay_shifts(exempt, groups) == (2, [])
    failed = _replay_records([("s/ate", "failed", np.nan, np.nan)])
    assert screen_module.replay_shifts(failed, groups) == (0, ["s/ate@r0"])
    replayed = inside.assign(replay="ok", estimate=inside["estimate"] + 10.0)
    assert screen_module.replay_shifts(replayed, groups) == (0, [])


def test_an_unresolved_replay_keeps_the_study_waiting() -> None:
    unresolved = screen_module.RefitReading("mismatch:1", [], [], 3)
    assert screen_module._disposition("x", False, True, unresolved) == (
        "refits",
        "refit records: mismatch:1",
    )
    shifted = screen_module.RefitReading("ok", [], [], 2)
    disposition, reason = screen_module._disposition("dose-response", False, True, shifted)
    assert disposition == "neutral"
    assert "2 refit records" in reason and screen_module.REPLAY_CAUSES["dose-response"] in reason


def test_every_replay_cause_names_a_registered_study() -> None:
    slugs = {record.slug for record in registered()}
    assert set(screen_module.REPLAY_CAUSES) <= slugs


def test_every_committed_refit_record_names_a_scale_and_each_mismatch_is_judged(
    screen: pd.DataFrame,
) -> None:
    """Each committed scope or refit record names a scale, and the screen judges each
    mismatched record: accepted into ``replay_shifts``, or the study waits on
    ``mismatch:<k>`` (which :func:`test_no_study_awaits_its_refits` then fails)."""
    from tests.studies.evidence import asymmetric_refit as refit

    if not refit.AUDIT_DIR.exists():
        return
    by_study = screen.set_index("study")
    for directory in sorted(path for path in refit.AUDIT_DIR.iterdir() if path.is_dir()):
        missed = 0
        for name in (refit.SCOPE_FILE, refit.REFITS_FILE):
            records = refit.read_records(directory, name)
            if records is None:
                continue
            assert list(records.columns) == list(refit.RECORD_COLUMNS), directory.name
            assert records["scale"].astype(str).str.len().gt(0).all(), (directory.name, name)
            missed += int((records["replay"] == "mismatch").sum())
        row = by_study.loc[directory.name]
        if missed and not str(row["scope"]).startswith("mismatch:"):
            assert int(row["replay_shifts"]) > 0, directory.name


# ------------------------------------------------------------------ settling by a full refit


def _settle_records(group: screen_module.Group, lower_ratio: np.ndarray) -> pd.DataFrame:
    """Settle records of every replicate of ``group`` that replay it, each with the shipped
    lower side at ``lower_ratio`` times the normal half-width and the upper side normal."""
    from tests.studies.evidence import asymmetric_refit as refit

    rows = group.rows
    z = float(stats.norm.ppf(0.975))
    half = z * rows["std_error"].to_numpy(float)
    estimate = rows["estimate"].to_numpy(float)
    frame = pd.DataFrame(
        {
            "stage": "settle",
            "phase": "primary",
            "group": group.key,
            "replicate": rows["replicate"].to_numpy(int),
            "estimate": estimate,
            "std_error": rows["std_error"].to_numpy(float),
            "committed_estimate": estimate,
            "committed_std_error": rows["std_error"].to_numpy(float),
            "replay": "ok",
            "scale": "skew_t",
            "alpha": 0.05,
            "shipped_lower": estimate - np.broadcast_to(lower_ratio, estimate.shape) * half,
            "shipped_upper": estimate + half,
            "normal_lower": estimate - half,
            "normal_upper": estimate + half,
        }
    )
    return frame.reindex(columns=list(refit.RECORD_COLUMNS))


def _narrowing_study() -> tuple[screen_module._Study, dict[str, screen_module.Group], float]:
    record = interval_audit.record_for("canonical-stratified-tmle")
    group = _truth_group(covered_at=0.97)
    return (
        screen_module._Study(record, None, [group]),
        {group.key: group},
        float(stats.norm.ppf(0.975)),
    )


def test_a_settled_flip_that_holds_is_listed_settled_and_keeps_the_study_moved() -> None:
    """Every replicate's refitted lower side at 0.9 normal half-widths uncovers the rows at
    0.97, as the bound's narrow corner does: the verdict moves on the exact intervals alone."""
    study, groups, z = _narrowing_study()
    group = groups["s/ate"]
    exact = screen_module.settled_verdicts(_settle_records(group, np.array(0.9)), group, z)
    assert exact is not None
    flips = screen_module.bound_flips(study, groups, {}, z, {"s/ate": exact})
    assert flips == ["s/ate@settled"]
    reading = screen_module.RefitReading("ok", ["s/ate"], flips, 0, 1)
    disposition, reason = screen_module._disposition("x", False, True, reading)
    assert disposition == "moved"
    assert "settled by full refit of 1 group" in reason and "regenerate" in reason


def test_a_settled_flip_that_clears_makes_the_study_neutral() -> None:
    """The R_a bound narrows the lower side to 0.9 and flips the verdict at its corners; the
    refitted intervals narrow it to 0.99 only, which uncovers nothing."""
    study, groups, z = _narrowing_study()
    group = groups["s/ate"]
    assert screen_module.bound_flips(study, groups, {"s/ate": (0.9, 1.0, 1.0, 1.0)}, z)
    exact = screen_module.settled_verdicts(_settle_records(group, np.array(0.99)), group, z)
    assert exact is not None
    assert screen_module.bound_flips(study, groups, {}, z, {"s/ate": exact}) == []
    reading = screen_module.RefitReading("ok", ["s/ate"], [], 0, 1)
    disposition, reason = screen_module._disposition("x", False, True, reading)
    assert disposition == "neutral" and reason.startswith("settled by full refit")


def test_the_settled_judgement_reads_each_replicates_own_interval() -> None:
    """Mutation control: the narrowing sits only on the replicates at 0.97, and no uniform
    ratio a corner applies reproduces it.  Judged per replicate it flips the floor; with the
    records' replicates rotated by one, so each interval lands on another row, it does not,
    and the records with every other replicate dropped settle nothing."""
    study, groups, z = _narrowing_study()
    group = groups["s/ate"]
    far = (group.rows["estimate"] - group.rows["truth"]).to_numpy() > 0.6 * 1.959964 * 0.1
    ratio = np.where(far, 0.9, 1.0)
    records = _settle_records(group, ratio)
    exact = screen_module.settled_verdicts(records, group, z)
    assert exact is not None
    assert int((exact[0] == 0).sum()) == int(far.sum())
    assert screen_module.bound_flips(study, groups, {}, z, {"s/ate": exact}) == ["s/ate@settled"]
    rotated = screen_module.settled_verdicts(_settle_records(group, np.roll(ratio, 1)), group, z)
    assert rotated is not None and not np.array_equal(rotated[0], exact[0])
    assert screen_module.bound_flips(study, groups, {}, z, {"s/ate": rotated}) == []
    assert screen_module.settled_verdicts(records.iloc[::2], group, z) is None


def test_a_settle_record_off_the_committed_interval_does_not_settle() -> None:
    """The substitution needs the committed ``covered`` to be the refit's normal interval: a
    group whose committed run read a wider interval (here 1.1 normal half-widths, so the rows
    at 1.05 are covered) stays on its bound."""
    group = _truth_group(covered_at=1.05)
    rows = group.rows.copy()
    z = float(stats.norm.ppf(0.975))
    rows["covered"] = (
        (rows["estimate"] - rows["truth"]).abs() <= 1.1 * z * rows["std_error"]
    ).astype(int)
    wider = screen_module.Group(group.key, group.kind, group.pool, rows, group.index)
    assert screen_module.settled_verdicts(_settle_records(wider, np.array(1.0)), wider, z) is None
    assert screen_module.settled_verdicts(_settle_records(group, np.array(1.0)), group, z)


def test_an_accepted_settle_mismatch_is_placed_on_the_committed_row() -> None:
    """A refit that missed its committed row inside the replay rule keeps its own ratio, moved
    to the committed estimate and scaled by the committed standard error; an ``own`` record
    keeps the committed ``covered`` whatever its limits."""
    _, groups, z = _narrowing_study()
    group = groups["s/ate"]
    records = _settle_records(group, np.array(1.0))
    first = records.index[0]
    records.loc[first, ["estimate", "shipped_lower", "normal_lower", "shipped_upper",
                        "normal_upper"]] += 0.01  # fmt: skip
    records.loc[first, "replay"] = "mismatch"
    exact = screen_module.settled_verdicts(records, group, z)
    assert exact is not None
    assert np.array_equal(exact[0], group.rows["covered"].to_numpy(int))
    owned = _settle_records(group, np.array(0.5)).assign(scale="own")
    kept = screen_module.settled_verdicts(owned, group, z)
    assert kept is not None and np.array_equal(kept[0], group.rows["covered"].to_numpy(int))


def test_the_refit_reading_settles_from_the_committed_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end on record files: the R_a refits narrow the lower side to 0.9 and the study
    reads ``moved`` at the corners; a ``settle.csv`` whose every replicate narrows it to 0.99
    clears the flip and settles the group."""
    from tests.studies.evidence import asymmetric_refit as refit

    study, groups, z = _narrowing_study()
    group = groups["s/ate"]
    directory = tmp_path / study.record.slug
    directory.mkdir()
    monkeypatch.setattr(refit, "AUDIT_DIR", tmp_path)
    head = _settle_records(group, np.array(1.0)).assign(stage="scope").iloc[:1]
    head.to_csv(directory / refit.SCOPE_FILE, index=False)
    ratios = np.linspace(0.9, 1.0, refit.REFIT_REPLICATES)
    some = _settle_records(group, np.ones(len(group.rows)))
    some = some.iloc[: refit.REFIT_REPLICATES].assign(stage="refits")
    some["shipped_lower"] = some["estimate"] - ratios * (some["estimate"] - some["normal_lower"])
    some.to_csv(directory / refit.REFITS_FILE, index=False)
    corners = screen_module._refit_reading(study, groups, z, needed=["s/ate"], exact=set())
    assert corners.scope == "ok" and corners.settled == 0
    assert any(flip.startswith("s/ate@lo") for flip in corners.flips), corners.flips
    _settle_records(group, np.array(0.99)).to_csv(directory / refit.SETTLE_FILE, index=False)
    settled = screen_module._refit_reading(study, groups, z, needed=["s/ate"], exact=set())
    assert (settled.scope, settled.flips, settled.settled) == ("ok", [], 1)
    _settle_records(group, np.array(0.9)).to_csv(directory / refit.SETTLE_FILE, index=False)
    held = screen_module._refit_reading(study, groups, z, needed=["s/ate"], exact=set())
    assert (held.flips, held.settled) == (["s/ate@settled"], 1)


def test_the_settle_stage_refits_every_replicate_of_each_flagged_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settle stage names the screen's flagged groups (a name that holds ``@`` keeps it)
    and selects every replicate of each, where the R_a stage selects the first 20."""
    from tests.studies.evidence import asymmetric_refit as refit

    screen = pd.DataFrame(
        {
            "study": ["a", "b", "c"],
            "disposition": ["moved", "neutral", "moved"],
            "bound_flips": ["s/x@lo-lo;s/x@hi-hi;L4/y[1 vs 0 @ t=5]@unbounded", "", ""],
        }
    )
    assert refit.settle_groups("a", screen) == {"s/x", "L4/y[1 vs 0 @ t=5]"}
    assert refit.settle_groups("b", screen) == set()
    assert refit.settle_targets(screen) == ["a"]
    later = pd.concat(
        [
            screen,
            pd.DataFrame(
                {
                    "study": ["d", "cde-tmle"],
                    "disposition": ["moved", "moved"],
                    "bound_flips": ["s/x@settled;s/y@lo-lo", "s/z@lo-lo"],
                }
            ),
        ]
    )
    assert "cde-tmle" in screen_module.REGENERATE
    assert refit.settle_targets(later) == ["a"]  # moved by a full refit, or regenerated
    keys = [(cell, replicate) for cell in ("s", "t") for replicate in range(50)]
    job = refit.Job(
        "primary", "primary", keys, [], lambda key, drawn: None, 2,
        lambda key: key[0], lambda key: key[1], None, None,
        lambda key: {f"{key[0]}/x"},
    )  # fmt: skip
    every = refit.select(job, replicates=None, names={"s/x"})
    assert every == list(range(50))
    assert refit.select(job, replicates=refit.REFIT_REPLICATES, names={"s/x"}) == list(range(20))
    monkeypatch.setattr(refit, "settle_groups", lambda slug: {"s/x"})
    assert refit.stage_selection("a", "settle") == ({"s/x"}, None)
    assert refit.stage_selection("a", "scope") == (None, 1)


def test_a_key_of_several_scenarios_writes_each_ones_groups() -> None:
    """A primary key that fits two scenarios on one draw names its cell ``a;b``; it writes
    the groups of both, so the scope stage selects it (the ``missing:<k>`` of ``cde-tmle``)."""
    from tests.studies.evidence import asymmetric_refit as refit

    names = refit._ScenarioGroups(
        lambda key: ";".join(key[0]), {"z0": ("ey0", "ate"), "z1": ("ey0",)}
    )
    assert names((("z0", "z1"), 0, 2000)) == {"z0/ey0", "z0/ate", "z1/ey0"}
    assert names((("z1",), 0, 2000)) == {"z1/ey0"}


def test_a_key_writes_the_reported_scenarios_its_study_owner_names() -> None:
    """``mtp-policy-variance`` publishes ``<twin>_policy`` from the twin's own fit, and no key
    has that cell; the study's ``owner`` maps it back, so the refits select the twin's keys
    (the ``refits:4`` of its four ``*_policy`` differences).  Without the mapping a key
    writes its own scenario only."""
    from tests.studies.evidence import asymmetric_refit as refit
    from tests.studies.evidence import band_rerun

    scenarios = {"t": ("ey",), "t_policy": ("ate",), "u": ("ey",)}
    owners = refit.owned_scenarios(scenarios, lambda name: name.removesuffix("_policy"))
    assert owners == {"t": ("t", "t_policy"), "u": ("u",)}
    names = refit._ScenarioGroups(lambda key: key[0], scenarios, owners)
    assert names(("t", 0)) == {"t/ey", "t_policy/ate"}
    assert refit._ScenarioGroups(lambda key: key[0], scenarios)(("t", 0)) == {"t/ey"}
    record = band_rerun.record_for("mtp-policy-variance")
    (job,) = refit.capture_primary(record)
    written = set().union(*(job.names_of(key) for key in job.keys))
    declared = {f"{s}/{e}" for s, estimands in record.scenarios.items() for e in estimands}
    assert declared <= written, sorted(declared - written)
    wanted = {"atom_ltmle_tail_policy/ate_regimen[p5 vs natural]"}
    chosen = refit.select(job, replicates=refit.REFIT_REPLICATES, names=wanted)
    assert {job.cell_of(job.keys[i]) for i in chosen} == {"atom_ltmle_tail"}
    assert len(chosen) == refit.REFIT_REPLICATES


def _power_group(n: int = 400) -> screen_module.Group:
    """A power cell: ``covered`` is the coverage of the null 0 (``1 - rejected``), so it is not
    the identity interval's coverage of the truth, and a share of replicates sits just past
    the rejection edge."""
    rng = np.random.default_rng(11)
    z = 1.959964
    se = np.full(n, 0.01)
    statistic = np.where(rng.uniform(size=n) < 0.1, 2.0, rng.uniform(2.5, 5.0, size=n))
    estimate = statistic * se
    rejected = (np.abs(estimate) / se > z).astype(float)
    rows = pd.DataFrame(
        {
            "replicate": np.arange(n),
            "estimate": estimate,
            "std_error": se,
            "truth": np.full(n, 0.035),
            "covered": (1.0 - rejected).astype(int),
            "rejected": rejected,
        }
    ).set_index("replicate", drop=False)
    return screen_module.Group("power/wald__c", "property", "", rows, rows.index)


def test_a_power_cell_settles_on_its_rejections_against_zero() -> None:
    """Witness: a cell whose ``covered`` reads the null 0 is not identity-consistent, yet its
    rejections are the normal test of 0.  Widening the lower side by 5% un-rejects the
    replicates at 2.0 standard errors, and the settled ``covered`` is ``1 - rejected``."""
    group = _power_group()
    z = float(stats.norm.ppf(0.975))
    assert not screen_module._identity_consistent(group, z)
    assert screen_module._null_coverage(group, z)
    near = (group.rows["estimate"] / group.rows["std_error"]).to_numpy() < 2.1
    exact = screen_module.settled_verdicts(_settle_records(group, np.array(1.05)), group, z)
    assert exact is not None
    covered, rejected = exact
    assert rejected is not None
    assert np.array_equal(rejected, np.where(near, 0.0, 1.0))
    assert np.array_equal(covered, 1 - rejected.astype(int))
    same = screen_module.settled_verdicts(_settle_records(group, np.array(1.0)), group, z)
    assert same is not None and np.array_equal(same[0], group.rows["covered"].to_numpy(int))


def test_a_cell_whose_covered_is_not_one_minus_rejected_stays_unbounded() -> None:
    """Mutation control: one replicate's ``covered`` no longer equals ``1 - rejected``; the
    cell reads neither the truth nor the null, so its refit does not settle it."""
    group = _power_group()
    z = float(stats.norm.ppf(0.975))
    rows = group.rows.copy()
    first = rows.index[0]
    rows.loc[first, "covered"] = 1 - int(rows.loc[first, "covered"])
    broken = screen_module.Group(group.key, group.kind, group.pool, rows, group.index)
    assert not screen_module._null_coverage(broken, z)
    assert (
        screen_module.settled_verdicts(_settle_records(broken, np.array(1.05)), broken, z) is None
    )
