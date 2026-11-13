"""The PR 3 interval screen: committed rows read under correlated MOVER and the skew-widened t.

The package forms the interval of an unclustered bounded difference of two bounded means of one
fit by correlated MOVER on the two means' logit intervals, and the interval of an unclustered
estimate of a continuous outcome by the skew-widened t (``docs/technical-reference/inference.md``,
"Small-sample scale").  The committed rows of the registered studies were written under the
identity normal interval of a difference and the normal interval of a continuous estimate.
:mod:`tests.studies.evidence.interval_audit` screens the bounded means; this module screens the
two classes PR 3 moved, one row per registered study in :data:`SCREEN_FILE`.

**Differences: recomputed exactly.**  MOVER is a deterministic function of the two arm estimates
and standard errors and of the difference's standard error: the correlation is
``(s_1^2 + s_0^2 - s_d^2) / (2 s_1 s_0)``, exact where the difference's curve is the difference of
its arm curves.  So a committed difference whose replicate also records both arm rows is
recomputed with no refit.  The screen finds the arms by the estimate identity: a group ``D`` is a
paired difference when two groups ``A`` and ``B`` of the same replicates, every estimate inside
``(0, 1)``, give ``d = a - b`` to :data:`IDENTITY_TOLERANCE` on every replicate and an implied
correlation inside ``[-1, 1]``, and ``D``'s name does not read a level (``b = a - d`` holds too).  That is the identity :func:`cleverly.inference.with_components`
checks before it records the arms, so the pairing is the package's.  A paired group whose
committed ``covered`` is not the plain identity interval (a t or cluster reference, or a margin)
is outside MOVER's scope, as the package keeps a clustered estimate on its own reference, and is
counted in ``out_of_scope``.  A difference with no recorded arms (a truth row of a difference
head, or a property cell whose name reads ``ate``, that pairs with nothing) is counted in
``unpaired``: the screen cannot recompute it, and the runner brief of PR 3 says which of them
need refits.  Each moved group is judged as the bounded-mean screen judges one: a truth group
flips when its exact coverage interval changes side of the coverage floor, and a property family
with a crossing is re-judged by the study's own ``summarize_properties``
(:func:`tests.studies.evidence.interval_audit._moved_verdicts`), with ``rejected`` recomputed
against the null 0 where the committed ``rejected`` reads it.

**Continuous estimates: a widening screen.**  No committed study outside ``small-sample-wald``
and ``asymmetric-wald`` records the degrees of freedom or the curve's acceleration that the
skew-widened t reads, so these rows cannot be recomputed.  The skew-widened interval contains the
normal interval on every sample (``max(t, t - aK) >= t >= z`` on each side), so coverage can only
rise and rejections only fall, and it lies inside the symmetric interval of half-width
``z f se`` once ``f >= (t + |a| K) / z``.  The screen reads every group that is not a paired difference and whose
committed ``covered`` is the plain normal interval (a superset: it cannot tell a continuous
outcome from a binary one) at each factor of :data:`FACTORS`, and names each verdict that moves
with the smallest factor that moves it.  A verdict that moves at no factor is neutral for any
skew-widened interval inside the largest one.  A verdict that moves needs the refits of plan 9.8
(the smallest df and the 99% quantile of ``|a| K`` over ``R_a`` refits) to say whether its study
is neutral or moved; a verdict that reads only the coverage floor or a type I error cannot move
against the study, because both readings only improve.

Refresh the screen with::

    python -m tests.studies.evidence.asymmetric_audit --jobs 4
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from tests.studies.evidence import interval_audit
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

#: The screen, one row per registered study.
SCREEN_FILE = ROOT / "tests" / "canonical" / "asymmetric-audit-screen.csv"

#: The columns of the screen, in order.
SCREEN_COLUMNS = (
    "study",
    "disposition",
    "paired_groups",
    "out_of_scope",
    "moved_groups",
    "max_coverage_change",
    "mover_flips",
    "unpaired",
    "widening_groups",
    "widening_flips",
    "reason",
)

#: The absolute tolerance of the estimate identity ``d = a - b`` that pairs a difference.
IDENTITY_TOLERANCE = 1e-9

#: The symmetric widening factors of the continuous screen, as multiples of the normal
#: half-width.  ``t`` at 250 df is 1.0062 times ``z`` and at 30 df 1.0396 times; the widening
#: ``|a| K`` adds about ``2 z^2 + 1 = 8.7`` times the curve's acceleration.
FACTORS = (1.02, 1.05, 1.10, 1.25)

#: The truth-row heads of a bounded difference: the built-in difference targets and the
#: longitudinal contrast.  A truth group of such a head that pairs with no arms is ``unpaired``.
DIFFERENCE_HEADS = frozenset(
    {"ate", "atc", "att", "ate_regime", "ate_learned_rule", "ate_ipsi", "ate_policy"}
    | {"ate_rr_tilt", "ate_regimen", "risk_difference"}
)

#: A property cell whose name reads a treatment-effect difference.
_DIFFERENCE_CELL = re.compile(r"(^|_)(ate|att|atc)(_|$|\[)")
#: A property cell whose name reads a level: never a difference, since ``b = a - d`` holds
#: whenever ``d = a - b`` does.
_LEVEL_CELL = re.compile(r"(^|_)(ey\d*|ey_[a-z_]+|risk|cif|survival)(_|$|\[)")

#: Studies the screen does not recompute, and why: their committed rows already hold the
#: shipped interval, or hold a candidate rule as the record of what a study measured.
CURRENT: dict[str, str] = {
    "asymmetric-wald": "its rows carry the shipped rules: MOVER, the logit interval and the "
    "skew-widened t",
    "small-sample-wald": "its rows hold the rule it measured, the record under "
    "F3-normal-reference, not a shipped interval",
}

#: Why a study holds no verdict the shipped intervals move, for each study whose screen row
#: records a MOVER flip.
SCREEN_REASONS: dict[str, str] = {}


@dataclass(frozen=True)
class Group:
    """One row group: a truth group (scenario, estimand) or a positive property cell.

    Parameters
    ----------
    key : str
        ``scenario/estimand`` or ``property/cell``.
    kind : str
        ``"truth"`` or ``"property"``.
    pool : str
        The rows a pairing may read: the scenario of a truth group, or ``""`` for every
        positive property cell.
    rows : pandas.DataFrame
        The committed rows, indexed by replicate.
    index : pandas.Index
        The rows' index in the committed frame, in the order of ``rows``.
    """

    key: str
    kind: str
    pool: str
    rows: pd.DataFrame
    index: pd.Index


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def mover_interval(
    d: Any,
    p1: Any,
    s1: Any,
    p0: Any,
    s0: Any,
    sd: Any,
    *,
    alpha: float = 0.05,
    correlated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated MOVER on the arms' logit normal intervals, vectorized; ``nan`` off ``(0, 1)``.

    Parameters
    ----------
    d : array-like
        The difference estimates, ``p1 - p0``.
    p1, p0 : array-like
        The arm estimates.
    s1, s0 : array-like
        The arm standard errors.
    sd : array-like
        The difference's standard error.
    alpha : float, default=0.05
        One minus the level.
    correlated : bool, default=True
        ``False`` drops the correlation term, the mutation control of the screen's test.

    Returns
    -------
    tuple of numpy.ndarray
        The lower and upper limits.
    """
    d, p1, s1, p0, s0, sd = (np.asarray(x, dtype=float) for x in (d, p1, s1, p0, s0, sd))
    z = _z(alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        inside = (p1 > 0.0) & (p1 < 1.0) & (p0 > 0.0) & (p0 < 1.0)
        q1, q0 = np.where(inside, p1, 0.5), np.where(inside, p0, 0.5)
        h1, h0 = z * s1 / (q1 * (1 - q1)), z * s0 / (q0 * (1 - q0))
        l1, u1 = expit(logit(q1) - h1), expit(logit(q1) + h1)
        l0, u0 = expit(logit(q0) - h0), expit(logit(q0) + h0)
        r = np.clip((s1**2 + s0**2 - sd**2) / (2.0 * s1 * s0), -1.0, 1.0)
        if not correlated:
            r = np.zeros_like(r)
        down1, up1, down0, up0 = q1 - l1, u1 - q1, q0 - l0, u0 - q0
        lower = d - np.sqrt(np.maximum(down1**2 + up0**2 - 2 * r * down1 * up0, 0.0))
        upper = d + np.sqrt(np.maximum(up1**2 + down0**2 - 2 * r * up1 * down0, 0.0))
    return np.where(inside, lower, np.nan), np.where(inside, upper, np.nan)


def _groups(record: StudyRecord, property_rows: pd.DataFrame | None) -> Iterator[Group]:
    path = record.artifacts / "replicates.parquet"
    if path.exists():
        rows = read_rows(path)
        rows = rows.loc[rows["implementation"] == record.implementation]
        for (scenario, estimand), group in rows.groupby(["scenario", "estimand"], sort=True):
            yield Group(
                f"{scenario}/{estimand}",
                "truth",
                str(scenario),
                group.set_index("replicate", drop=False),
                group.index,
            )
    if property_rows is not None:
        rows = property_rows.loc[property_rows["role"] == "positive"]
        for (prop, cell), group in rows.groupby(["property", "cell"], sort=True):
            if group["replicate"].is_unique:
                yield Group(
                    f"{prop}/{cell}", "property", "", group.set_index("replicate", drop=False),
                    group.index,
                )  # fmt: skip


def _values(group: Group, column: str) -> np.ndarray:
    return group.rows[column].to_numpy(dtype=float)


def _is_arm(group: Group) -> bool:
    estimate, se = _values(group, "estimate"), _values(group, "std_error")
    return bool(
        len(estimate)
        and np.isfinite(estimate).all()
        and ((estimate > 0.0) & (estimate < 1.0)).all()
        and (np.isfinite(se) & (se > 0.0)).all()
    )


def _is_level(group: Group, heads: frozenset[str]) -> bool:
    """Whether the group's name reads a level: a truth head of a bounded level, or a property
    cell whose name (before ``__``) reads one."""
    name = group.key.split("/", 1)[1]
    if group.kind == "truth":
        return name.split("[", 1)[0] in heads
    return bool(_LEVEL_CELL.search(name.split("__", 1)[0]))


def _is_candidate_difference(group: Group, heads: frozenset[str]) -> bool:
    estimate, se = _values(group, "estimate"), _values(group, "std_error")
    return bool(
        not _is_level(group, heads)
        and len(estimate)
        and np.isfinite(estimate).all()
        and (np.abs(estimate) <= 1.0).all()
        and (np.isfinite(se) & (se > 0.0)).all()
    )


def _suffix(key: str) -> str:
    return key.rsplit("__", 1)[-1] if "__" in key else ""


def pair(groups: list[Group], heads: frozenset[str] | None = None) -> dict[str, tuple[str, str]]:
    """Each paired difference of one pool: its key to the keys of its minuend and subtrahend.

    A pair must reproduce the difference's estimate on every one of its replicates to
    :data:`IDENTITY_TOLERANCE` and give an implied correlation inside ``[-1, 1]`` (to 1e-6)
    on each.  Among several pairs, the one whose property-cell suffix matches the difference's
    is taken, then the first in key order.  A group whose name reads a level (``heads``, by
    default :func:`~tests.studies.evidence.interval_audit.bounded_heads`) is never a
    difference, because ``b = a - d`` holds whenever ``d = a - b`` does.
    """
    heads = interval_audit.bounded_heads() if heads is None else heads
    arms = [group for group in groups if _is_arm(group)]
    found: dict[str, tuple[str, str]] = {}
    if len(arms) < 2:
        return found
    for difference in groups:
        if not _is_candidate_difference(difference, heads):
            continue
        first = difference.rows.index[0]
        holding = [arm for arm in arms if arm.key != difference.key and first in arm.rows.index]
        if len(holding) < 2:
            continue
        at_first = np.array([float(arm.rows.at[first, "estimate"]) for arm in holding])
        order = np.argsort(at_first)
        ordered = at_first[order]
        target = float(difference.rows.at[first, "estimate"])
        candidates = []
        for i, minuend in enumerate(holding):
            wanted = at_first[i] - target
            lo = np.searchsorted(ordered, wanted - IDENTITY_TOLERANCE, "left")
            hi = np.searchsorted(ordered, wanted + IDENTITY_TOLERANCE, "right")
            for j in order[lo:hi]:
                if j != i:
                    candidates.append((minuend, holding[j]))
        best: tuple[int, str, str] | None = None
        for minuend, subtrahend in candidates:
            replicates = difference.rows.index
            if not (
                replicates.isin(minuend.rows.index).all()
                and replicates.isin(subtrahend.rows.index).all()
            ):
                continue
            a = minuend.rows.loc[replicates]
            b = subtrahend.rows.loc[replicates]
            d = _values(difference, "estimate")
            gap = np.abs(d - (a["estimate"].to_numpy(float) - b["estimate"].to_numpy(float)))
            if not (gap <= IDENTITY_TOLERANCE).all():
                continue
            s1, s0 = a["std_error"].to_numpy(float), b["std_error"].to_numpy(float)
            sd = _values(difference, "std_error")
            r = (s1**2 + s0**2 - sd**2) / (2.0 * s1 * s0)
            if not (np.abs(r) <= 1.0 + 1e-6).all():
                continue
            match = int(_suffix(minuend.key) == _suffix(difference.key)) + int(
                _suffix(subtrahend.key) == _suffix(difference.key)
            )
            choice = (-match, minuend.key, subtrahend.key)
            if best is None or choice < best:
                best = choice
        if best is not None:
            found[difference.key] = (best[1], best[2])
    return found


def _identity_consistent(group: Group, z: float) -> bool:
    """Whether every committed ``covered`` is the identity normal interval's, to round-off."""
    estimate, se, truth = (_values(group, c) for c in ("estimate", "std_error", "truth"))
    covered = group.rows["covered"].to_numpy(dtype=int)
    distance = np.abs(estimate - truth) / se - z
    clear = np.abs(distance) > 1e-9
    return bool(((distance <= 0) == (covered == 1))[clear].all())


def _null_zero_consistent(group: Group, z: float) -> bool:
    if "rejected" not in group.rows.columns:
        return False
    rejected = group.rows["rejected"].to_numpy(dtype=float)
    if not np.isfinite(rejected).all():
        return False
    statistic = np.abs(_values(group, "estimate")) / _values(group, "std_error") - z
    clear = np.abs(statistic) > 1e-9
    return bool(((statistic > 0) == (rejected == 1))[clear].all())


@dataclass
class _Study:
    record: StudyRecord
    committed: pd.DataFrame | None
    groups: list[Group]


def _load(record: StudyRecord) -> _Study:
    path = record.artifacts / "property-replicates.parquet"
    committed = read_rows(path) if path.exists() else None
    groups = list(_groups(record, committed))
    if interval_audit._stopping_rule_study(record):
        groups = [
            group
            for group in groups
            if not group.key.endswith(
                (interval_audit.WITNESS_SUFFIX, interval_audit.WIDENED_SUFFIX)
            )
        ]
    return _Study(record, committed, groups)


def _judge(
    study: _Study,
    moved: dict[str, tuple[np.ndarray, np.ndarray | None]],
    groups: dict[str, Group],
) -> list[str]:
    """The verdicts that move when each group of ``moved`` takes its new ``covered`` and
    ``rejected``: truth groups by the coverage floor, property families by the study's rule."""
    margins = study.record.margins
    flips: list[str] = []
    families: set[str] = set()
    recomputed = None if study.committed is None else study.committed.copy()
    for key, (covered, rejected) in moved.items():
        group = groups[key]
        committed = group.rows["covered"].to_numpy(dtype=int)
        total = len(covered)
        before = interval_audit._exact_interval(
            int(committed.sum()), total, margins.confidence_level
        )
        after = interval_audit._exact_interval(int(covered.sum()), total, margins.confidence_level)
        if group.kind == "truth":
            floor = margins.coverage_floor
            if (before[0] >= floor) != (after[0] >= floor):
                flips.append(key)
            continue
        assert recomputed is not None
        recomputed.loc[group.index, "covered"] = covered
        changed = interval_audit._crosses(before, after, margins)
        if rejected is not None:
            old = group.rows["rejected"].to_numpy(dtype=float)
            if (old != rejected).any():
                recomputed.loc[group.index, "rejected"] = rejected
                changed = True
        if changed:
            families.add(key.split("/", 1)[0])
    if families:
        assert study.committed is not None and recomputed is not None
        flips.extend(
            interval_audit._moved_verdicts(
                study.record, study.committed, recomputed, sorted(families)
            )
        )
    return flips


def screen_row(record: StudyRecord) -> dict[str, Any]:
    """One study's screen row, recomputed from its committed rows."""
    slug = record.slug
    if slug in CURRENT:
        return {
            **dict.fromkeys(SCREEN_COLUMNS, 0),
            "study": slug,
            "disposition": "current",
            "max_coverage_change": 0.0,
            "mover_flips": "",
            "unpaired": "",
            "widening_flips": "",
            "reason": CURRENT[slug],
        }
    study = _load(record)
    alpha = record.margins.alpha
    z = _z(alpha)
    by_key = {group.key: group for group in study.groups}
    pools: dict[str, list[Group]] = {}
    for group in study.groups:
        pools.setdefault(group.pool, []).append(group)
    heads = interval_audit.bounded_heads()
    pairs: dict[str, tuple[str, str]] = {}
    for pool in pools.values():
        pairs.update(pair(pool, heads))

    # Differences: MOVER, exactly.
    out_of_scope = 0
    largest = 0.0
    mover: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for key, (first, second) in pairs.items():
        group = by_key[key]
        if not _identity_consistent(group, z):
            out_of_scope += 1
            continue
        replicates = group.rows.index
        a, b = by_key[first].rows.loc[replicates], by_key[second].rows.loc[replicates]
        low, high = mover_interval(
            _values(group, "estimate"),
            a["estimate"],
            a["std_error"],
            b["estimate"],
            b["std_error"],
            _values(group, "std_error"),
            alpha=alpha,
        )
        truth = _values(group, "truth")
        covered = ((low <= truth) & (truth <= high)).astype(int)
        rejected = None
        if group.kind == "property" and _null_zero_consistent(group, z):
            rejected = ((low > 0.0) | (high < 0.0)).astype(float)
        committed = group.rows["covered"].to_numpy(dtype=int)
        if (covered != committed).any() or (
            rejected is not None and (rejected != group.rows["rejected"].to_numpy(float)).any()
        ):
            mover[key] = (covered, rejected)
            largest = max(largest, abs(float(covered.mean()) - float(committed.mean())))
    mover_flips = _judge(study, mover, by_key)

    unpaired = sorted(
        group.key
        for group in study.groups
        if group.key not in pairs
        and (
            (
                group.kind == "truth"
                and group.key.split("/", 1)[1].split("[", 1)[0] in DIFFERENCE_HEADS
            )
            or (group.kind == "property" and _DIFFERENCE_CELL.search(group.key.split("/", 1)[1]))
        )
    )

    # Continuous estimates: the symmetric widening that contains the skew-widened interval.
    widened = [
        group
        for group in study.groups
        if group.key not in pairs
        and np.isfinite(_values(group, "estimate")).all()
        and (np.isfinite(_values(group, "std_error")) & (_values(group, "std_error") > 0)).all()
        and np.isfinite(_values(group, "truth")).all()
        and _identity_consistent(group, z)
    ]
    smallest: dict[str, float] = {}
    for factor in FACTORS:
        moved: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        for group in widened:
            estimate, se, truth = (_values(group, c) for c in ("estimate", "std_error", "truth"))
            covered = (np.abs(estimate - truth) <= z * factor * se).astype(int)
            rejected = None
            if group.kind == "property" and _null_zero_consistent(group, z):
                rejected = (np.abs(estimate) > z * factor * se).astype(float)
            if (covered != group.rows["covered"].to_numpy(dtype=int)).any() or (
                rejected is not None and (rejected != group.rows["rejected"].to_numpy(float)).any()
            ):
                moved[group.key] = (covered, rejected)
        for key in _judge(study, moved, by_key):
            smallest.setdefault(key, factor)

    if mover_flips:
        disposition = "declared" if slug in SCREEN_REASONS else "moved"
    elif smallest:
        disposition = "refits"
    else:
        disposition = "neutral"
    return {
        "study": slug,
        "disposition": disposition,
        "paired_groups": len(pairs),
        "out_of_scope": out_of_scope,
        "moved_groups": len(mover),
        "max_coverage_change": round(largest, 6),
        "mover_flips": ";".join(mover_flips),
        "unpaired": ";".join(unpaired),
        "widening_groups": len(widened),
        "widening_flips": ";".join(f"{key}@{factor:g}" for key, factor in sorted(smallest.items())),
        "reason": SCREEN_REASONS.get(slug, ""),
    }


def _row(slug: str) -> dict[str, Any]:
    import warnings

    warnings.simplefilter("ignore")
    return screen_row(interval_audit.record_for(slug))


def write_screen(slugs: Iterable[str] | None = None, *, jobs: int = 1) -> Path:
    """Recompute the screen rows of ``slugs`` (every registered study by default) and write
    the screen, keeping the committed rows of studies not recomputed; its path."""
    every = sorted(record.slug for record in registered())
    chosen = every if slugs is None else sorted(slugs)
    if jobs > 1:
        from multiprocessing import Pool

        with Pool(jobs) as pool:
            rows = dict(zip(chosen, pool.map(_row, chosen, chunksize=1), strict=True))
    else:
        rows = {slug: _row(slug) for slug in chosen}
    if slugs is not None and SCREEN_FILE.exists():
        kept = read_screen().set_index("study", drop=False)
        for slug in every:
            if slug not in rows and slug in kept.index:
                rows[slug] = kept.loc[slug].to_dict()
    frame = pd.DataFrame(
        [rows[slug] for slug in every if slug in rows], columns=list(SCREEN_COLUMNS)
    )
    frame.to_csv(SCREEN_FILE, index=False, lineterminator="\n")
    return SCREEN_FILE


def read_screen() -> pd.DataFrame:
    """The committed screen, with empty text cells as ``""``."""
    return pd.read_csv(SCREEN_FILE, keep_default_na=False)


def main(argv: list[str] | None = None) -> None:
    """Rewrite the screen."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("studies", nargs="*", help="recompute only these studies' rows")
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)
    print(write_screen(arguments.studies or None, jobs=arguments.jobs))


if __name__ == "__main__":
    main()
