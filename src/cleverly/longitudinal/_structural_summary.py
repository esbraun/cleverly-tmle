"""The readable summaries of a structural nested mean fit and of its model check (X89 T2).

``StructuralNestedMeanResult.summary()`` and ``StructuralModelCheck.summary()`` build here, on
the shared :mod:`cleverly._summary` layer. Every fact is read from the result: the blip layout
from its config, the numbers from its estimates and diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._inference_status import (
    NO_SIMULTANEOUS_BANDS,
    NO_T_REFERENCE_BANDS,
    T_REFERENCE_NOTE,
    supplies_inference,
)
from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    count_text,
    note,
    number_text,
    pvalue_text,
    vocabulary,
)
from .._summary._estimates import estimates_table, plain_support_reason
from .._summary._learners import is_parametric, learner_name
from .._summary._render import decimals_for
from ..estimators._point_summary import _bands, _decimals_of, _warning_notes
from ..inference.cluster import fewest_clusters
from ._summary import _covariates, _time_points, estimates_repr, population_facts

if TYPE_CHECKING:
    from .structural import StructuralModelCheck, StructuralNestedMeanResult


# ----------------------------------------------------------------- row meanings


def _reference_level(result: StructuralNestedMeanResult, time: int) -> str:
    from .structural import _level_labels

    return str(_level_labels(result.data, time)[0])


def _outcome(result: StructuralNestedMeanResult) -> str:
    return str(result.data.outcome_name)


def structural_meaning(result: StructuralNestedMeanResult, name: str) -> str:
    """What one reported row of a structural fit estimates, in plain words.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fitted result.
    name : str
        A key of ``result.estimates``.

    Returns
    -------
    str
        The meaning, or the name itself when the summary has no words for it.
    """
    from .structural import _INTERCEPT, REFERENCE_NAME

    outcome = _outcome(result)
    kind = result.kinds.get(name)
    if kind == "reference" or name == REFERENCE_NAME:
        return vocabulary.STRUCTURAL_MEANINGS["reference"].format(Y=outcome)
    inside = name.partition("[")[2].removesuffix("]")
    if kind == "blip":
        treatment, _, rest = inside.partition(":")
        level, _, feature = rest.partition(":")
        node = next((node for node in result.config.nodes if node.treatment == treatment), None)
        if node is None:
            return name
        words = (
            vocabulary.STRUCTURAL_DOSE.format(A=treatment)
            if node.continuous
            else vocabulary.STRUCTURAL_LEVEL.format(
                A=treatment, level=level, ref=_reference_level(result, node.time)
            )
        )
        key = "intercept" if feature == _INTERCEPT else "blip"
        return vocabulary.STRUCTURAL_MEANINGS[key].format(
            feature=feature, Y=outcome, change=words, t=node.time
        )
    if kind == "level":
        return vocabulary.STRUCTURAL_MEANINGS["level"].format(Y=outcome, d=inside)
    if kind == "contrast":
        labels = {label for label, _ in result.config.regimens}
        parts = inside.split(" vs ")
        for cut in range(1, len(parts)):
            plan, reference = " vs ".join(parts[:cut]), " vs ".join(parts[cut:])
            if plan in labels and reference in labels:
                return vocabulary.STRUCTURAL_MEANINGS["contrast"].format(
                    Y=outcome, d=plan, r=reference
                )
    return name


# ----------------------------------------------------------------- tables


def _features_table(result: StructuralNestedMeanResult) -> SummaryTable:
    rows = []
    for node in result.config.nodes:
        levels = (
            "dose"
            if node.continuous
            else f"{', '.join(node.levels)} against {_reference_level(result, node.time)}"
        )
        features = ", ".join(feature.name for feature in node.features)
        rows.append(
            (
                Cell(node.time, str(node.time)),
                Cell(node.treatment, node.treatment),
                Cell(levels, levels),
                Cell(features, features),
            )
        )
    return SummaryTable(
        (
            vocabulary.TIME,
            vocabulary.BLIP_TREATMENT,
            vocabulary.BLIP_LEVELS,
            vocabulary.BLIP_FEATURES,
        ),
        tuple(rows),
        title=vocabulary.BLIP_TITLE,
        kind="other",
    )


# ----------------------------------------------------------------- header and notes


def _learners(result: StructuralNestedMeanResult) -> dict[str, Any]:
    estimator = result.estimator
    outcome = getattr(estimator, "outcome_learner", None)
    regression = getattr(estimator, "regression_learner", None)
    return {
        "outcome": outcome,
        "treatment": getattr(estimator, "treatment_learner", None),
        "regression": outcome if regression is None else regression,
    }


def _header(result: StructuralNestedMeanResult) -> tuple[list[tuple[str, str]], int]:
    data = result.data
    config = result.config
    question = (
        vocabulary.STRUCTURAL_QUESTION_PLANS
        if config.regimens and not config.reads_covariates
        else vocabulary.STRUCTURAL_QUESTION
    )
    observed = np.asarray(data.observed_outcomes(), dtype=float)
    observed = observed[np.isfinite(observed)]
    binary = observed.size > 0 and bool(np.isin(observed, (0.0, 1.0)).all())
    left: list[tuple[str, str]] = [
        (vocabulary.QUESTION, question.format(Y=_outcome(result))),
        (
            vocabulary.OUTCOME,
            f"{_outcome(result)} ({'binary' if binary else 'continuous'}), at the end",
        ),
        (vocabulary.TREATMENT, ", ".join(data.treatment_names)),
    ]
    covariates = _covariates(result)  # type: ignore[arg-type]
    if covariates is not None:
        left.append((vocabulary.ADJUSTED_FOR, covariates))
    if data.censoring_names:
        left.append(
            (
                vocabulary.CENSORING,
                vocabulary.CENSORING_FACT.format(names=", ".join(data.censoring_names)),
            )
        )
    left.append((vocabulary.METHOD, vocabulary.STRUCTURAL_METHOD))
    left.extend(population_facts(data))
    learners = _learners(result)
    outcome = learner_name(learners["outcome"])
    regression = learner_name(learners["regression"])
    if regression != outcome:
        outcome += f"; {regression} for the other regressions"
    right: list[tuple[str, str]] = [
        (vocabulary.UNITS if data.panel is not None else vocabulary.ROWS, count_text(result.n)),
        (vocabulary.TIME_POINTS, _time_points(result)),  # type: ignore[arg-type]
        (
            vocabulary.OUT_OF_FOLD,
            "no" if config.n_folds <= 1 else f"{config.n_folds} folds",
        ),
        (vocabulary.OUTCOME_MODEL, outcome),
        (vocabulary.TREATMENT_MODEL, learner_name(learners["treatment"])),
    ]
    return [*left, *right], len(left)


def _notes(result: StructuralNestedMeanResult) -> list[Note]:
    config = result.config
    sentences = [vocabulary.NOTE_TEXTS["structural_model"]]
    if config.reads_covariates:
        sentences.append(vocabulary.NOTE_TEXTS["structural_covariate_feature"])
    elif len(config.regimens) >= 2:
        sentences.append(vocabulary.NOTE_TEXTS["structural_check"])
    notes = [note("structural_model", " ".join(sentences))]
    inferential = any(estimate.supplies_inference for estimate in result.estimates.values())
    if config.n_folds <= 1 and inferential:
        flexible = any(
            learner is not None and not is_parametric(learner)
            for learner in _learners(result).values()
        )
        if flexible:
            notes.append(
                note("in_sample_flexible", vocabulary.NOTE_TEXTS["in_sample_flexible_longitudinal"])
            )
    if result.data.is_weighted:
        notes.append(note("weighted_population", vocabulary.NOTE_TEXTS["weighted_population"]))
    notes.extend(_warning_notes(result.fit_warnings))
    return notes


# ----------------------------------------------------------------- full detail


def _settings(result: StructuralNestedMeanResult) -> SummarySection:
    config = result.config
    data = result.data
    diagnostics = result.diagnostics
    lines: list[str] = []
    if config.n_folds > 1:
        lines.append(f"Folds: {config.n_folds}.")
    lines.append(
        vocabulary.STRUCTURAL_FAMILY.format(
            family=vocabulary.FAMILY_WORDS.get(config.family, config.family)
        )
    )
    if diagnostics.smallest_eigenvalue:
        smallest = min(diagnostics.smallest_eigenvalue)
        lines.append(vocabulary.STRUCTURAL_EIGENVALUE.format(value=f"{smallest:.3g}"))
    shares = [share for share in diagnostics.extreme_share if not np.isnan(share)]
    if shares:
        lines.append(
            vocabulary.STRUCTURAL_EXTREME.format(
                margin=f"{config.positivity_margin:g}", share=f"{max(shares):.1%}"
            )
        )
    estimates = result.estimates
    t_reference = any(estimate.reference_df is not None for estimate in estimates.values())
    if result.simultaneous is None and len(estimates) > 1:
        if not supplies_inference(result.inference_status):
            lines.append(NO_SIMULTANEOUS_BANDS)
        elif t_reference:
            lines.append(NO_T_REFERENCE_BANDS)
    if t_reference and data.cluster is not None:
        lines.append(
            T_REFERENCE_NOTE.format(
                clusters=fewest_clusters(
                    data.cluster, weights=data.weights if data.is_weighted else None
                ),
                stratum="",
            )
        )
    reasons = list(
        dict.fromkeys(
            estimate.support.reason
            for estimate in estimates.values()
            if estimate.support.label == vocabulary.EXPERIMENTAL and estimate.support.reason
        )
    )
    for reason in reasons:
        lines.append(f"Why the rows are experimental: {plain_support_reason(reason)}")
    return SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines))


def _assumptions() -> SummarySection:
    lines = [vocabulary.ASSUMPTIONS_LEAD_GENERIC]
    lines.extend(f"- {item}" for item in vocabulary.LONGITUDINAL_ASSUMPTIONS)
    lines.append(f"- model restriction: {vocabulary.STRUCTURAL_RESTRICTION}")
    return SummarySection(vocabulary.SECTION_ASSUMPTIONS, tuple(lines))


# ----------------------------------------------------------------- the summaries


def structural_summary(result: StructuralNestedMeanResult, detail: Detail) -> Summary:
    """Build the summary of a structural nested mean fit.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fitted result.
    detail : {"short", "full"}
        ``"full"`` adds the Assumptions, Settings, Method details and Study description
        sections.

    Returns
    -------
    Summary
        The summary.
    """
    level_value = 1.0 - result.config.alpha_sig
    level = f"{level_value * 100:g}%"
    header, split = _header(result)
    table, notes = estimates_table(result.estimates, level=level_value)
    tables = [table]
    if len(result.estimates) > 1:
        tables.append(
            SummaryTable(
                ("", vocabulary.MEANING),
                tuple(
                    (Cell(name, name), Cell.of(structural_meaning(result, name)))
                    for name in result.estimates
                ),
                title=vocabulary.KEY_TITLE,
            )
        )
    tables.append(_features_table(result))
    bands = _bands(result, _decimals_of(table), level)  # type: ignore[arg-type]
    if bands is not None:
        tables.append(bands[0])
        notes.append(bands[1])
    notes.extend(_notes(result))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(_assumptions())
        sections.append(_settings(result))
        sections.append(
            SummarySection(vocabulary.SECTION_METHOD, (vocabulary.STRUCTURAL_CONSTRUCTION,))
        )
        sections.append(SummarySection(vocabulary.SECTION_STUDY, (vocabulary.NO_PROTOCOL,)))
    return Summary.build(
        title=vocabulary.STRUCTURAL_TITLE,
        header=header,
        header_split=split,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def structural_repr(result: StructuralNestedMeanResult) -> str:
    """The one-line repr of a structural fit.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fitted result.

    Returns
    -------
    str
        The repr.
    """
    return estimates_repr(
        type(result).__name__,
        result.estimates,
        n=result.n,
        method="SNMM",
        alpha=result.config.alpha_sig,
    )


def check_frame_rows(check: StructuralModelCheck) -> dict[str, list[Any]]:
    """The columns of :meth:`StructuralModelCheck.to_frame`.

    Parameters
    ----------
    check : StructuralModelCheck
        The check.

    Returns
    -------
    dict of str to list
        One entry per contrast in each column.
    """
    spread = np.sqrt(np.clip(np.diag(np.atleast_2d(check.covariance)), 0.0, None))
    return {
        "contrast": list(check.names),
        "structural": [float(value) for value in check.structural],
        "nonparametric": [float(value) for value in check.nonparametric],
        "difference": [float(value) for value in check.difference],
        "std_err": [float(value) for value in spread],
    }


def check_summary(check: StructuralModelCheck, detail: Detail) -> Summary:
    """Build the summary of a structural model check.

    Parameters
    ----------
    check : StructuralModelCheck
        The check.
    detail : {"short", "full"}
        ``"full"`` adds how the check works and what its degrees of freedom mean.

    Returns
    -------
    Summary
        The summary, with the reading of the test.
    """
    columns = check_frame_rows(check)
    decimals = decimals_for(columns["std_err"])
    rows = tuple(
        (
            Cell(name, name),
            Cell(a, number_text(a, decimals)),
            Cell(b, number_text(b, decimals)),
            Cell(d, number_text(d, decimals)),
            Cell(s, number_text(s, decimals)),
        )
        for name, a, b, d, s in zip(
            columns["contrast"],
            columns["structural"],
            columns["nonparametric"],
            columns["difference"],
            columns["std_err"],
            strict=True,
        )
    )
    tables = [
        SummaryTable(
            (
                "",
                vocabulary.CHECK_STRUCTURAL,
                vocabulary.CHECK_LTMLE,
                vocabulary.CHECK_DIFFERENCE,
                vocabulary.CHECK_SPREAD,
            ),
            rows,
            title=vocabulary.CHECK_TABLE_TITLE,
            kind="other",
        )
    ]
    header: list[tuple[str, str]] = [
        (vocabulary.QUESTION, vocabulary.CHECK_QUESTION),
        (vocabulary.CHECK_CONTRASTS, str(len(check.names))),
        (vocabulary.CHECK_SPLITS_LABEL, str(check.n_splits)),
    ]
    if check.tested:
        infinite = not np.isfinite(check.reference_df)
        test = (
            vocabulary.CHECK_TEST_CHI2.format(df=check.df)
            if infinite
            else vocabulary.CHECK_TEST_F.format(
                df=check.df, df2=f"{check.reference_df - check.df + 1:.1f}"
            )
        )
        header.append((vocabulary.CHECK_TEST, test))
        tables.append(
            SummaryTable(
                (
                    vocabulary.CHECK_STATISTIC,
                    vocabulary.CHECK_CONTRASTS_COLUMN,
                    vocabulary.CHECK_COVARIANCE_DF,
                    vocabulary.CHECK_PVALUE,
                ),
                (
                    (
                        Cell(float(check.statistic), f"{check.statistic:.3f}"),
                        Cell(check.df, str(check.df)),
                        Cell(
                            float(check.reference_df),
                            "infinite" if infinite else f"{check.reference_df:.1f}",
                        ),
                        Cell(float(check.pvalue), pvalue_text(check.pvalue)),
                    ),
                ),
                title=vocabulary.CHECK_JOINT_TITLE,
                kind="other",
            )
        )
        reading = vocabulary.CHECK_READING
    else:
        reading = vocabulary.CHECK_NO_TEST.format(reason=check.reason)
    notes: list[Note] = []
    if check.support.label == vocabulary.EXPERIMENTAL:
        reason = plain_support_reason(check.support.reason or "")
        notes.append(
            note(
                "experimental",
                vocabulary.NOTE_TEXTS["check_experimental"].format(reason=reason).strip(),
            )
        )
    if check.n_splits > 1:
        notes.append(
            note("check_splits", vocabulary.NOTE_TEXTS["check_splits"].format(k=check.n_splits))
        )
    sections = (
        [SummarySection(vocabulary.SECTION_HOW_TO_READ, vocabulary.CHECK_HOW_TO_READ)]
        if detail == "full"
        else []
    )
    return Summary.build(
        title=vocabulary.CHECK_TITLE,
        header=header,
        header_split=len(header),
        tables=tables,
        notes=notes,
        reading=reading,
        sections=sections,
    )


def check_repr(check: StructuralModelCheck) -> str:
    """The one-line repr of a structural model check.

    Parameters
    ----------
    check : StructuralModelCheck
        The check.

    Returns
    -------
    str
        For example ``StructuralModelCheck(contrasts=2, p=0.41, splits=5)``.
    """
    from .._summary import short_repr

    facts: list[tuple[str, Any]] = [("contrasts", len(check.names))]
    facts.append(("p", pvalue_text(check.pvalue)) if check.tested else ("", "no test"))
    facts.append(("splits", check.n_splits))
    return short_repr(type(check).__name__, facts)
