"""The intervention objects, before any of them reaches an estimator.

Everything here is exact: a regime is a density over the arms, so its shape, its
normalisation and the arm each unit is assigned to are checked directly rather than
inferred from an estimate that used them.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import DataError
from cleverly.interventions import (
    Incremental,
    RegimeSet,
    Rule,
    Shift,
    Static,
    Stochastic,
    as_interventions,
    check_support,
    refuse_unsupported,
)


def make_data(n: int = 40, *, levels: tuple = (0, 1), seed: int = 0) -> CausalData:
    rng = np.random.default_rng(seed)
    k = len(levels)
    frame = {
        "Y": rng.binomial(1, 0.4, n).astype(float),
        "A": np.asarray([levels[i % k] for i in range(n)]),
        "W1": rng.normal(size=n),
        "W2": rng.normal(size=n),
    }
    import pandas as pd

    return CausalData.from_frame(
        pd.DataFrame(frame), outcome="Y", treatment="A", covariates=["W1", "W2"]
    )


# ------------------------------------------------------------------ the kinds


def test_static_puts_all_mass_on_one_arm() -> None:
    data = make_data()
    density = Static(1).density(data)
    assert density.shape == (data.n, 2)
    assert np.array_equal(density, np.tile([0.0, 1.0], (data.n, 1)))


def test_static_uses_the_users_own_levels() -> None:
    data = make_data(levels=("low", "medium", "high"))
    # "high" sorts second of the three, so its code is 0.0 -- a regime that read the
    # label as a position would put its mass in the wrong column.
    assert data.treatment_levels == ("high", "low", "medium")
    density = Static("medium").density(data)
    assert np.array_equal(density, np.tile([0.0, 0.0, 1.0], (data.n, 1)))


def test_static_names_itself_after_its_level() -> None:
    assert Static("high").name == "always high"
    assert Static(1, name="treat everyone").name == "treat everyone"


def test_static_refuses_a_level_the_data_does_not_have() -> None:
    data = make_data()
    with pytest.raises(DataError, match="not a level of A"):
        Static(7).density(data)


def test_rule_assigns_per_row() -> None:
    data = make_data()
    rule = Rule(lambda w: np.where(np.asarray(w["W1"]) > 0.0, 1, 0), name="treat if W1 > 0")
    density = rule.density(data)
    expected = (data.covariates[:, 0] > 0.0).astype(float)
    assert np.array_equal(density[:, 1], expected)
    assert np.array_equal(density.sum(axis=1), np.ones(data.n))


def test_a_constant_rule_is_the_static_regime() -> None:
    """The property every equivalence claim downstream rests on."""
    data = make_data()
    rule = Rule(lambda w: np.ones(len(w), dtype=int), name="everyone")
    assert np.array_equal(rule.density(data), Static(1).density(data))


def test_rule_sees_covariates_only() -> None:
    """A regime is a function of W; reading Y or A is a different object entirely."""
    data = make_data()
    seen: list[list[str]] = []
    Rule(lambda w: (seen.append(list(w.columns)), np.zeros(len(w), dtype=int))[1], "peek").density(
        data
    )
    assert seen == [["W1", "W2"]]


def test_rule_refuses_an_undeclared_level() -> None:
    data = make_data()
    rule = Rule(lambda w: np.full(len(w), 3), name="off by three")
    with pytest.raises(DataError, match="not a level of A"):
        rule.density(data)


def test_rule_refuses_the_wrong_length() -> None:
    data = make_data()
    with pytest.raises(DataError, match="one treatment level per row"):
        Rule(lambda w: np.zeros(3, dtype=int), name="short").density(data)


def test_stochastic_passes_a_normalised_density_through() -> None:
    data = make_data()
    values = np.column_stack([np.full(data.n, 0.3), np.full(data.n, 0.7)])
    assert np.array_equal(Stochastic(lambda w: values, "coin").density(data), values)


def test_stochastic_refuses_rows_that_do_not_sum_to_one() -> None:
    data = make_data()
    values = np.column_stack([np.full(data.n, 0.3), np.full(data.n, 0.5)])
    with pytest.raises(DataError, match="rows summing to"):
        Stochastic(lambda w: values, "unnormalised").density(data)


def test_stochastic_refuses_a_negative_probability() -> None:
    data = make_data()
    values = np.column_stack([np.full(data.n, -0.2), np.full(data.n, 1.2)])
    with pytest.raises(DataError, match="negative probability"):
        Stochastic(lambda w: values, "negative").density(data)


def test_stochastic_refuses_the_wrong_number_of_arms() -> None:
    data = make_data(levels=(0, 1, 2))
    values = np.column_stack([np.full(data.n, 0.5), np.full(data.n, 0.5)])
    with pytest.raises(DataError, match=r"expected \(40, 3\)"):
        Stochastic(lambda w: values, "two of three").density(data)


# ------------------------------------------------------------------- refusals


def test_an_ipsi_is_redirected_to_its_own_keyword_rather_than_refused() -> None:
    """It used to be a refusal; it is now a signpost, and the type says which.

    The message still explains *why* the regime path cannot express it -- g* is a
    functional of P, so the influence function carries a further term -- because that is
    what stops a reader building one out of ``Stochastic`` and believing the standard
    error.  What changed is that there is now somewhere to send them.
    """
    with pytest.raises(ValueError, match=r"TMLE\(incremental=") as raised:
        refuse_unsupported("ipsi")
    assert not isinstance(raised.value, NotImplementedError)
    assert "functional of P" in str(raised.value)


@pytest.mark.parametrize("kind", ["mtp", "shift"])
def test_a_shift_is_redirected_to_its_own_keyword_rather_than_refused(kind: str) -> None:
    """A shift is implemented; what it is not is an *intervention*.

    The refusal used to say the learner interface had no ``predict_density``, which
    stopped being true when the conditional density estimator landed.  It still raises,
    because ``interventions=`` takes regimes -- a distribution over arms given ``W`` --
    and a shift reads the dose the unit received.  So the message points at the keyword
    that does work, and the type changes from ``NotImplementedError`` to ``ValueError``
    to say the difference is one of API rather than of derivation.
    """
    with pytest.raises(ValueError, match=r"TMLE\(shifts=") as raised:
        refuse_unsupported(kind)
    assert not isinstance(raised.value, NotImplementedError)


@pytest.mark.parametrize(
    ("declared", "keyword"),
    [
        (Shift(0.5, 5.0, name="up"), r"TMLE\(shifts="),
        (Incremental(2.0, name="d2"), r"TMLE\(incremental="),
    ],
)
def test_the_wrong_keyword_reaches_the_signpost_rather_than_falling_through(
    declared: object, keyword: str
) -> None:
    """The two tests above call the refusal; this one arrives at it the way a user does.

    Without this, ``refuse_unsupported`` has no call site in the library at all and those
    tests only prove that a function raises when called.  ``as_interventions`` used to send
    both of these to ``Static``, which wrapped the object as though it were a treatment
    *level* -- so the fit failed much later, about something else, having first built a
    regime named ``"always Shift(delta=0.5, cap=5.0, name='up')"``.
    """
    with pytest.raises(ValueError, match=keyword):
        as_interventions(declared)
    with pytest.raises(ValueError, match=keyword):
        as_interventions([Static(0.0), declared])


def test_a_regime_still_passes_through_as_interventions() -> None:
    """The negative control for the check above: it must not catch a genuine regime."""
    assert as_interventions((1, 0)) == (Static(1), Static(0))
    static = Static(1.0, name="always")
    assert as_interventions(static) == (static,)
    assert as_interventions(None) == ()


# ----------------------------------------------------------------- regime set


def test_regime_set_keys_by_code_and_labels_separately() -> None:
    data = make_data()
    regimes = RegimeSet.evaluate([Static(0), Static(1)], data)
    assert regimes.codes == (0.0, 1.0)
    assert regimes.labels == {0.0: "always 0", 1.0: "always 1"}
    assert regimes.values.shape == (data.n, 2, 2)
    assert np.array_equal(regimes.column(1.0), Static(1).density(data))


def test_regime_set_defaults_its_reference_to_the_first_supplied() -> None:
    data = make_data()
    assert RegimeSet.evaluate([Static(0), Static(1)], data).reference == 0.0
    chosen = RegimeSet.evaluate([Static(0), Static(1)], data, reference="always 1")
    assert chosen.reference == 1.0


def test_regime_set_refuses_an_unknown_reference() -> None:
    data = make_data()
    with pytest.raises(DataError, match="is not one of the regimes"):
        RegimeSet.evaluate([Static(0)], data, reference="nonexistent")


def test_regime_set_refuses_duplicate_names() -> None:
    data = make_data()
    with pytest.raises(DataError, match="must be distinct"):
        RegimeSet.evaluate([Static(0, name="same"), Static(1, name="same")], data)


def test_regime_set_subset_slices_rather_than_re_evaluates() -> None:
    data = make_data()
    rule = Rule(lambda w: (np.asarray(w["W1"]) > 0).astype(int), name="rule")
    regimes = RegimeSet.evaluate([rule], data)
    index = np.arange(0, data.n, 2)
    assert np.array_equal(regimes.subset(index).values, regimes.values[index])


def test_is_static_separates_the_degenerate_regimes() -> None:
    data = make_data()
    rule = Rule(lambda w: (np.asarray(w["W1"]) > 0).astype(int), name="rule")
    values = np.column_stack([np.full(data.n, 0.3), np.full(data.n, 0.7)])
    assert RegimeSet.evaluate([Static(0), Static(1)], data).is_static
    assert not RegimeSet.evaluate([rule], data).is_static
    assert not RegimeSet.evaluate([Stochastic(lambda w: values, "coin")], data).is_static


def test_bare_levels_are_read_as_static_regimes() -> None:
    interventions = as_interventions((1, 0))
    assert [item.level for item in interventions] == [1, 0]  # type: ignore[attr-defined]
    kept = Static(1, name="explicit")
    assert as_interventions([kept]) == (kept,)
    assert as_interventions(None) == ()


# ------------------------------------------------------------------- support


def test_support_report_finds_the_assigned_arm_not_the_marginal() -> None:
    """The failure a marginal overlap table cannot see.

    Both regimes face the same propensity column.  ``always 1`` is well supported and
    the rule sends exactly the rows whose propensity is smallest to arm 1, so its
    minimum is far lower -- which is the whole reason a regime needs its own report.
    """
    data = make_data(n=40)
    g1 = np.linspace(0.005, 0.9, data.n)
    propensity = np.column_stack([1.0 - g1, g1])
    rule = Rule(lambda w: np.zeros(len(w), dtype=int), name="never")
    regimes = RegimeSet.evaluate([Static(1), rule], data)

    report = check_support(regimes, data.treatment, propensity)
    assert report.regimes["always 1"].min_support_propensity == pytest.approx(g1.min())
    assert report.regimes["never"].min_support_propensity == pytest.approx((1.0 - g1).min())
    assert report.worst is not None and report.worst.name == "always 1"
    assert report.regimes["always 1"].tail_mass[0.01] > 0.0
    assert "always 1" in report.summary()


def test_support_report_counts_structural_violations() -> None:
    data = make_data(n=40)
    g1 = np.full(data.n, 0.5)
    g1[:3] = 0.0
    propensity = np.column_stack([1.0 - g1, g1])
    regimes = RegimeSet.evaluate([Static(1)], data)
    assert check_support(regimes, data.treatment, propensity).regimes["always 1"].unsupported == 3


def test_support_report_effective_sample_size_is_n_when_the_regime_is_the_mechanism() -> None:
    """A stochastic regime equal to g has every ratio at one, and so loses nothing."""
    data = make_data(n=40)
    g1 = np.full(data.n, 0.5)
    propensity = np.column_stack([1.0 - g1, g1])
    regimes = RegimeSet.evaluate([Stochastic(lambda w: propensity, "observed")], data)
    support = check_support(regimes, data.treatment, propensity).regimes["observed"]
    assert support.effective_sample_size == pytest.approx(data.n)
    assert support.max_ratio == pytest.approx(1.0)
