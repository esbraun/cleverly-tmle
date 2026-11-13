"""The per-coordinate Student t reference of the simultaneous band (Fix-2 plan, section 4).

``band_reference="t"`` keeps the multiplier critical value ``c`` and bands estimate ``j`` at
``c_j = t.isf(norm.sf(c), nu_j)``, with ``nu_j = max(Kish_j, 5)`` read from the centred curve.
These tests pin the definition and its witnesses: the shared Kish count, the mapping on a heavy
column against two Gaussian ones with a mutation control for each way it can go wrong, the
limit, the refusals, the unchanged default band and the one-estimate case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from cleverly.inference import kish_count

# ------------------------------------------------------------------------------ 1. definition


def _mean_zero(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


class TestTheKishCount:
    def test_it_is_the_squared_mass_over_the_fourth_moment_of_the_centred_curve(self) -> None:
        rng = np.random.default_rng(1)
        curve = rng.standard_exponential(500) ** 2 + 3.0
        centred = curve - curve.mean()
        expected = float(np.sum(centred**2) ** 2 / np.sum(centred**4))
        assert kish_count(curve) == pytest.approx(expected, rel=1e-12)
        # A nonzero-mean witness: the uncentred count differs by far more than rounding.
        assert kish_count(curve, centered=False) > 2.0 * kish_count(curve)

    def test_appended_zero_rows_leave_a_mean_zero_curve_unchanged(self) -> None:
        curve = _mean_zero(np.random.default_rng(2).standard_t(3, size=400))
        padded = np.concatenate([curve, np.zeros(600)])
        assert kish_count(padded) == pytest.approx(kish_count(curve), rel=1e-12)

    def test_rows_select_the_terms_read(self) -> None:
        curve = np.array([3.0, -1.0, -1.0, -1.0, 0.0, 0.0])
        rows = curve != 0.0
        assert kish_count(curve, rows=rows) == pytest.approx(144.0 / 84.0, rel=1e-12)
        assert kish_count(curve, rows=np.flatnonzero(rows)) == kish_count(curve, rows=rows)

    def test_a_zero_curve_is_infinite(self) -> None:
        assert kish_count(np.zeros(10)) == math.inf
        assert kish_count(np.full(10, 2.5)) == math.inf

    @pytest.mark.parametrize("factor", [-3.0, 0.01, 7.0])
    def test_rescaling_leaves_it_unchanged(self, factor: float) -> None:
        curve = _mean_zero(np.random.default_rng(3).standard_t(4, size=300))
        assert kish_count(factor * curve) == pytest.approx(kish_count(curve), rel=1e-12)

    def test_the_sparse_report_reads_the_same_count(self) -> None:
        from cleverly.inference.sparse import influence_row

        curve = np.random.default_rng(4).standard_normal(1_000) ** 3
        row = influence_row("x", curve)
        assert row.kish == pytest.approx(kish_count(curve), rel=1e-12)
        # The count the report computed before it called the shared function.
        assert row.kish == pytest.approx(row.n / row.kurtosis, rel=1e-12)

    def test_it_matches_fay_and_graubard_to_order_r_over_n_squared(self) -> None:
        """``R / (1 + (R - 1)/(n - 1)^2)`` with ``R`` the count (plan 2.1 step 2)."""
        curve = _mean_zero(np.random.default_rng(5).standard_normal(2_000))
        count = kish_count(curve)
        realised = count / (1.0 + (count - 1.0) / (curve.size - 1.0) ** 2)
        assert abs(realised / count - 1.0) < 2.0 * count / curve.size**2
        assert stats.kurtosis(curve, fisher=False) == pytest.approx(curve.size / count, rel=1e-9)


# ------------------------------------------------------------------------ shared constructions


def _heavy(n: int, terms: int, rng: np.random.Generator) -> np.ndarray:
    """A mean-zero curve whose square mass sits on ``terms`` entries: Kish about ``terms``."""
    curve = 1e-3 * rng.standard_normal(n)
    curve[:terms] += rng.choice([-1.0, 1.0], size=terms) * (1.0 + 0.05 * rng.random(terms))
    return curve - curve.mean()


def _family(n: int = 10_000, seed: int = 0) -> np.ndarray:
    """The plan's family and its witnesses, one column each.

    0: heavy, Kish about 8; 1 and 2: Gaussian, Kish about ``n / 3``; 3: Gaussian around a mean
    of 3, so its uncentred Kish is more than twice its centred one; 4: Kish about 2, below the
    floor.
    """
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((n, 3))
    return np.column_stack(
        [
            _heavy(n, 8, rng),
            gaussian[:, 0] - gaussian[:, 0].mean(),
            gaussian[:, 1] - gaussian[:, 1].mean(),
            3.0 + gaussian[:, 2],
            _heavy(n, 2, rng),
        ]
    )


def _own_df(curves: np.ndarray, floor: float = 5.0) -> np.ndarray:
    """``max((sum d^2)^2 / sum d^4, floor)`` of each centred column, written from the formula."""
    centred = curves - curves.mean(axis=0)
    kish = np.sum(centred**2, axis=0) ** 2 / np.sum(centred**4, axis=0)
    return np.maximum(kish, floor)


C = 2.4


def _mapping_failures(curves: np.ndarray, nus: np.ndarray, values: np.ndarray) -> list[str]:
    """Every pin of test 3 that ``(nus, values)`` breaks, as a list of short names."""
    failures = []
    expected_nu = _own_df(curves)
    if not np.allclose(nus, expected_nu, rtol=1e-12, atol=0.0):
        failures.append("df")
    expected = stats.t.isf(stats.norm.sf(C), expected_nu)
    if not np.allclose(values / C, expected / C, rtol=0.0, atol=1e-12):
        failures.append("mapping")
    if not values[0] / C > 1.2:
        failures.append("heavy widening")
    for j in (1, 2):
        excess = values[j] / C - 1.0
        predicted = (C**2 + 1.0) / (4.0 * expected_nu[j])
        if not (abs(excess / predicted - 1.0) <= 0.10 and excess < 1e-3):
            failures.append(f"expansion {j}")
    return failures


# ------------------------------------------------------------------- 3. the mapping's witness


class TestTheMapping:
    def test_each_column_reads_its_own_df_and_the_same_tail(self) -> None:
        from cleverly.inference.multiplier import t_critical_values

        curves = _family()
        nus, values = t_critical_values(C, curves)
        assert _mapping_failures(curves, nus, values) == []
        kish = _own_df(curves, floor=0.0)
        assert 6.0 < kish[0] < 10.0 and kish[4] < 5.0
        assert kish[1] > 2_200 and kish[2] > 2_200
        assert nus[4] == 5.0

    @staticmethod
    def _mutants(curves: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        tail = stats.norm.sf(C)
        right = _own_df(curves)
        pooled = np.full_like(right, max(kish_count(curves.reshape(-1)), 5.0))
        first = np.full_like(right, right[0])
        infinite = np.full_like(right, np.inf)
        uncentred = np.maximum(
            [kish_count(curves[:, j], centered=False) for j in range(curves.shape[1])], 5.0
        )
        unfloored = _own_df(curves, floor=0.0)
        return {
            "df pooled over columns": (pooled, stats.t.isf(tail, pooled)),
            "df of the first column": (first, stats.t.isf(tail, first)),
            "nu infinite": (infinite, np.full_like(right, C)),
            "one-sided tail": (right, stats.t.isf(2.0 * tail, right)),
            "inverted map": (right, stats.norm.isf(stats.t.sf(C, right))),
            "uncentred Kish": (uncentred, stats.t.isf(tail, uncentred)),
            "floor after the quantile": (
                right,
                np.maximum(stats.t.isf(tail, unfloored), stats.t.isf(tail, 5.0)),
            ),
            "no floor": (unfloored, stats.t.isf(tail, unfloored)),
        }

    @pytest.mark.parametrize(
        "mutant",
        [
            "df pooled over columns",
            "df of the first column",
            "nu infinite",
            "one-sided tail",
            "inverted map",
            "uncentred Kish",
            "floor after the quantile",
            "no floor",
        ],
    )
    def test_a_mutation_fails_the_pins(self, mutant: str) -> None:
        curves = _family()
        nus, values = self._mutants(curves)[mutant]
        assert _mapping_failures(curves, nus, values) != [], mutant

    def test_the_uncentred_witness_differs_by_more_than_a_factor_of_two(self) -> None:
        column = _family()[:, 3]
        assert kish_count(column, centered=False) > 2.0 * kish_count(column)

    def test_the_band_uses_each_c_j_on_its_scale(self) -> None:
        from cleverly.inference import make_estimate, simultaneous_bands

        curves = _family(n=4_000, seed=1)
        names = [f"x{j}" for j in range(curves.shape[1])]
        estimates = {
            name: make_estimate(
                name, 0.5, 0.1 * curves[:, j], n=curves.shape[0], rule_widened=False
            )
            for j, name in enumerate(names)
        }
        bands = simultaneous_bands(estimates, random_state=3, band_reference="t", kind="normal")
        c = bands.critical_value
        assert bands.reference == "t"
        for j, name in enumerate(names):
            nu = _own_df(curves)[j]
            assert bands.reference_df[name] == pytest.approx(nu, rel=1e-12)
            expected = stats.t.isf(stats.norm.sf(c), nu)
            assert bands.critical_values[name] == pytest.approx(expected, rel=1e-12)
            half = (bands.bands[name][1] - bands.bands[name][0]) / 2.0
            assert half == pytest.approx(expected * estimates[name].std_error, rel=1e-12)


# --------------------------------------------------------------- 1. clusters, floor and zero


class TestTheDefinitionOnABand:
    def test_a_clustered_band_reads_the_centred_cluster_sums(self) -> None:
        from cleverly.inference import cluster_sums
        from cleverly.inference.multiplier import t_critical_values

        rng = np.random.default_rng(7)
        curves = rng.standard_t(3, size=(600, 2))
        cluster = np.repeat(np.arange(60), 10)
        nus, _ = t_critical_values(C, curves, cluster)
        summed = cluster_sums(curves, cluster)
        assert nus == pytest.approx(_own_df(summed), rel=1e-12)
        # A witness: the row-level count differs.
        assert not np.allclose(nus, _own_df(curves), rtol=1e-3)

    def test_a_zero_curve_has_infinite_df_and_the_critical_value_itself(self) -> None:
        from cleverly.inference.multiplier import t_critical_values

        curves = np.column_stack([np.zeros(50), np.random.default_rng(8).standard_normal(50)])
        nus, values = t_critical_values(C, curves)
        assert nus[0] == math.inf and values[0] == C


# ------------------------------------------------------------------------- 2. scale invariance


def test_a_bounded_estimate_reads_the_same_df_on_either_scale() -> None:
    from cleverly.inference import make_estimate, simultaneous_bands

    curves = _family(n=2_000, seed=2)
    estimates = {
        "p": make_estimate(
            "p", 0.7, 0.2 * curves[:, 0], n=2_000, rule_widened=False, bounds=(0.0, 1.0)
        ),
        "q": make_estimate(
            "q", 0.4, 0.3 * curves[:, 1], n=2_000, rule_widened=False, bounds=(0.0, 1.0)
        ),
    }
    logit = simultaneous_bands(estimates, random_state=1, band_reference="t", band_scale="logit")
    identity = simultaneous_bands(
        estimates, random_state=1, band_reference="t", band_scale="identity"
    )
    assert logit.reference_df == identity.reference_df
    assert logit.critical_values == identity.critical_values
    assert logit.scales["p"] == "logit" and identity.scales["p"] == "identity"


# ------------------------------------------------------------------------------------ 4. limit


class TestTheLimit:
    def test_the_mapping_tends_to_c(self) -> None:
        from cleverly.inference.multiplier import _t_quantile

        assert 0.0 <= _t_quantile(C, 1e11) - C < 1e-9

    def test_an_infinite_df_returns_c_without_scipy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cleverly.inference.multiplier import _t_quantile

        def refuse(*args: object, **kwargs: object) -> float:
            raise AssertionError("scipy was called at df=inf")

        monkeypatch.setattr(stats.t, "isf", refuse)
        assert _t_quantile(C, math.inf) == C

    def test_a_gaussian_band_at_a_million_rows_follows_the_expansion(self) -> None:
        from cleverly.inference import make_estimate, simultaneous_bands

        n = 1_000_000
        curves = np.random.default_rng(9).standard_normal((n, 3))
        curves -= curves.mean(axis=0)
        estimates = {
            f"g{j}": make_estimate(f"g{j}", 0.0, curves[:, j], n=n, rule_widened=False)
            for j in range(3)
        }
        normal = simultaneous_bands(estimates, random_state=0, kind="normal")
        t_band = simultaneous_bands(estimates, random_state=0, kind="normal", band_reference="t")
        c = normal.critical_value
        assert t_band.critical_value == c
        for name in estimates:
            ratio = (t_band.bands[name][1] / normal.bands[name][1]) - 1.0
            predicted = (c**2 + 1.0) / (4.0 * t_band.reference_df[name])
            assert ratio > 0.0
            assert abs(ratio / predicted - 1.0) <= 0.10, name
            assert predicted < 2e-5


# ----------------------------------------------------------------------------- 6. refusals


class TestTheRefusals:
    @pytest.mark.parametrize("reference", ["normal", "t"])
    def test_a_t_reference_estimate_is_refused_under_either_reference(self, reference: str) -> None:
        from cleverly.exceptions import CapabilityError
        from cleverly.inference import make_estimate, simultaneous_bands
        from cleverly.inference.multiplier import T_REFERENCE_BAND_REFUSAL

        curves = _family(n=200)
        estimates = {
            "a": make_estimate("a", 0.0, curves[:, 1], n=200, rule_widened=False, reference_df=8),
            "b": make_estimate("b", 0.0, curves[:, 2], n=200, rule_widened=False),
        }
        with pytest.raises(CapabilityError) as raised:
            simultaneous_bands(estimates, band_reference=reference)  # type: ignore[arg-type]
        assert raised.value.refusal == T_REFERENCE_BAND_REFUSAL

    def test_an_unknown_reference_is_a_value_error(self) -> None:
        from cleverly.estimators import TMLE
        from cleverly.inference import make_estimate, simultaneous_bands
        from cleverly.longitudinal import LTMLE

        curves = _family(n=200)
        estimates = {
            name: make_estimate(name, 0.0, curves[:, j], n=200, rule_widened=False)
            for j, name in ((1, "a"), (2, "b"))
        }
        message = "band_reference must be 'normal' or 't'; got 'student'"
        with pytest.raises(ValueError, match=message):
            simultaneous_bands(estimates, band_reference="student")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=message):
            TMLE(band_reference="student")
        with pytest.raises(ValueError, match=message):
            LTMLE({"always": 1}, band_reference="student")


# --------------------------------------------------------------------- 8. unchanged default


def _replay_estimates() -> dict[str, object]:
    """The estimates the base commit's band was recorded on (1702164b)."""
    from cleverly.inference.influence import make_estimate

    n = 600
    rng = np.random.default_rng(20261112)
    base = rng.standard_normal((n, 3))
    heavy = np.zeros(n)
    heavy[:6] = rng.standard_normal(6) * 20
    curves = np.column_stack([base[:, 0], 0.5 * base[:, 0] + base[:, 1], heavy, base[:, 2]])
    curves -= curves.mean(axis=0)
    return {
        "p": make_estimate(
            "p", 0.8, 0.3 * curves[:, 0], n=n, rule_widened=False, bounds=(0.0, 1.0)
        ),
        "d": make_estimate(
            "d", 0.2, 0.4 * curves[:, 1], n=n, rule_widened=False, bounds=(-1.0, 1.0)
        ),
        "h": make_estimate("h", 1.5, curves[:, 2], n=n, rule_widened=False),
        "r": make_estimate(
            "r",
            1.3,
            0.2 * curves[:, 3],
            n=n,
            rule_widened=False,
            scale="ratio",
            log_psi=float(np.log(1.3)),
        ),
    }


#: ``simultaneous_bands(estimates, random_state=7, n_replicates=2000, ...)`` at the base commit
#: 1702164b, printed with ``repr``: the critical value and each band.
BASE_BANDS: dict[tuple[str, str], tuple[float, dict[str, tuple[float, float]]]] = {
    ("logit", "mammen"): (
        2.531098523523079,
        {
            "p": (0.7673080230956884, 0.8291217931903456),
            "d": (0.15486719361260048, 0.24429973498649993),
            "h": (1.3199683721348965, 1.6800316278651035),
            "r": (1.273798731554696, 1.326740212668702),
        },
    ),
    ("logit", "rademacher"): (
        2.3405344443745837,
        {
            "p": (0.7698939829282454, 0.827051921079808),
            "d": (0.15829031457285403, 0.24099719373789452),
            "h": (1.3335227877622917, 1.6664772122377083),
            "r": (1.2757528840472816, 1.3247079596156066),
        },
    ),
    ("identity", "normal"): (
        2.445573208983076,
        {
            "p": (0.7701446247144939, 0.8298553752855062),
            "d": (0.15676715282026005, 0.24323284717973997),
            "h": (1.3260516049514839, 1.6739483950485161),
            "r": (1.2746753862777482, 1.325827750495022),
        },
    ),
}


def _shipped_band(estimates: dict[str, object], scale: str, kind: str) -> object:
    """The base commit's body: the critical value of ``std_error``, then the shipped endpoints."""
    from cleverly.inference import multiplier

    items = list(estimates.items())
    ic = np.column_stack([estimate.influence_curve for _, estimate in items])  # type: ignore[attr-defined]
    se = np.array([estimate.std_error for _, estimate in items])  # type: ignore[attr-defined]
    critical = multiplier.multiplier_critical_value(
        ic,
        se,
        n=600,
        n_replicates=2000,
        kind=kind,
        random_state=7,  # type: ignore[arg-type]
    )
    bands, _ = multiplier._band_endpoints(items, critical, scale)  # type: ignore[arg-type]
    return critical, bands


class TestTheUnchangedDefault:
    @pytest.mark.parametrize(("scale", "kind"), sorted(BASE_BANDS))
    def test_the_default_band_replays_the_base_commit(self, scale: str, kind: str) -> None:
        from cleverly.inference.multiplier import simultaneous_bands

        estimates = _replay_estimates()
        settings = {"random_state": 7, "n_replicates": 2000, "band_scale": scale, "kind": kind}
        default = simultaneous_bands(estimates, **settings)  # type: ignore[arg-type]
        named = simultaneous_bands(estimates, band_reference="normal", **settings)  # type: ignore[arg-type]
        # Bit for bit against the shipped body run in this process, and against each other.
        critical, bands = _shipped_band(estimates, scale, kind)
        assert default.critical_value == named.critical_value == critical
        assert default.bands == named.bands == bands
        assert default.reference == "normal" and default.reference_df == {}
        assert default.critical_values == {}
        # Against the numbers the base commit printed, to the rounding a library version moves.
        recorded_critical, recorded = BASE_BANDS[(scale, kind)]
        assert default.critical_value == pytest.approx(recorded_critical, rel=1e-12)
        for name, (low, high) in recorded.items():
            assert default.bands[name] == pytest.approx((low, high), rel=1e-12)

    def test_passing_the_t_values_under_normal_fails_the_replay(self) -> None:
        from cleverly.inference.multiplier import simultaneous_bands

        estimates = _replay_estimates()
        mutant = simultaneous_bands(
            estimates, random_state=7, n_replicates=2000, band_reference="t"
        )
        _, bands = _shipped_band(estimates, "logit", "mammen")
        assert mutant.bands["h"] != bands["h"]

    def test_a_different_standard_error_body_fails_the_replay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cleverly.inference import multiplier

        estimates = _replay_estimates()
        _, bands = _shipped_band(estimates, "logit", "mammen")

        shipped = multiplier._band_std_error

        def recomputed(estimate: object) -> float:
            # A body that moves one estimate's standard error by 1e-12, the size of a reordered
            # sum.  A common factor on every standard error would cancel against the critical
            # value, so the mutant moves one.
            value = shipped(estimate)  # type: ignore[arg-type]
            return value * (1.0 + 1e-12) if estimate.name == "h" else value  # type: ignore[attr-defined]

        monkeypatch.setattr(multiplier, "_band_std_error", recomputed)
        mutant = multiplier.simultaneous_bands(estimates, random_state=7, n_replicates=2000)
        assert mutant.bands != bands


# ------------------------------------------------------------------------------- 10. k = 1


def test_one_estimate_gets_a_t_interval_wider_than_its_own() -> None:
    from cleverly.inference import make_estimate, simultaneous_bands

    curve = _family(n=500, seed=4)[:, 0]
    estimate = make_estimate("one", 0.3, curve, n=500, rule_widened=False)
    bands = simultaneous_bands({"one": estimate}, band_reference="t")
    assert bands.critical_value == pytest.approx(stats.norm.ppf(0.975), rel=1e-15)
    nu = bands.reference_df["one"]
    assert nu == pytest.approx(_own_df(curve[:, None])[0], rel=1e-12)
    expected = stats.t.ppf(0.975, nu)
    low, high = bands.bands["one"]
    assert (high - low) / 2.0 == pytest.approx(expected * estimate.std_error, rel=1e-12)
    pointwise = estimate.ci
    assert low < pointwise[0] and high > pointwise[1]


# ----------------------------------------------------------- 11 and 13: the family rule
#
# No builder sets "constant_arm_cell_plugin" or "out_of_range_plugin" until the pointwise row
# (Fix-3) lands its statuses. These tests drive the band's two rules through two withheld
# statuses that exist today, patched in as the names the band reads, so the mechanism is pinned
# now. Tests 11 and 13 on stratified fits land with the statuses themselves.


@pytest.fixture
def stand_in_statuses(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    from cleverly.inference import multiplier

    monkeypatch.setattr(multiplier, "CONSTANT_ARM_CELL_STATUS", "constant_node_plugin")
    monkeypatch.setattr(multiplier, "OUT_OF_RANGE_STATUS", "stack_failure_plugin")
    return "constant_node_plugin", "stack_failure_plugin"


def _family_estimates(status: dict[str, str] | None = None) -> dict[str, object]:
    from cleverly.inference import make_estimate

    curves = _family(n=800, seed=5)
    status = status or {}
    return {
        name: make_estimate(
            name,
            psi,
            0.05 * curves[:, j],
            n=800,
            rule_widened=False,
            bounds=(0.0, 1.0),
            inference=status.get(name, "influence_curve"),  # type: ignore[arg-type]
        )
        for j, (name, psi) in enumerate((("a", 0.4), ("b", 0.5), ("c", 1.0)))
    }


class TestTheFamilyRule:
    def test_no_current_estimate_is_a_constant_cell_or_an_out_of_range_member(self) -> None:
        from cleverly.inference.multiplier import band_member, constant_arm_members

        estimates = _family_estimates({"c": "constant_node_plugin"})
        assert constant_arm_members(estimates) == ()
        assert not band_member(estimates["c"])  # type: ignore[arg-type]

    @pytest.mark.parametrize("reference", ["normal", "t"])
    def test_a_constant_cell_refuses_the_band_by_name(
        self, stand_in_statuses: tuple[str, str], reference: str
    ) -> None:
        from cleverly.exceptions import CapabilityError
        from cleverly.inference.multiplier import (
            constant_arm_band_refusal,
            constant_arm_members,
            median_bands,
            simultaneous_bands,
        )

        constant, _ = stand_in_statuses
        estimates = _family_estimates({"a": constant})
        assert constant_arm_members(estimates) == ("a",)  # type: ignore[arg-type]
        with pytest.raises(CapabilityError) as raised:
            simultaneous_bands(estimates, band_reference=reference)  # type: ignore[arg-type]
        assert raised.value.refusal == constant_arm_band_refusal(["a"])
        assert "the estimate a reads a treatment arm" in str(raised.value)
        with pytest.raises(CapabilityError) as repeated:
            median_bands([estimates, estimates], estimates, band_reference=reference)  # type: ignore[arg-type,list-item]
        assert repeated.value.refusal == constant_arm_band_refusal(["a"])

    def test_dropping_the_constant_member_would_band_another_family(
        self, stand_in_statuses: tuple[str, str]
    ) -> None:
        """The mutation control: the rule a withheld estimate follows today leaves a band."""
        from cleverly.inference.multiplier import simultaneous_bands

        constant, _ = stand_in_statuses
        estimates = _family_estimates({"a": constant})
        kept = {name: value for name, value in estimates.items() if name != "a"}
        assert set(simultaneous_bands(kept, band_scale="identity").bands) == {"b", "c"}  # type: ignore[arg-type]

    def test_an_out_of_range_member_keeps_its_band_behaviour(
        self, stand_in_statuses: tuple[str, str]
    ) -> None:
        from cleverly.exceptions import CapabilityError
        from cleverly.inference.multiplier import band_member, simultaneous_bands

        _, outside = stand_in_statuses
        withheld = _family_estimates({"c": outside})
        plain = _family_estimates()
        assert band_member(withheld["c"])  # type: ignore[arg-type]
        # The logit band refuses it with today's refusal, not the constant-cell one.
        for estimates in (withheld, plain):
            with pytest.raises(CapabilityError, match="lies on or outside its range"):
                simultaneous_bands(estimates)  # type: ignore[arg-type]
        # The identity band covers the full family, bit for bit as without the status.
        settings = {"random_state": 2, "band_scale": "identity"}
        assert (
            simultaneous_bands(withheld, **settings).bands  # type: ignore[arg-type]
            == simultaneous_bands(plain, **settings).bands  # type: ignore[arg-type]
        )


# ------------------------------------------------------------------------------ 7. support

T_OPTION = ("mammen", "logit", "t")


class TestTheSupport:
    @pytest.fixture(scope="class")
    def rows(self) -> dict[str, object]:
        from cleverly._support import DEFAULT_BAND_OPTION, SupportRow, classify_key
        from cleverly.datasets import make_linear_ate
        from cleverly.estimators import TMLE
        from tests.conftest import linear_in_sample
        from tests.unit.test_support_key import key_of

        frame = make_linear_ate(n=200, seed=3)[0]
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        kinds = frozenset({"point", "wald", "band"})

        def classified(**fields: object) -> object:
            row = SupportRow("RB", key, kinds, ("study-b",), **fields)  # type: ignore[arg-type]
            return classify_key(key, learners, rows=[row])

        short = (("band", "Row RB is red for the default band."),)
        return {
            "default only": classified(),
            "both": classified(band_options=(DEFAULT_BAND_OPTION, T_OPTION)),
            "t only": classified(band_options=(T_OPTION,)),
            "both, short": classified(
                band_options=(DEFAULT_BAND_OPTION, T_OPTION), shortfalls=short
            ),
            "t only, short": classified(band_options=(T_OPTION,), shortfalls=short),
            "default only, short": classified(shortfalls=short),
        }

    @staticmethod
    def _band(classification: object, **settings: object) -> object:
        from cleverly._support import Support, band_support

        supported = Support(
            "supported", "Row RB covers this configuration.", ("study-b",), row="RB"
        )
        return band_support(classification, "influence_curve", [supported], **settings)  # type: ignore[arg-type]

    def test_a_t_band_on_a_row_that_does_not_list_it_is_experimental(self, rows: dict) -> None:
        band = self._band(rows["default only"], options=T_OPTION)
        assert band.label == "experimental"  # type: ignore[attr-defined]
        assert "band_reference='t'" in band.reason  # type: ignore[attr-defined]
        assert self._band(rows["default only"]).label == "supported"  # type: ignore[attr-defined]

    def test_a_row_that_lists_it_gives_the_row_support(self, rows: dict) -> None:
        assert self._band(rows["both"], options=T_OPTION).label == "supported"  # type: ignore[attr-defined]
        assert self._band(rows["both"]).label == "supported"  # type: ignore[attr-defined]

    def test_a_row_that_lists_only_the_t_band_leaves_the_default_experimental(
        self, rows: dict
    ) -> None:
        assert self._band(rows["t only"], options=T_OPTION).label == "supported"  # type: ignore[attr-defined]
        default = self._band(rows["t only"])
        assert default.label == "experimental"  # type: ignore[attr-defined]
        assert "band_reference='normal'" in default.reason  # type: ignore[attr-defined]

    def test_a_band_shortfall_binds_the_default_band_and_not_a_listed_t_band(
        self, rows: dict
    ) -> None:
        for name in ("both, short", "t only, short"):
            assert self._band(rows[name], options=T_OPTION).label == "supported", name  # type: ignore[attr-defined]
            default = self._band(rows[name])
            assert default.label == "experimental", name  # type: ignore[attr-defined]
            assert "red for the default band" in default.reason, name  # type: ignore[attr-defined]
        # A t band on a row that does not list it still reads the shortfall.
        unlisted = self._band(rows["default only, short"], options=T_OPTION)
        assert unlisted.label == "experimental"  # type: ignore[attr-defined]
        assert "red for the default band" in unlisted.reason  # type: ignore[attr-defined]
        # A listed t band of another shape reads experimental for want of a cell of its shape,
        # not for the default band's shortfall.
        clustered = self._band(rows["both, short"], options=T_OPTION, shape="clustered")
        assert clustered.label == "experimental"  # type: ignore[attr-defined]
        assert "clustered t band" in clustered.reason  # type: ignore[attr-defined]
        assert "red for the default band" not in clustered.reason  # type: ignore[attr-defined]

    def test_dropping_the_shortfall_rule_fails(
        self, rows: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: with the default band's shortfall bound on every option, the
        listed t band reads experimental, which the test above refuses."""
        from cleverly import _support

        monkeypatch.setattr(_support, "DEFAULT_BAND_OPTION", T_OPTION)
        band = self._band(rows["both, short"], options=T_OPTION)
        assert band.label == "experimental"  # type: ignore[attr-defined]
        assert "red for the default band" in band.reason  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("row_id", "shortfall"),
        [("G08.in_sample", True), ("T30", True), ("T13", True), ("G23.cross_fitted", False)],
    )
    def test_the_licensed_rows_support_the_t_band(self, row_id: str, shortfall: bool) -> None:
        from cleverly._support import Classification, band_support, registered_rows, support_for

        row = next(r for r in registered_rows() if r.id == row_id)
        classification = Classification(row.key, {}, row)
        estimate = support_for(classification, "influence_curve")
        t_band = band_support(
            classification, "influence_curve", [estimate], options=T_OPTION, draws=10_000
        )
        assert t_band.label == "supported", t_band.reason
        assert "band-t-reference" in row.evidence
        default = band_support(classification, "influence_curve", [estimate], draws=10_000)
        assert default.label == ("experimental" if shortfall else "supported"), default.reason
        assert ("red for the default band" in default.reason) is shortfall
        assert "red for the default band" not in t_band.reason
        few = band_support(
            classification, "influence_curve", [estimate], options=T_OPTION, draws=1_000
        )
        assert few.label == "experimental"

    @pytest.mark.parametrize("shape", ["clustered", "weighted", "repeated"])
    def test_a_t_band_of_another_shape_is_experimental_on_every_row(
        self, rows: dict, shape: str
    ) -> None:
        for name in ("both", "t only"):
            band = self._band(rows[name], options=T_OPTION, shape=shape)
            assert band.label == "experimental", name  # type: ignore[attr-defined]
            assert f"No band-t-reference cell measures a {shape} t band." in band.reason  # type: ignore[attr-defined]
        unclassified = self._band(None, options=T_OPTION, shape=shape)
        assert f"a {shape} t band" in unclassified.reason  # type: ignore[attr-defined]
        # The default band of any shape keeps what its row says.
        assert self._band(rows["both"], shape=shape).label == "supported"  # type: ignore[attr-defined]

    def test_dropping_the_shape_check_fails(
        self, rows: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: a band_support that ignores ``shape`` licenses the band."""
        from cleverly import _support

        shipped = _support.band_support

        def ignoring(*args: object, **kwargs: object) -> object:
            kwargs.pop("shape", None)
            return shipped(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_support, "band_support", ignoring)
        band = ignoring(rows["both"], "influence_curve", [], options=T_OPTION, shape="clustered")
        assert band.label != "experimental" or "clustered t band" not in band.reason  # type: ignore[attr-defined]

    def test_band_shape_reads_the_fit(self) -> None:
        from cleverly._support import band_shape

        assert band_shape() == "plain"
        assert band_shape(weighted=True) == "weighted"
        assert band_shape(cluster=np.arange(3), weighted=True) == "clustered"
        assert band_shape(cluster=np.arange(3), repeated=True) == "repeated"


# --------------------------------------------------------------------------- the estimators


class TestTheEstimators:
    @pytest.fixture(scope="class")
    def frame(self) -> object:
        from cleverly.datasets import make_binary_outcome

        return make_binary_outcome(n=600, seed=11)[0]

    def _fit(self, frame: object, **settings: object) -> object:
        from cleverly.estimators import TMLE
        from tests.conftest import FAST_KWARGS

        covariates = [name for name in frame.columns if name.startswith("W")]  # type: ignore[attr-defined]
        return (
            TMLE(**{**FAST_KWARGS, "random_state": 0, "simultaneous": True, **settings})  # type: ignore[arg-type]
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()
        )

    def test_a_point_fit_builds_the_t_band_from_its_own_curves(self, frame: object) -> None:
        from cleverly.inference import simultaneous_bands

        normal = self._fit(frame)
        t_fit = self._fit(frame, band_reference="t")
        normal_band, t_band = normal.simultaneous, t_fit.simultaneous  # type: ignore[attr-defined]
        assert normal_band.reference == "normal" and t_band.reference == "t"
        assert t_band.critical_value == normal_band.critical_value
        direct = simultaneous_bands(
            dict(t_fit.estimates),
            random_state=0,
            band_reference="t",  # type: ignore[attr-defined]
        )
        assert direct.bands == t_band.bands
        for name, (low, high) in t_band.bands.items():
            normal_low, normal_high = normal_band.bands[name]
            assert low <= normal_low and high >= normal_high, name
        # No registered cell lists the t triple yet, so the t band reads experimental.
        assert t_band.support.label == "experimental"
        text = str(t_fit.summary())  # type: ignore[attr-defined]
        assert "Simultaneous" in text

    def test_the_method_configuration_reaches_the_engine(self) -> None:
        from cleverly import Inference, TMLEMethod

        method = TMLEMethod(inference=Inference(band_reference="t"))
        assert method.inference.band_reference == "t"
