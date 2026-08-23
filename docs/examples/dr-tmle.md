# DR-TMLE: an interval when nobody recorded why a ward adopted

This test of change is about inference rather than about the estimate. An ordinary TMLE is doubly
robust for *consistency* and singly robust for *inference*. This page shows what that distinction
means for a program that cannot model its own rollout, and what the DR-TMLE variant does about it.

Read [DR-TMLE](../technical-reference/dr-tmle.md) for the applied framing, and the
[DR-TMLE production contract](../drtmle.md) for the theorem, the refusals, and the release claim.
The contract is authoritative. Read
[its release claim](../drtmle.md#the-release-claim-in-one-paragraph) before you rely on this variant.

## The applied question

The rounding evaluation is being repeated, and this time the analyst has a specific worry about
which half of the analysis can be trusted.

The outcome side is in good shape. Experience scores depend on acuity, prior admissions, length of
stay, and age in ways a flexible model can learn from thousands of discharges.

The adoption side is not. Wards adopted because a manager decided to. Managers weighed ward culture,
how stretched the staffing felt that quarter, and their own conviction. None of that was written
down. Any model of adoption built from the recorded variables is a crude approximation of a decision
process the data does not contain.

The analyst can live with a crude adoption model. Double robustness says the estimate stays
consistent as long as the outcome regression is good. The question is whether the *interval* stays
valid too.

This page analyses one hospital's patients, so patients are treated as independent.

## Why this method

The second-order remainder of an ordinary TMLE is a product of two errors:

$$
R_2 = \lVert \hat g - g_0 \rVert \cdot \lVert \hat Q - Q_0 \rVert .
$$

One inconsistent nuisance still leaves the product shrinking, so the estimate stays consistent. The
interval needs the strictly stronger condition that $\sqrt{n} R_2 \to 0$. With both nuisances
converging at $n^{-1/4}$ the product delivers it. With only one, the bad factor stops shrinking,
the remainder becomes first order in the good one's error, and no nonparametric estimator drives
that below $n^{-1/2}$.

The estimator then stops being asymptotically linear. Its bias does not grow. Its coverage decays as
the sample grows, which is the opposite of what a program expects from more data.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| you doubt one nuisance and still want an interval | an interval entitled to be believed under weaker conditions | the reduced regressions are refitted inside the alternation, so a fit costs several rounds of several regressions |
| both nuisances are in fact consistent | nothing. The corrections vanish row by row, and the fit **is** the ordinary efficient estimator | the extra cost, for no gain. This is the case the variant is not for |
| you want a narrower interval | this is not that | the corrected curve is the estimator's own influence function, and it is not the efficient one |

DR-TMLE solves two further score equations, built from reduced-dimension regressions, so that valid
inference can survive one inconsistent primary nuisance.

## The data

The law is `make_nonlinear_ate` again. A gradient-boosted learner is approximately right for the
outcome regression on this law. A logistic regression is wrong for the adoption mechanism, by
construction. That pairing is the analyst's situation.

```python
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2_000, seed=55)
frame = frame.rename(
    columns={
        "Y": "experience_score",
        "A": "hourly_rounding",
        "W1": "acuity",
        "W2": "prior_admissions",
        "W3": "length_of_stay",
        "W4": "age",
    }
)
print("population ATE:", truth["ate"])
```

## Design and identification

Nothing changes in the question. DR-TMLE targets the same parameter under the same assumptions.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="experience_score",
        treatment="hourly_rounding",
        adjustment=("acuity", "prior_admissions", "length_of_stay", "age"),
    ),
)
effect = study.identify(ATE(reference=0))

for method in effect.available_methods():
    print(method.name, method.available)
```

The availability check matters more here than elsewhere. DR-TMLE refuses continuous treatment, ATT
and ATC, the intervention axes, MSM projections, and composition with C-TMLE. Each refusal names
what a derivation would need. The list is in [the contract](../drtmle.md#refused-by-name).

## Estimate

The primary nuisances are the analyst's: a flexible outcome regression and a crude adoption model.
The reduced regressions are the extra machinery the variant needs.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, DRTMLEMethod, ModelSpec, Runtime, TMLEMethod

models = ModelSpec(
    outcome_learner=HistGradientBoostingRegressor(random_state=55),
    treatment_learner=LogisticRegression(max_iter=1000),
)
folds = CrossFitting(n_folds=3, learner_folds=2)
runtime = Runtime(random_state=55, n_jobs=1)

doubly_robust = effect.estimate(
    method=DRTMLEMethod(
        models=models,
        cross_fitting=folds,
        runtime=runtime,
        guard=("Q", "g"),
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=LinearRegression(),
    )
)
print(doubly_robust.summary())
print("population ATE:", truth["ate"])
```

`guard=` says which extra equations to solve, and therefore which corrections the reported curve
subtracts. Both are on by default. `guard=("g",)` solves one of them and reports
$D = D^{*} - D^{*}_{Q}$.

## The failure mode: consistency without valid inference

Start with the claim that is exactly checkable on one fit. An empty guard solves no extra equation,
so it is a plain TMLE.

```python
ordinary = effect.estimate(method=TMLEMethod(models=models, cross_fitting=folds, runtime=runtime))
empty_guard = effect.estimate(
    method=DRTMLEMethod(
        models=models,
        cross_fitting=folds,
        runtime=runtime,
        guard=(),
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=LinearRegression(),
    )
)
print("ordinary TMLE psi:", ordinary["ate"].psi)
print("guard=()      psi:", empty_guard["ate"].psi)
print("identical:", ordinary["ate"].psi == empty_guard["ate"].psi)
```

The two agree bit for bit. That is the contract's own statement, and it is worth confirming, because
it fixes what the variant is: the same estimator plus extra equations, not a different target.

Now compare the ordinary fit with the guarded one.

```python
def show(label, result):
    point = result["ate"]
    low, high = point.ci
    print(
        f"{label:16s} psi={point.psi:6.3f}  se={point.std_error:6.4f}  CI=({low:.3f}, {high:.3f})"
    )


show("ordinary TMLE", ordinary)
show("DR-TMLE", doubly_robust)
print("population ATE:", truth["ate"])
```

The point estimates are close, and the intervals are similar in width. **That is the expected
result, and it is the hardest thing about this variant to teach.**

The difference DR-TMLE makes is not visible in one sample. Both estimators are consistent here,
because the outcome regression is good. What differs is the repeated-sampling behaviour of the
interval as the program collects more discharges. The ordinary interval's coverage decays, because
its remainder is first order in the outcome regression's error. The corrected interval is entitled
to be believed under the weaker condition.

A single fit cannot show a decaying coverage rate. Anyone who claims otherwise is reading noise.

What a single fit *can* show is the size of the corrections that were solved away.

```python
print(doubly_robust.diagnostics.corrections().summary())
```

Each row is one correction equation, per arm, with the score it solved and the residual. The report
also states whether the truncations were active, because the theorem's scope requires that they are
not.

## How far to trust this

```python
print(doubly_robust.diagnostics.score_equations().summary())
print(doubly_robust.validate().summary())
```

The score report ends with the sentence that governs how the interval should be read. Validity is
not efficiency. The curve reported is $D = D^{*} - D^{*}_{Q} - D^{*}_{g}$. It is entitled to be
believed under weaker conditions than $D^{*}$, rather than efficient under them. The union model it
stays valid over is larger than the model it is efficient in.

The most important limitation on this page is not a diagnostic result. It is what the diagnostics
cannot do.

**Solved scores do not establish nuisance consistency.** Every score above converged to
approximately zero, on this fit, with an adoption model the analyst already believes is wrong.
Convergence is a property of the targeting step. It is not evidence that the reduced regressions or
the primary nuisances converge at the rates the theorem needs. Those are rate conditions on
estimated functions, and a fit's own output cannot verify them. The contract devotes
[a whole section](../drtmle.md#6-solved-scores-do-not-establish-nuisance-consistency) to this, and
calls it the single most important thing on that page.

| layer | establishes | does not establish |
| --- | --- | --- |
| `guard=()` equality | the variant reduces exactly to the ordinary estimator when no equation is solved | anything about the guarded fit |
| the corrections report | the extra equations were solved, and whether any truncation was active | that the reduced regressions are consistent |
| the score report | the targeting converged on the corrected curve | the rate conditions behind the interval |
| the theorem and its checks | the implementation computes what Theorem 1 derives, against exact laws, the Gateaux derivative, and the remainder identities | that your fitted nuisances satisfy the theorem's hypotheses |

DR-TMLE ships under **conditional validity**. The interval is valid conditional on the practitioner
obtaining adequate primary and reduced-regression fits. This variant has a method entry and a
contract, and it has no row in the
[implementation validation grid](../technical-reference/index.md#implementation-validation-grid),
because no registered repeated-sampling study covers it. Read that absence as part of the claim.

## Where to go next

If your worry is which ward characteristics belong in the adoption model rather than how well any of
them can be fitted, the entry for that is [collaborative TMLE](collaborative-tmle.md). The two do
not compose, and the refusal is by construction rather than a gap. A reduced regression conditions
on the fitted adoption probability as a covariate, and a collaborative one is deliberately not an
estimate of the true probability.
