# DR-TMLE: an interval when the recorded assignment rule is hard to model

This test of change is about inference rather than the estimate. An ordinary TMLE is doubly
robust for *consistency* and singly robust for *inference*. This page shows what that distinction
means for a program that cannot model its own rollout, and what the DR-TMLE variant does about it.

Read [DR-TMLE](../technical-reference/dr-tmle/index.md) for the applied framing, and the
[DR-TMLE production contract](../technical-reference/dr-tmle/index.md) for the theorem, the refusals, and the release claim.
The contract is authoritative. Read
[its release claim](../technical-reference/dr-tmle/index.md#the-release-claim-in-one-paragraph) before you rely on this variant.

## The applied question

The navigation evaluation is being repeated. This time the analyst has a specific worry about
which nuisance function can be estimated well.

The outcome side is in good shape. Transition scores depend on discharge risk, prior utilization,
medication burden, and age in ways a flexible model can learn from thousands of discharges.

The assignment side is difficult. The program logged every common cause of assignment and outcome,
but its operational rule contains thresholds, interactions, and rotating capacity constraints. A
simple logistic model of assignment is therefore misspecified even though the causal adjustment set
is measured.

The analyst can live with a crude assignment model. Double robustness says the estimate stays
consistent as long as the outcome regression is good. The question is whether the *interval* stays
valid too.

This distinction is essential. If an unrecorded discharge-team judgement affects both assignment
and recovery, exchangeability fails. DR-TMLE does not repair that design failure.

## Why this method

The discharge team's assignment model is the one this analysis doubts. DR-TMLE exists for that
case. It solves two further score equations, built from reduced-dimension regressions, so the
interval can stay valid when one primary nuisance is inconsistent.

Ordinary TMLE is doubly robust for the point estimate and singly robust for the interval. One
consistent nuisance still gives a consistent estimate. The interval needs both nuisances to
converge fast enough. The
[DR-TMLE reference entry](../technical-reference/dr-tmle/index.md#what-this-solves) derives the
remainder term and the rate conditions, and tabulates what the variant buys and what it costs.

Read that table before you choose this method. Most of its rows are reasons not to. With both
nuisances consistent the corrections converge to zero, so the extra cost buys nothing. The
corrected interval is also not narrower than the ordinary one.

## The data

The law is `make_nonlinear_ate` again. A gradient-boosted learner is approximately right for the
outcome regression on this law. A logistic regression is wrong for the assignment mechanism, by
construction. That pairing is the analyst's situation.

```python
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2_000, seed=55)
frame = frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "prior_utilization",
        "W3": "medication_burden",
        "W4": "age",
    }
)
print("population ATE:", truth["ate"])
```

## Design and identification

Nothing changes in the question. DR-TMLE targets the same parameter under the same assumptions.

The study therefore keeps the shared eligibility, time zero, protocol version, and reserved
capacity. It also keeps every common cause in the adjustment set. The deliberate failure on this
page is statistical misspecification of the recorded assignment mechanism. It is not an omitted
common cause.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "prior_utilization", "medication_burden", "age"),
    ),
)
effect = study.identify(ATE(reference=0))

for method in effect.available_methods():
    print(method.name, method.available)
```

The availability check matters more here than elsewhere. DR-TMLE refuses continuous treatment, ATT
and ATC, the intervention axes, MSM projections, and composition with C-TMLE. Each refusal names
what a derivation would need. The list is in [the contract](../technical-reference/dr-tmle/supported-estimands.md#refused-by-name).

## Estimate

The primary nuisances are the analyst's: a flexible outcome regression and a crude assignment model.
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
approximately zero, on this fit, with an assignment model the analyst already believes is wrong.
Convergence is a property of the targeting step. It is not evidence that the reduced regressions or
the primary nuisances converge at the rates the theorem needs. Those are rate conditions on
estimated functions, and a fit's own output cannot verify them. The contract devotes
[a whole section](../technical-reference/dr-tmle/diagnostics.md#solved-scores-do-not-establish-nuisance-consistency) to this, and
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
[implementation validation grid](../technical-reference/method-evidence/validation-grid.md),
because no registered repeated-sampling study covers it. Read that absence as part of the claim.

## Where to go next

If your worry is which baseline variables belong in the assignment model rather than how well any of
them can be fitted, the entry for that is [collaborative TMLE](collaborative-tmle.md). The two do
not compose, and the refusal is by construction rather than a gap. A reduced regression conditions
on the fitted assignment probability as a covariate, and a collaborative one is deliberately not an
estimate of the true probability.
