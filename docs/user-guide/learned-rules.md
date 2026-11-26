# Learned treatment rules

Use `LearnedRuleValue` to estimate the value of a treatment rule that the fit learns from the data.
The fit learns one rule inside each training fold and reports an average of the values of those
rules, weighted by fold (equal weights in the example below). The interval can under-cover when the effect is zero or near zero for a share of units.
[Learned rules](../technical-reference/point-treatment-tmle.md#learned-rules) states the
conditions.

Two registered studies measure the 95% interval over 6,000 replications at n = 2,000. The table
gives its coverage at each law.

| law | coverage | reading |
| --- | --- | --- |
| no effect for any unit | 0.8938, with a 99% interval of 0.8832 to 0.9038 | `under-covers at the exceptional law`, in the [boundary study](../technical-reference/method-evidence/learned-rule-cvtmle-boundary.md) |
| a weak effect, $0.15 W_1$ on the logit scale | 0.8948, with a 99% interval of 0.8842 to 0.9048 | `under-covers at the weak_blip law`, in the same study |
| two laws that meet the published conditions, one with a correctly specified outcome learner and one with a misspecified one | 0.9465 and 0.9413 | every verdict passes, in the [gated study](../technical-reference/method-evidence/learned-rule-cvtmle.md) |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from cleverly import CrossFitting, LearnedRuleValue, ModelSpec, TMLEMethod

fold_evaluated = TMLEMethod(
    models=ModelSpec(
        outcome_learner=make_pipeline(
            PolynomialFeatures(2, interaction_only=True, include_bias=False),
            LogisticRegression(max_iter=1000),
        ),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True),
)
learned = study.estimate(LearnedRuleValue(), method=fold_evaluated, random_state=3)
record = learned.extra["learned_rule"]
(split,) = record.splits  # one split into five folds
shares = split.arm_shares  # per fold: (share left untreated, share treated)
```

The rule of fold $v$ treats a unit when the outcome regression fitted on the other folds predicts
a higher outcome under treatment. A tie assigns control. The outcome learner therefore sets the
rule, so give it the treatment-by-covariate terms that the rule can depend on.

The treatment can also have three or more options. The rule then gives each unit the option with
the highest predicted outcome, and a tie goes to the first option in sorted order. A
[study](../technical-reference/method-evidence/learned-rule-arms.md) measured this with three
options, ten folds scored one at a time (`fold_evaluation=True`) and 2,000 rows, on data where
one option's predicted outcome was clearly the highest for almost every unit. Over 3,200 samples
its 95% interval covered the target 0.942 of the time. With three options and
`fold_evaluation=True`, cleverly marks the estimate supported at any number of folds, although
the study measured ten only.

With four or more options, or with three and any other setting on this page (the default
`CrossFitting()`, the in-sample fit, weights, clusters, strata, several splits or a comparison
with a fixed policy), cleverly marks the estimate experimental. The three rule learners described
in [Learned rules](../technical-reference/point-treatment-tmle.md#learned-rules),
`loss="blip"`, `loss="weighted_classification"` and `loss="ep"`, compare two options only, so with more than
two they raise `CapabilityError` and name the default rule as the way out.

A rule can also read only the covariates you choose, which makes it easier to apply and to
explain. Pass `LearnedRule(loss="ep", modifiers=...)` as the rule. Each fold then fits a curve of
the treatment effect against those covariates, with the EP learner of
[effects that differ between people](conditional-effects.md), and treats where the curve is
above zero. The curve corrects the outcome model's predictions with the treatment model. So
when the treatment model is right, the curve still approaches the closest curve that the
default shapes can draw, even if the outcome model is wrong. The default shapes are up to six
smooth waves for each numeric covariate and one step for each level of a categorical one,
added together, so with several covariates the curve is a sum of one curve per covariate.

```python
from cleverly.bases import Modifiers
from cleverly.interventions import LearnedRule

by_w1 = LearnedRule(loss="ep", modifiers=Modifiers(["W1"], domains={"W1": (-5, 5)}))
targeted = study.estimate(LearnedRuleValue(rule=by_w1), method=fold_evaluated, random_state=3)
```

The covariates must be ones the study adjusts for, and each continuous one needs a range that
holds every value. `OptimalRuleValue` with this rule needs all of the adjustment covariates,
because the best rule may use any of them. The curve also picks its own flexibility inside each
fold, so a fit takes noticeably longer than one with the default rule.

A [study](../technical-reference/method-evidence/learned-rule-ep.md) measured this rule with the
default shapes on two simulated datasets. With 16,000 rows its 95% interval covered the target
95% of the time, and with 2,000 rows 94% to 95% of the time. With 1,000 rows, on the dataset
where the outcome model was wrong, it covered 92% of the time, because the reported standard
error was about 9% too small. When the treatment has no effect for anyone, the curve has no
fixed sign, so the folds disagree on whom to treat, and the interval covered only 90% of the
time. cleverly cannot detect that case.

With the default shapes, two options, each fold scored on its own (`fold_evaluation=True`, as
above), no weights, no groups and one split, cleverly marks the estimate supported. With your own
`sieve=` or `learner=`, or with weights, groups, several splits or a comparison with a fixed
policy, it marks the estimate experimental. `OptimalRuleValue` below, with this rule, is
supported only when you pass the known treatment probabilities.

The reported value is a data-adaptive target. It depends on the realized split and on the fitted
rules. It is not the value of one rule fitted on all rows (the in-sample fit below), and it is not
the value of the best possible rule, which `OptimalRuleValue` below asks for. The table gives
what the result holds.

| where | content |
| --- | --- |
| `learned.estimates` | one parameter, `ey_learned_rule[learned rule]` |
| `learned.extra["learned_rule"].splits` | for each split into folds, the fold sizes, the fold estimates, the share of each fold's rows that the rule gives each option (`arm_shares`, in the order of `record.arms`), and quantiles of the gap between the two highest predicted outcomes (`gap_quantiles`). Under the default `CrossFitting()` the fold estimates describe each fold's part of one pooled estimate, not separate estimates |
| `learned.nuisance.regimes.values[:, 1, 0]` | the rule of each row with two options, `1.0` for treat. With more options, column `a` is `1.0` for the option at position `a` of `record.arms`, and `cleverly.interventions.learned.realized_arms` returns those positions |
| `learned.summary()` | a line that states the target |

Use `CrossFitting(n_folds=5, fold_evaluation=True)`, as above. It scores each fold's rule on that
fold alone and gives every fold equal weight, and the studies above measure that setting. The
default `CrossFitting()` also runs: it scores all held-out rows together and weights each fold by
its share of the rows (of the total weight, when you pass `weights=`). A
[third study](../technical-reference/method-evidence/learned-rule-stacked.md) measures it with
ten folds at n = 2,000, on rows, with known weights and with clusters, and cleverly marks the
default report `supported` in those settings. Over 8,800 samples of each setting its 95% interval
covered the target 0.943 to 0.949 of the time. Prefer it when you pass `id=` and cluster sizes
vary with the outcome: there the fold-evaluated estimate carries a small bias, about 0.11
standard errors in that study.

`CrossFitting(enabled=False)` runs too, and it answers a different question. It learns one rule
from all the rows and reports the value of that one rule, scored on the same rows that chose it.
Its interval rests on two things: the rule settles as the data grow, and the models are simple
enough not to chase noise in the rows that chose the rule. A model with a fixed set of terms and
nothing selected from the data is simple enough. cleverly reads this from the model's class: a
`LogisticRegression`, a `LinearRegression`, a `GLM`, or a class marked
`cleverly_learner_family = "parametric"` passes, and any other model, the pipeline above
included, makes the fit warn with `InSampleLearnerWarning`.

To declare that the pipeline above has fixed terms, subclass it and add the mark:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from cleverly import CrossFitting, LearnedRuleValue, ModelSpec, TMLEMethod


class FixedTerms(Pipeline):
    cleverly_learner_family = "parametric"


in_sample = TMLEMethod(
    models=ModelSpec(
        outcome_learner=FixedTerms(
            [
                ("terms", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
                ("fit", LogisticRegression(max_iter=1000)),
            ]
        ),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(enabled=False),
)
full_sample = study.estimate(LearnedRuleValue(), method=in_sample, random_state=3)
```

The mark is your statement that the model has fixed terms, and cleverly does not test it. With
it the fit gives no warning. The result is marked supported when both models have fixed terms,
the rule comes from the outcome model as above, the outcome and the treatment are binary, and the
fit has no weights, clusters or strata. An
[in-sample study](../technical-reference/method-evidence/learned-rule-in-sample.md) measures
this fit at n = 2,000. Over 14,400 samples of each of two settings, its 95% interval covered the
value of the chosen rule 0.947 to 0.950 of the time.

The in-sample fit chooses the rule and estimates its value on the same rows, so the estimate
leans optimistic. When the rule is stable this is small: about 0.16 to 0.21 of a standard error
at 2,000 rows in that study, the only size it measured, and theory expects more at smaller
sizes. When the estimated benefit of treatment is close to zero for many people, the rule can
change from sample to sample, and the lean grows: about 1.07 standard errors in the 19% of
samples where the rule changed. The interval still covered about 94% of the time in that case,
and cleverly cannot tell that it has happened. The cross-fitted estimate (the default) scores
each rule on rows that did not choose it, and it does not lean this way.

To estimate the value of the best possible rule instead, ask for `OptimalRuleValue()` in place of
`LearnedRuleValue()`. The best rule gives every unit the option with the highest average outcome
among units with the same covariates. The fit and its numbers are the same as above, reported as
`ey_optimal_rule[learned rule]`, and only the question changes. The answer is the best rule's value
only if the outcome model (or the rule learner) can learn the true pattern of benefit, the benefit is exactly zero for
almost no unit, and the learned rules settle on the best rule fast enough as the data grow. When
the outcome model misses the pattern, the fit estimates the value of a worse rule, which sits below
the best value, and its interval need not contain the best value.

An [optimal-value study](../technical-reference/method-evidence/learned-rule-optimal-value.md)
measures this fit when the treatment probabilities are known, as in a randomized trial, and
declared with `treatment_probabilities=` ([declare the design](data-design.md)). On one setting
with 2,000 rows and an outcome model that matched the truth, its 95% interval covered the best
rule's value 0.949 of the time over 13,200 samples. The estimate sits slightly below the best
value, by about a tenth of a standard error there, because the learned rules are a little worse
than the best rule, and that gap shrinks as the data grow. When cleverly fits the treatment
probabilities instead, or the fit uses clusters, several splits or a rule learner other than the
`LearnedRule(loss="ep")` rule above, it marks the result experimental.

No setting tells cleverly whether the outcome model learned the true pattern of benefit, or whether
the benefit is zero for many people. In the study, an outcome model that missed the pattern gave
an interval that contained the best value in 9% of samples, and a setting where the benefit was
zero for 60% of people gave 91%. The fit takes `weights=` (then the answer is the best rule's
value in the population the weights describe), `strata=` (the best rule's value inside each
group), the in-sample fit and an intermediate held at a value, each marked experimental.

Two checks of hidden confounding run on this fit. The omitted-variable bound, the range the
estimate could move to under left-out common causes of a stated strength, reads that strength as
holding for every rule. Its range reaches further above the estimate than below it, because hidden
causes could also make another rule the best one, and it has no confidence limits.
`simulated_confounding` learns the rule again in each simulated scenario, so the rule moves with
the confounding. The
[refusals of the value of the optimal rule](../technical-reference/scope-and-refusals.md#not-written-yet)
list the settings where neither runs.
[The value of the optimal rule](../technical-reference/point-treatment-tmle.md#learned-rules)
states the conditions.

To compare the learned rule with a fixed policy, such as treating everyone, ask for
`LearnedRuleContrast(against=(Static(1),))`, where `Static(1)` treats everyone. The fit reports
three rows: the learned rule's value, the fixed policy's value `ey_known_regime[always 1]`, and
their difference `ate_learned_rule[learned rule vs always 1]`. The same `TMLE` call is
`TMLE(learned_rule=LearnedRule(), interventions=[Static(1)])`, with the same numbers.

Both values come from the same rows and the same fitted models, and the interval of the
difference uses the correlation between the two estimates, measured on those rows. No published
source proves that interval for a learned rule, so a
[study](../technical-reference/method-evidence/learned-rule-contrast.md) measured it instead.
With 2,000 rows, ten folds and fixed-term logistic models, beside always-treat and never-treat,
each 95% interval covered its target 0.946 to 0.956 of the time over 5,600 samples. When the
outcome model left out a term the treated outcomes needed, coverage stayed between 0.942 and
0.951. That study measured a correct treatment model only.

`against=` takes more than fixed policies. A `Stochastic` plan, such as offering treatment to
half of the eligible people at random, is compared the same way. `NaturalCourse()` compares the
learned rule with current practice: its row `ey_natural_course[natural course]` is the average
outcome under the treatment people actually got, and the difference reads as the gain from
switching to the rule. That interval is right when the outcome model is right or the treatment
chances are supplied. When the outcome model is wrong and the treatment chances are estimated,
no result says the interval errs on the safe side, so read it as approximate. A plain arm
estimand such as `ate` refuses beside a learned rule, and its message names these spellings.
`OptimalRuleContrast(against=(Static(1),))` compares the best rule's value with the fixed policy
instead, reported as `ate_optimal_rule[learned rule vs always 1]`.

Each row carries a support label. Rows comparing the learned rule with fixed policies read
supported when the treatment has two options, the fit uses the default cross-fitting with one
split and the rule comes from the outcome model, with no weights, clusters or strata. The
best-rule comparison, `OptimalRuleContrast`, reads supported only when the treatment chances are
supplied (`treatment_probabilities=`), and experimental when the fit estimates them
([the value of the optimal rule](../technical-reference/point-treatment-tmle.md#learned-rules)).
Rows comparing the learned rule with a random plan or with current practice read supported in
the same setting, from their own
[study](../technical-reference/method-evidence/learned-rule-contrast-comparators.md).
[Contrasts with known regimes](../technical-reference/point-treatment-tmle.md#learned-rules)
lists which settings read supported, and the
[refusal table](../technical-reference/scope-and-refusals.md#not-written-yet) lists the
comparisons that refuse.

To learn a rule while holding a yes-or-no variable that the treatment affects (an intermediate,
such as whether a patient took a follow-up dose) at one of its values, declare the variable with
`PointTreatment(intermediate="Z")` and ask for `LearnedRuleValue(intermediate=0.0)` or
`LearnedRuleValue(intermediate=1.0)`. The `TMLE` estimator gives both values at once:
`TMLE(learned_rule=LearnedRule()).fit(..., intermediate="Z")` returns one result per value $z$,
the same numbers as the study fit. Each result estimates the average outcome if everyone followed the rule learned for
$Z = z$ with $Z$ set to $z$. That rule is the best treatment for a unit given $Z = z$, not the best
choice of treatment and intermediate together. An
[intermediate study](../technical-reference/method-evidence/learned-rule-intermediate.md) measured
this with 2,000 rows, ten folds and correct logistic models for the outcome, the treatment and the
intermediate: each 95% interval covered its target 0.946 to 0.949 of the time over 20,000 samples
at each value. Such a fit is marked supported when each fold is scored on its own held-out rows
(`CrossFitting(fold_evaluation=True)`) with one split, no weights or clusters, two options and the
rule from the outcome model, and experimental otherwise.

Every one of these fits needs, among other things, a treatment with distinct options rather than
a continuous dose and complete outcomes, and a cross-fitted fit needs at least two folds. A fit
that lacks one of them refuses before any model is fitted. The refusal table under
[Learned rules](../technical-reference/point-treatment-tmle.md#learned-rules) lists the rest.

A refit that relearned the rules would estimate the value of other rules. So the `subset`,
`random_common_cause`, `bootstrap_stability` and `bootstrap_measurement_error` tests of `refute`,
and `simulated_confounding`, keep the outer folds and the rule of each row, and read the value of
the fit's own rules. The `placebo` test refits as the fit did: under random treatment every rule
has the value of the mean outcome, which is what it reads against. The generated-outcome tests
refuse, because a learned rule has no generated truth yet.

The omitted-variable bound runs for this target. It gives a point bound under cross-fitting, and
in sample it also gives its one-sided limits. `benchmark()`, the E-value and the missingness tilt
read `unavailable`. The support report and `truncation_curve` run. With an intermediate held at
a value, the `placebo` test, `simulated_confounding` and the omitted-variable bound refuse, because
each would need its own check at that value.
[X11](../roadmap.md#x11-learned-policy-follow-ups) and
[F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) hold the refused
requests.

The [treatment-rules example](../examples/treatment-rules.ipynb) runs this fit end to end.
