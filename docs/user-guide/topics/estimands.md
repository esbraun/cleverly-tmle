# Estimands

An estimand is the causal quantity you ask for, written as an object such as `ATE()`, and choosing
it is the second half of identification after the design. These pages cover a treatment decided
once. Pick the page by the kind of question; a treatment decided at several times has its own plans
under [treatments over time](treatments-over-time.md).

| your question | page |
| --- | --- |
| the average treatment effect (ATE), the effect among the treated (ATT), a risk or odds ratio as the quantity you ask for, the mean under a rule you fix in advance or a random assignment, a trend line across set plans, or the effect with a later variable held fixed (a controlled direct effect) | [average effects and fixed plans](../estimands.md) |
| the value of a rule for whom to treat that the fit learns from the data, rather than one you fix in advance | [learned treatment rules](../learned-rules.md) |
| what happens if each person's dose (a price, a discount) is shifted or scaled, if each person's odds of treatment are multiplied, which raises or lowers their chance, or the mean outcome at every dose | [dose shifts and odds changes](../policies.md) |
| how the effect changes with a few baseline columns, such as age, or the effect within groups | [conditional effects](../conditional-effects.md) |
| which of several candidate treatments or practices (exposures) has an effect you can detect | [rank candidate exposures](../variable-importance.md) |

```{toctree}
:hidden:
:maxdepth: 1

Average effects and fixed plans </user-guide/estimands>
Learned treatment rules </user-guide/learned-rules>
Dose shifts and odds changes </user-guide/policies>
Conditional effects </user-guide/conditional-effects>
Rank candidate exposures </user-guide/variable-importance>
```
