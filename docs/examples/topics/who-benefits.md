# Who benefits and whom to treat

These tutorials ask how the effect differs between people, and turn that into a rule for whom to
treat.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| an effect that changes with a customer's or patient's characteristics, such as age, and you want to see how before deciding whom to target | [who gains most from a coupon](../conditional-effects.ipynb) | within each age group, a simple comparison of customers with and without a coupon mixes the coupon's effect with who received it, and here it overstates the gain for older customers who gain almost nothing. The curve has no confidence interval, so the page checks it on held-out rows before acting on it, and fits a simpler shape with intervals |
| an effect that may differ between groups of patients, such as those who live alone, and you need an interval for each group and for their difference | [does navigation help patients who live alone more?](../subgroup-effects.ipynb) | within each group, a simple comparison of offered and other patients mixes the offer's effect with who received it: teams offered navigation more to high-risk patients, so the comparison understates the gain in every group |
| a treatment that helps some patients and harms others, so you must decide who gets it, or share a limited number of places at random | [who should get the offer](../treatment-rules.ipynb) | a rule scored on the same patients it was learned from looks better than it will do on new patients |

## See also

- [Two decisions after discharge: a learned treatment rule with longitudinal TMLE](../two-decision-rules.ipynb) learns a rule for each of two decisions in a row, such as an offer at discharge and another at day 14.

```{toctree}
:hidden:
:maxdepth: 1

/examples/conditional-effects
/examples/subgroup-effects
/examples/treatment-rules
```
