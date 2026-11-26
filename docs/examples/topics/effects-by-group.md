# Effects by characteristic or group

These tutorials break the effect down by a recorded characteristic, such as age, or by groups of
people, such as those who live alone.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| an effect that changes with a customer's or patient's characteristics, such as age, and you want to see how before deciding whom to target | [who gains most from a coupon](../conditional-effects.ipynb) | within each age group, a simple comparison of customers with and without a coupon mixes the coupon's effect with who received it, and here it overstates the gain for older customers who gain almost nothing. The curve has no confidence interval, so the page checks it on held-out rows before acting on it, and fits a simpler shape with intervals |
| an effect that may differ between groups of patients, such as those who live alone, and you need an interval for each group and for their difference | [does navigation help patients who live alone more?](../subgroup-effects.ipynb) | within each group, a simple comparison of offered and other patients mixes the offer's effect with who received it: teams offered navigation more to high-risk patients, so the comparison understates the gain in every group |

## See also

- [Who should get the offer: learned rules and random offers with TMLE](../treatment-rules.ipynb) turns differences in the effect into a rule for whom to treat.

```{toctree}
:hidden:
:maxdepth: 1

Who gains most from a coupon </examples/conditional-effects>
Does navigation help patients who live alone more? </examples/subgroup-effects>
```
