# Amounts, discounts and policies

These tutorials go beyond treated against untreated: policies that change who gets the treatment,
more or less of a treatment, and a continuous dose such as a discount.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| policies to compare, such as offering a program only to high-risk patients, raising each patient's chance of an offer, or raising how much of the program each patient gets | [three offer policies](../interventions.ipynb) | each policy has its own starting point: offering only to high-risk patients is compared with offering no one, and the other two with current practice, so the answers do not rank the policies or compare one with another directly. A staffing limit on how much each patient can get is part of the question: the same raise with and without the limit gives two different answers |
| a team chooses among several contact schedules, or sets two service times, and you want the effect of nudging those choices or raising both amounts | [more contacts and more service time](../schedules-and-intensity.ipynb) | when two services help each other, the effects of raising each one alone do not add up to the effect of raising both |
| several treatment levels, summarized by one trend | [a trend across three contact schedules](../msm-projections.ipynb) | a straight line through the levels still reports a slope when it does not fit them, so the slope alone can mislead |
| a continuous dose, such as a discount from 0% to 40%, and how the outcome changes across its range | [renewal discounts](../dose-response.ipynb) | subscribers who got bigger discounts renewed less, because the engine gave them to less active subscribers. The curve has no interval at one exact discount, so the page reads it averaged over nearby discounts. Where the engine almost never gives some subscribers a discount, the intervals run short even when the overlap check looks clean |

## See also

- [Monthly prices: longitudinal TMLE for a continuous treatment](../sequential-pricing.ipynb) raises a continuous price that is set again each month.

```{toctree}
:hidden:
:maxdepth: 1

Three offer policies </examples/interventions>
More contacts and more service time </examples/schedules-and-intensity>
A trend across three contact schedules </examples/msm-projections>
Renewal discounts </examples/dose-response>
```
