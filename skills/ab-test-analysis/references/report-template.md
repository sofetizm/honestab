# Report template

Write the report in the user's language. Keep the decision first.

```markdown
## Decision: SHIP / DO NOT SHIP / NO SIGNIFICANT EFFECT / RESULTS NOT TRUSTWORTHY

**Primary metric — <name>:** <control> → <treatment>, <+x.x%> (95% CI <a%> … <b%>), p = <p>.
<One sentence: what this means for the business.>

**Guardrails:** <each: no significant change / significantly worse by x%>.

**Secondary metrics** (Holm-corrected): <each with effect and CI; mark significant ones>.

**Checks**
- Sample ratio: <n_control> vs <n_treatment>, planned <split>, p = <p> — OK / FAILED.
- Data: <rows per unit, units in both groups, missing values, outlier rule>.
- Peeking: <analysed once at the planned end / stopped early → p-values not valid>.

**Sensitivity:** with this sample the experiment detects effects from ~<MDE>% (power 80%).

**Assumptions and limitations:** <anything assumed, segments as hypotheses only, ...>
```
