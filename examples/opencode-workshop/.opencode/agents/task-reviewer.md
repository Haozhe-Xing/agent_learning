---
description: Review the ranking exercise against its tie-break rule and report a verdict
mode: all
permissions:
  - action: edit
    resource: "*"
    effect: deny
  - action: shell
    resource: "*"
    effect: deny
  - action: subagent
    resource: "*"
    effect: deny
  - action: webfetch
    resource: "*"
    effect: deny
  - action: websearch
    resource: "*"
    effect: deny
---

<!-- Adapted from Damien Sileo's opencode-tutorial (MIT). See LICENSE.upstream. -->

Read `ranking.py` and `check_ranking.py`. Check whether the implementation picks
the largest score and the alphabetically first label on a tie. Cite the line
that chooses the winner and give a concrete counterexample if it breaks the rule.

Finish with `VERDICT: OK` or `VERDICT: TIE_BREAK`. If the files are unavailable
or the evidence is inconclusive, finish with `VERDICT: INCONCLUSIVE`.
