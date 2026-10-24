# Part VII: Research Extensions

You can already build, verify and serve an Agent using existing models. Training and self-improvement appear here because they need the task definitions, failure traces, evaluation sets, permission boundaries and rollback mechanisms developed earlier. A first application can skip this part.

## Diagnose the Failure First

Fix missing sources with retrieval, unit errors with tools, lost state with recovery and unauthorized execution with host checks. Consider training or automatic optimization only after these engineering causes are excluded and fixed tasks still expose a repeatable model-behavior deficiency.

| Chapter | Research question | Required inputs | Evidence of improvement |
|---|---|---|---|
| [24 Agentic-RL](./chapter_agentic_rl/README.md) | Can trajectories and verifiable feedback improve the policy? | Executable environment, train/validation/held-out sets, reviewable rewards and compute budget | Compare with a frozen baseline on the same independent held-out set and inference budget; test reward gaming and regressions |
| [25 Self-Improvement](./chapter_self_evolving/README.md) | Can prompt, skill or system changes produce repeatable gains? | Versioned candidates, frozen gates, activation/deactivation and rollback records | Validate candidates offline before activation; revert uncertain gains or regressions |

Prompt or skill updates need not train model parameters. You may read chapter 25 directly after the evaluation and security foundations in chapters 18–19. The order organizes research topics rather than imposing training on every reader.

Each example states compute requirements and verification limits. Local candidate search or fixed scores do not establish real training gains. The deliverable is an auditable experiment record.
