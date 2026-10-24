# Part V: Validation and Deployment

Earlier iterations should already retain inputs, expectations, actual outputs and failures. This part turns them into repeatable gates, establishes execution boundaries and defines service delivery. Evaluation and safety start with chapter 3 and are studied systematically here; frameworks are not prerequisites for using them.

| Chapter | Previous limitation | Artifact | Condition for continuing |
|---|---|---|---|
| [18 Evaluation](./chapter_20_evaluation/README.md) | One successful demonstration does not establish improvement | Fixed regression set, per-case verdicts, configuration and costs | Compare identical inputs to the baseline; failed gates block release |
| [19 Security](./chapter_21_security/README.md) | Normal cases pass while hostile data or unauthorized actions remain possible | Trust boundary, permission matrix and attack/rejection records | Documents cannot grant permission; host code rejects before execution |
| [20 Deployment](./chapter_22_deployment/README.md) | A working local function is not an operable service | API contract, configuration, failure responses and recovery checklist | Interface verdicts agree with local results; timeouts and faults have explicit outcomes |

Repository `reference-agent/` supplies a runnable interface, evaluation and basic-guard teaching baseline. Live-model quality, production authentication, durable task recovery, containers and load checks require additional evidence for the target environment. A running server is not business acceptance.

Choose one [capstone](./part6.md) next, or stop when your current task is delivered. Enter [research extensions](./part7.md) only when fixed evaluation identifies a model-behavior limitation worth investigating.
