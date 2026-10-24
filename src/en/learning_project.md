# The Continuing Project and Its Acceptance Checks

Run a complete small application before taking its parts apart. The independent project in `reference_code/evidence_assistant/` starts with a **local-document research assistant**: read several fictional project documents, answer multiple questions, and produce a report that exposes retrieved evidence, plan state, context-budget records, and citations.

This is a next exercise for beginners who can run Python and read functions and dictionaries. Its difficulty comes from real data handoffs and failure handling. Standard-library code, offline documents, and verbatim excerpts let you start without a model, database, or framework setup. Open-web research, fact cross-checking, long-term user memory, a web service, and adapters for every framework remain extensions.

## Run It, Then Replace Documents and Questions

From the repository root with Python 3.11+:

```bash
python -m reference_code.evidence_assistant
```

The default reads fictional documents from `sample_data/` and automatically loads `sample_task_specs.json`: three questions with 15 frozen points, covering release acceptance (4), tool budgets/timeouts (6), and rollback (5). Documents feed retrieval and answering; task specifications are used only for independent acceptance and never become evidence in the answer context. Each finding separates source support from coverage of these requirements.

Replace the default with two practice documents, two questions, and separately specified requirements. This example uses a fresh temporary directory and keeps the task JSON outside the evidence directory:

```bash
research_demo_dir=$(mktemp -d "${TMPDIR:-/tmp}/evidence-study-en.XXXXXX")
mkdir "$research_demo_dir/docs"
cat > "$research_demo_dir/docs/release.md" <<'EOF'
Release acceptance requires passing regression tests and a project owner's approval of the rollback rehearsal record.
EOF
cat > "$research_demo_dir/docs/rollback.txt" <<'EOF'
After deployment failure, stop new traffic, roll back to the previous stable version, and retain the failure logs.
EOF
cat > "$research_demo_dir/tasks.json" <<'EOF'
{
  "schema_version": 1,
  "tasks": [
    {
      "question": "What does release acceptance require?",
      "requirements": [
        {"id": "regression", "description": "Pass regression tests", "all_of": ["passing regression tests"]},
        {"id": "approval", "description": "Owner approves the rollback rehearsal record", "all_of": ["a project owner's approval of the rollback rehearsal record"]}
      ]
    },
    {
      "question": "What should happen after deployment failure?",
      "requirements": [
        {"id": "traffic", "description": "Stop new traffic", "all_of": ["stop new traffic"]},
        {"id": "restore", "description": "Restore the previous stable version", "all_of": ["roll back to the previous stable version"]},
        {"id": "logs", "description": "Retain failure logs", "all_of": ["retain the failure logs"]}
      ]
    }
  ]
}
EOF
python -m reference_code.evidence_assistant \
  --docs "$research_demo_dir/docs" \
  --task-spec "$research_demo_dir/tasks.json" \
  --question 'What does release acceptance require?' \
  --question 'What should happen after deployment failure?' --format json
```

This should exit 0: both findings are `answered`, source support and coverage are `passed`, and the report has `accepted=true`. JSON questions match the command exactly. Matching normalizes whitespace and case, not paraphrases or punctuation. Every fragment in a requirement's `all_of` must appear in the answer. A task maintainer should independently define requirements before generation, rather than derive passing expectations from the generated answer.

Remove `--task-spec` and rerun the custom documents: even supported excerpts remain `evidence_found`, with `coverage.status=unknown`, `accepted=false`, and exit code 1. Only the bundled document directory automatically selects the frozen default specification. Python callers may instead explicitly supply an independent `verifier`.

The loader accepts 1–32 top-level UTF-8 `.md` / `.txt` regular files and 1–4 repeated `--question` arguments. Non-empty paragraphs become sources such as `release.md:p1`. Retrieval matches remaining Chinese character bigrams after filtering, or English words. Change “previous stable version” in the document: the excerpt must change, while the frozen requirement should report the missing literal point. Update a specification only after independently approving a changed business requirement.

`retrieval.hits` records each passage's `score` and `matched_terms`. The score counts distinct matched terms; it is not confidence or answer accuracy. Generic Chinese question phrases are filtered, and multi-bigram Chinese queries normally need at least two character-bigram matches; this is not Chinese word segmentation. A topical match may still lack an answer: a birthday question can retrieve a passage about the project owner and leave coverage unknown. Only no eligible hits produces `no_evidence`. This baseline does not perfectly recognize every unanswerable question.

`--max-steps` bounds plan steps, `--max-calls` bounds tool calls, and `--context-bytes` bounds actual messages measured as UTF-8 bytes. Their defaults are 8, 8, and 8192. Run the default task with `--max-steps 1` and require an incomplete result. Also try a question with no matches and a context budget too small for required evidence. `--output FILE` saves the report and refuses to overwrite an existing file.

Results separate `status` from `accepted`. Inspect `verification` for source support and `coverage` for independent task coverage. Its `matched_points`, `missing_points`, and `details` locate omissions. Supported excerpts missing configured points yield `partial`; unconfigured coverage yields `evidence_found` / `unknown`. Retrieval, generation, and source-check failures retain distinct failure states. Each finding's `stage` and `reason.code/message` identify the location.

Top-level `summary` counts accepted and unfinished questions, `budgets` records actual usage, and `failure` locates the first unfinished question (`question_index` is zero-based), step, and reason. Step/call limits can leave findings `not_started` or `retrieved`; context failures expose `required_bytes` and `limit_bytes`. Read these with `execution` and `tool_trace`: a completed execution plan alone is insufficient. Every question must pass both checks for `completed`, `accepted=true`, and exit code 0.

Offline answers are controlled excerpts, and frozen requirements use literal inclusion. They do not establish source truth, correct paraphrases, understanding of negation or contradictions, or general semantic completeness. Read `verification_scope` and use representative tasks and suitable independent acceptance for additional business requirements.

## Read the Main Path in Four Modules

Read these project files in order; the entire directory is not a prerequisite.

| File | Question to trace first | Small change to try |
|---|---|---|
| `research.py` | How do documents and questions become retrieval/answer steps and a report? | Replace inputs or report format while retaining failure reasons |
| `rag.py` | How do retrieved sources differ from sources actually cited in an accepted answer? | Observe no-match, missing-citation, and unknown-citation results |
| `planning.py` | How do step results reach later steps, and how do failure or exhaustion stop execution? | Lower the step budget and inspect the incomplete result and trace |
| `context.py` | Which messages does the answer consume, and what if required evidence cannot fit? | Lower the byte budget and require an explicit failure |

Next read `task_acceptance.py` and `sample_task_specs.json` to trace independent requirements, then `harness.py` for execution constraints. Compare the counterexamples in `tests/test_research.py` and `tests/test_retrieval_and_tasks.py`.

```text
reference_code/evidence_assistant/
├── research.py / rag.py / planning.py / context.py
├── task_acceptance.py       # Independent task coverage
├── sample_task_specs.json   # 3 default questions, 15 frozen points; not evidence
├── sample_data/             # Documents actually retrieved and quoted
├── harness.py / tests/
└── README.md / WALKTHROUGH.md / CONTRACTS.md
```

Persistent preferences in `memory.py`, the six-stage calculation in `demo.py`, and numeric contracts in `acceptance.py` / `contracts.py` are foundation exercises or later reading, not prerequisites for running the research application.

## Foundation Exercise: Six Iterations of One Calculation

Early chapters retain a controlled Moon-distance problem to expose one mechanism at a time. It is a different task from document research: one fixes numeric expectations; the other combines several documents, questions, and a report. Keep their business acceptance rules separate.

```bash
python -m reference_code.evidence_assistant.demo --all
```

Results identify `offline-fixture` mode. Each stage reads fictional `384400 km` evidence, extracts a quantity, normalizes units, computes light-travel time, and cites its source. The default is approximately 1.28222 seconds. This checks contracts and orchestration, not live-model quality.

| Stage | Chapter | Added artifact and failure check |
|---|---|---|
| tools | 3 | Original quantity, unit, and calculation; changed evidence must change the result |
| memory | 4 | Stored and re-read unit preference; check reopening, isolation, and deletion |
| planning | 5 | Dependencies, results, and execution state; failures and exhaustion cannot complete |
| rag | 6 | Retrieved evidence and actual citations; report missing evidence explicitly |
| context | 7 | Actual messages, byte budget, and omissions; required evidence cannot disappear silently |
| harness | 8 | Tool trace and independent verdict; denied calls and wrong artifacts fail |

Use `--stage memory` for one iteration. Check that input really reaches calculation:

```bash
printf 'Moon distance is 100 km.\n' > /tmp/light-distance.txt
python -m reference_code.evidence_assistant.demo --all --file /tmp/light-distance.txt --output-unit milliseconds
```

Every stage should produce approximately `0.33356409519815205` milliseconds; replacing `100 km` with `100000 m` preserves the result. This adapter accepts controlled decimal distance statements, at most 24 digits, normalized to `[1e-12, 1e18]` metres. The project's `CONTRACTS.md` defines the grammar.

An independent verifier re-parses original evidence without calling the producer's parser or calculator, then checks quantities, units, arithmetic, final answers, sources, and original-text hashes. Fixed expectations live in `tests/fixtures/light_travel_cases.json`, which the execution path never reads. Hashes establish consistency, not authenticity; default QA citation checks also do not establish semantic support.

## Two Critical Transitions

**Memory to planning:** in the foundation calculation, remembering “use milliseconds” cannot prevent skipped extraction, repeated calculation, or premature completion. Planning adds dependencies, result state, and stopping rules. Set an insufficient step budget and require an incomplete result. The research application reuses this principle by scheduling retrieval and answering for each question; it does not thereby acquire cross-session memory.

**Context to Harness:** in that same calculation, fitting required evidence into messages does not authorize extra tools or establish that the numeric result is correct. Host checks enforce permissions and call budgets and accept the output independently. The research application also needs execution constraints, while using its own original-evidence acceptance contract rather than light-travel arithmetic.

## Handoffs Across the Whole Book

| Stage | Handoff | Existing support and reader work |
|---|---|---|
| [1–2 Foundations](./part1.md) | Turn a requirement into a task card, baseline, and failure record | Setup entry points are supplied; readers retain actual call records |
| [3–8 Core iterations](./part2.md) | Observe mechanisms in the foundation problem, then trace retrieval, planning, context, and reporting in the research application | Both offline paths and their failure tests are supplied |
| [9 Skill reuse](./chapter_skill/README.md) | Package a verified workflow without expanding permissions | Design and exercises; no complete skill service |
| [10–11 Implementation choice](./part3.md) | Preserve task, permissions, and acceptance before choosing a migration | Existing framework examples use other tasks; readers adapt the research task for a controlled comparison |
| [12–15 Products and architecture](./part3.md) | Record an adopt/skip decision and a mechanism transfer | Optional cases; no uniform adapter for four products |
| [16–17 Collaboration and integration](./part4.md) | Extend a single-Agent baseline only for needed delegation or system boundaries | Runnable developer team and minimal stdio protocol; same-task gains need separate checks |
| [18–20 Validation and deployment](./part5.md) | Turn traces, failures, and acceptance into gates and runtime boundaries | A separate `reference-agent/` supplies evaluation and HTTP/SSE foundations; this app's service, authentication, load, and recovery remain exercises |
| [21–23 Domain delivery](./part6.md) | Deliver one domain project with independent acceptance | Design and assembly branches require actual domain components |
| [24–25 Research](./part7.md) | Start training or candidate optimization from measured limitations and a frozen baseline | Optional after application delivery, not prerequisites |

A first application can follow **1–8 → 10 or 11 if needed → 18–20 → one of 21–23**. Select chapter 9 and chapters 12–17 for reuse, products, collaboration, or integration needs. Use the [problem-driven decision table](./part3.md); stop migrating when the current implementation is clear and sufficient. The framework and domain chapters are not all connected to this one application.

## Keep Verification Evidence for Each Change

Run the independent project's tests from the repository root:

```bash
python reference_code/evidence_assistant/run_tests.py
```

Retain input, expectation, actual result, configuration, and verdict rationale. The application and foundation problem maintain their own success and failure cases; rerun both after changing shared modules. The project folder can also be copied separately and run with `python run.py` or tested with `python run_tests.py`; its `README.md` describes the entry points.

Vector memory and actual framework tool loops and state graphs are additional `examples/` integrations:

```bash
pip install -r examples/requirements.txt
python -m pytest examples/integration_tests -q
```

Their scope differs from the offline project. Actual ChromaDB, LangChain, and LangGraph integration does not automatically establish the research app's model factuality, retrieval recall, or cost; use separate representative task sets. Continue in chapters [18](./chapter_20_evaluation/README.md), [19](./chapter_21_security/README.md), and [20](./chapter_22_deployment/README.md) to develop regression gates, security checks, and deployment procedures.
