# ADR 0006 — M3.1 Targeted Memory and Environment Reliability

- **Status:** Accepted
- **Date:** 2026-08-14
- **Milestone:** M3.1
- **Supersedes:** the memory *integration* decision in ADR 0005 (the memory *subsystem*
  stands unchanged)

---

## 1. The measurement that forced this milestone

M3 shipped a working memory subsystem and injected it into every coding prompt. Measured
against a no-memory control on the `multi_repair` benchmark with `qwen2.5-coder:3b`:

| arm | successes |
|---|---|
| baseline (no memory) | 5/6 |
| naive always-inject | 0/6 |

Reproduced across two independent batches. The mechanism is context pressure: roughly 1,800
characters of lessons competing with the code itself inside an 8,192-token window.

The conclusion drawn was **not** "memory is useless". It was that *where* memory is injected
is a design decision with measurable consequences, and it had never been measured. M3.1
turns that decision into a named, comparable policy.

---

## 2. Memory strategy is a policy, not a behaviour

`edith.memory.strategy` defines five strategies over three retrieval points
(`CODER_INITIAL`, `CODER_REPAIR`, `DEBUGGER`):

| strategy | retrieves at | rationale |
|---|---|---|
| `none` | nowhere | the control arm, kept runnable |
| `always` | all three | the M3 behaviour, kept reproducible |
| `failure_triggered` | repair + debugger | an error is a better key than a title |
| `debugger_only` | debugger | leaves every coder prompt untouched |
| `high_relevance` | all three, score ≥ 14.0, ≤ 2 memories | pay context only for a strong match |

Two properties are deliberate:

- **The table is total.** Every strategy has an explicit entry, and `policy_for()` returns
  *no retrieval* for an unrecognised value. Failing open would inject unexpectedly, which is
  the direction measured to be worse.
- **Retrieval never widens visibility.** A strategy decides *whether* to ask; the store still
  decides *what may be seen*, in SQL. Project isolation is not a property of any strategy,
  and `test_another_projects_memory_is_invisible_under_every_strategy` asserts that across
  all five.

---

## 3. Retrieval keyed on the failure, not the intent

`RetrievalRequest` gained `error_text`, `paths`, and `min_score`. `LexicalRanker` scores
error-term overlap at 2.5 points per term (capped at 15.0) and adds 5.0 when a memory
mentions the component being changed.

The weighting is the point. *"test_low_stock failed with AssertionError"* identifies the
problem far more precisely than *"fix the inventory module"*, so a lesson matching the
observed error outranks one matching the task title. Relevance remains a **gate**: metadata
(importance, confidence, recency) can only break ties between already-relevant memories, so
no amount of importance makes an unrelated memory eligible.

---

## 4. What the strategy comparison actually measured

`multi_repair`, `qwen2.5-coder:3b-instruct-q4_K_M`, 6 runs per arm (two independent batches
of 3, reported separately as well as pooled — a pooled number that hides a disagreement
between batches is not a result):

| strategy | pooled | per batch | mean memory chars | mean repairs |
|---|---|---|---|---|
| `none` | **4/6 (67%)** | 2/3, 2/3 | 0 | 5.9 |
| `always` | 0/6 (0%) | 0/3, 0/3 | 14,202 | 6.0 |
| `failure_triggered` | 2/6 (33%) | 0/3, 2/3 | 12,379 | 5.6 |
| `debugger_only` | 3/6 (50%) | 1/3, 2/3 | 6,099 | 5.5 |
| `high_relevance` | 2/6 (33%) | 1/3, 1/3 | 6,688 | 5.7 |

**No arm beat the control.** Two things are worth separating by how much support they have:

- **Well supported.** `always` is 0/6 here and 0/6 in M3 — 0/12 across four independent
  batches, against a control that wins roughly two thirds of the time. Naive always-inject
  is genuinely harmful, not noise.
- **Suggestive only.** The ordering among the remaining arms is monotone in injected
  context (0 chars → 67%, ~6k → 33–50%, ~12–14k → 0–33%), but 6 runs cannot separate 4/6
  from 3/6, and `failure_triggered` swung from 0/3 to 2/3 between batches. No claim is made
  that `debugger_only` beats `high_relevance`.

`false_positives` is 0 in every arm and every batch, so the M2.1 integrity gate held
throughout: no arm reached PASS by weakening a test.

**Decision: the shipped default is `none`.** Not because memory is worthless, but because no
strategy has earned its context cost on this model, and shipping an unmeasured default is
what M3 already did once. The subsystem stays fully built and one config line from active;
`edith strategies` re-runs this comparison on any model or benchmark.

The most useful finding is one the strategy design did not anticipate. The per-prompt budget
(`max_chars = 2000`) is respected, but the *cumulative* injection scales with repair count —
a run with six repairs under `failure_triggered` still delivers ~12,800 characters in total.
A strategy that constrains only **where** memory is injected does not bound total context
pressure in a repair-heavy loop. Bounding cost per *execution*, not per prompt, is the open
problem M3.1 identifies and does not solve.

`false_positives` is 0 in every arm, so the M2.1 integrity gate held throughout: no arm
reached PASS by weakening a test.

---

## 5. A non-zero exit means four different things

`edith.environment.classify` replaces "non-zero exit ⇒ `TEST_FAILURE`" with a deterministic
four-way diagnosis:

| category | meaning | code ran? | policy |
|---|---|---|---|
| `ENVIRONMENT_FAILURE` | toolchain absent or never started | no | escalate |
| `DEPENDENCY_FAILURE` | a required package is not installed | no | escalate |
| `CODE_FAILURE` | the project's own code failed to load | partly | repair |
| `TEST_FAILURE` | it ran and an assertion did not hold | yes | repair |

Precedence is ordered deliberately: a missing runner masks everything downstream, a missing
dependency masks the code, and only once both are excluded does a failing assertion mean
what it appears to mean. `code_executed=False` records the honest fact that a run says
*nothing* about correctness — sending the Debugger after code that never imported spends the
entire repair budget on a guess.

The classifier is pattern matching on real output. No model is consulted, because a model
asked *"why did this fail?"* will always have an opinion.

Two defects were found by writing these tests, both of which had been shipping silently:

- The classifier required the quoted traceback form (`ModuleNotFoundError: No module named
  'pytest'`) and missed the bare form an interpreter prints (`python.exe: No module named
  pytest`) — i.e. it missed *precisely* the missing-runner case it exists to catch.
- `parse_requirements` skipped any line beginning with `http`, silently dropping `httpx` and
  `httpcore` from every generated manifest.

---

## 6. Dependencies are discovered, never guessed

`EnvironmentSpec` is provider-neutral; `Ecosystem` is an open enumeration and Python is
simply the first implementation. Discovery is deterministic: manifests are parsed, imports
are read from the **AST** (a commented-out or string-literal import is not a dependency, and
a regex cannot tell the difference), and the interpreter is asked what it actually has.

`DependencyOrigin` is ordered by trustworthiness, and `MODEL_SUGGESTION` exists so that a
model's proposal is *representable but marked* — it can never be installed unreviewed.

`EnvironmentReport` is the structured artifact a future UI or Dependency Agent consumes:
runtime, dependencies with status and origin, what is undeclared, what is missing, and the
notes explaining why. It is data, not prose, so no consumer has to parse a summary string.
No UI is built in this milestone.

Interpreter detection prefers a project-local `.venv`, then the interpreter running Edith.
`python` from PATH is never consulted: on Windows it is frequently a Store alias that exits
without doing anything, which is the exact confusing failure the subsystem exists to prevent.

---

## 7. Installation is an execution boundary

Install artifacts are *generated text*. `edith.environment.provision` is the only code that
puts them on disk, and it does so through the M1 tool gateway via `filesystem.write` — so the
path policy, the agent's write scope, and the audit log apply to an installer exactly as they
apply to source code. A denied write is returned as a denial; nothing falls back to `pathlib`.

`assert_safe()` refuses, at generation time, any dependency carrying an alternate package
source (`--index-url`, `--extra-index-url`, `--find-links`, `git+`, a URL) or a shell
metacharacter. Generation fails before anything reaches disk, so a rejected manifest never
leaves a partial set of scripts behind.

Every generated script creates a **project-local** environment. No script contains `--user`
or `sudo`, and each verifies that the packages it installed actually import, because "pip
reported success" and "the application runs" are different claims.

Nothing in this milestone installs anything. Running a generated script is a separate,
explicitly-approved `shell.run` call.

---

## 8. Consequences

- The shipped default memory strategy is chosen by measurement and is revisable by
  re-running `edith strategies`. It is not an assumption.
- The Debugger is no longer dispatched against failures where the code never executed, which
  removes a whole class of wasted repair attempts.
- `edith environment` reports what a project requires and what is missing, without a model in
  the loop.
- **Open:** bounding memory's total context cost per execution rather than per prompt. The
  measurement in §4 says this is where the remaining harm lives.
- **Open:** whether memory helps at all on a larger local model. Every result here is
  specific to a 3B model in an 8k window, and must not be generalised beyond it.
