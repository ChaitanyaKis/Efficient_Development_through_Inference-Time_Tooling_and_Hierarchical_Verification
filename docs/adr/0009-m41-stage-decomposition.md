# ADR 0009 — M4.1 Large-Stage Decomposition

- **Status:** Accepted
- **Date:** 2026-08-15
- **Milestone:** M4.1
- **Supersedes:** the monolithic generation path in ADR 0008 §2
- **Experiment:** `docs/experiments/0001-stage-decomposition.md`

---

## 1. The decision

Large structured generations are split into several small, independently-validated model
calls, and the artifact is assembled from them deterministically.

The UX agent runs four stages (`flows`, `steps` per flow, `screens`, `presentation`); the
Architect runs six (`components`, `data`, `api`, `decisions`, `threats`, `plan`). The
monolithic path remains reachable behind `--monolithic` so the comparison stays reproducible.

## 2. Why

M4 measured the monolithic UX call failing six consecutive times on the configured 3B model
with `<root>: Input should be an object`, while the much smaller Product Manager call
succeeded first time. Two explanations fitted: the task is harder, or the schema is bigger.

Experiment 0001 separated them. Same model, same brief, five independent trials per arm:

| arm | successful | model calls | mean runtime | largest schema |
|---|---|---|---|---|
| `ux_monolithic` | **0/5** | 1 | 132.1s | 4,800 B |
| `ux_decomposed` | **5/5** | 5 | 20.7s | 1,614 B |
| `architect_monolithic` | **0/5** | 1 | 99.1s | 6,968 B |
| `architect_decomposed` | **5/5** | 6 | 30.6s | 2,138 B |

The failure was structural capacity, not task difficulty. The same model produces a complete
valid artifact in 10/10 decomposed trials and 0/10 monolithic ones. Every monolithic failure
classified as `RETRY_EXHAUSTED`.

**Decomposition is also faster**, which is not the obvious result. It costs five or six calls
instead of one, but the monolithic path spends three full attempts failing before giving up.
More calls, less time — a property of failure being expensive rather than of small calls
being cheap.

**Total context roughly doubles while per-call context halves** (UX: 4,902 characters across
five prompts versus 2,303 in one; largest single prompt 1,068 versus 2,303). That is the
trade, and it is the right way round for a model whose limit is per-call capacity.

## 3. What was explicitly not done

M4.1 item 8 lists the ways a model failure must *not* be solved. None was used:

- No required field became optional.
- No enum became a free string.
- No schema stopped forbidding unknown fields — every stage output is still `extra="forbid"`.
- Retries were not increased; each stage keeps the same bounded budget.
- No malformed output is accepted, and nothing invalid is silently dropped into an artifact.

One M4 decision was **reverted** as a violation of this rule: `product_name` had been made
optional on the model-facing schemas to work around the model omitting it. It is now removed
from those schemas entirely. That is the stronger fix and the correct one — product identity
is system-owned (item 2), the caller already knows it, and asking the model to echo it back
adds a required field it can fail without adding any information. The *artifact* still
requires a name; the assembler supplies it from the PRD.

A test asserts that no schema in either arm asks the model for a name, id, version, status,
authority, or project.

## 4. Partial runs are a real state

A failed stage never destroys a validated one. `StageLedger` records each stage
independently, and the assembler builds from whatever survived:

- A failed `presentation` stage yields a specification with flows and screens and no
  components.
- A failed `steps` stage **omits that flow** rather than inventing one. A fabricated journey
  in a document a Frontend Agent will read as fact is worse than a missing one.
- A failed `components` stage skips everything downstream, because nothing else is meaningful
  without it.

Skipped is distinct from failed: a stage never given a chance did not fail, and collapsing
the two would make a report unreadable.

The artifact from a partial run **is stored** and **cannot be approved**. Missing stages are
recorded as blocking `STAGE_INCOMPLETE` validation issues, so the M4 approval gate refuses it
by the same mechanism that refuses any invalid artifact. Storing it is deliberate: an
operator can see what is missing and re-run one stage instead of regenerating everything.

## 5. Failure classification

Every stage failure is one of `MODEL_FAILURE`, `SCHEMA_VALIDATION_FAILURE`, `TIMEOUT`,
`RETRY_EXHAUSTED`, `CONTEXT_FAILURE`, `ENVIRONMENT_FAILURE`, or
`ARTIFACT_VALIDATION_FAILURE`, each mapping onto the platform-wide `FailureCategory`.

This is not bookkeeping. A model that will not produce a schema needs a smaller schema; an
unreachable runtime needs an operator; a timeout may just need re-running. A generic failure
hides which of those applies — and during this milestone the classifier immediately and
correctly reported Ollama being down as `ENVIRONMENT_FAILURE` while marking the downstream
stages `SKIPPED`, which is exactly the distinction it exists to make.

## 6. Context control

Each stage receives only what it needs: requirement *lines* rather than the whole PRD,
component *names* rather than the assembled architecture, and for a flow's steps, only the
requirements that flow serves.

Without this, context cost would scale with the number of stages and decomposition would
trade one large prompt for six medium ones. Measured, the largest single decomposed prompt is
roughly a third of the monolithic one.

## 7. Consequences

- The decomposed path is the default. `--monolithic` exists for reproducing experiment 0001.
- Model calls per specification rose from 1 to 5 (UX) and 1 to 6 (architecture). On a local
  zero-cost model this buys reliability with time that was being spent on failure anyway.
- **Open: sufficiency, not validity.** Requirement coverage is 3/3 for architecture but 2/3
  for UX. Decomposition produced valid, complete specifications that still miss a requirement
  about a third of the time. Validity was the M4.1 question; sufficiency is a different one
  and is not answered here.
- **Open:** one brief, one model. A larger brief produces longer stage prompts and might
  reintroduce the failure at a larger size; a larger model might not need decomposition at
  all. Neither is measured.
- **Open:** the runtime declines to compile these schemas into decoding grammars and falls
  back to `format=json` on every call, in both arms. On a runtime that enforced the grammar
  the monolithic arm might not fail this way — the result is about this stack, not about
  structured generation in general.
