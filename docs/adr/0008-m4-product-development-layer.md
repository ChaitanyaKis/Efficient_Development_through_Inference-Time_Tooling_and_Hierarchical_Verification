# ADR 0008 — M4 Product Development Layer

- **Status:** Accepted
- **Date:** 2026-08-15
- **Milestone:** M4
- **Extends:** ADR 0004 (instruction hierarchy), ADR 0007 (memory governor)

---

## 1. Artifacts, not conversation

CLAUDE.md's architectural rule is that agents communicate through structured artifacts and
project state rather than by talking to each other. M4 is where that becomes the primary
mechanism: a Product Manager writes a PRD, a UX agent reads it and writes a specification, an
Architect reads both and writes a system design and an implementation plan.

No agent reads another agent's transcript. Each stage's input is a *document*, and the
document is the contract. That makes the pipeline restartable, inspectable, and reviewable at
every step — none of which is true of a chat log.

Every artifact carries the same envelope: id, kind, version, project, author, status,
authority, dependencies (with the versions actually read), source references, validation
state, and supersession links. The body is kind-specific and strictly typed, and the envelope
refuses a body that does not validate against its kind's schema.

---

## 2. Identity is assigned by the system, never by the model

Requirement ids are the most load-bearing strings in the platform. Every downstream artifact,
task, and test references them, and M7's impact graph will be built entirely out of them.

So the model never produces one. It emits a flat, unnumbered list; `draft_to_prd` assigns
`REQ-001`, `REQ-002`, and pairs each with an `AC`. Same for flows, screens, components,
decisions, and tasks.

This is not stylistic. A 3B model asked to emit unique, correctly-formatted, densely-numbered
ids across a nested document will eventually emit `REQ-1`, a duplicate, or a gap — and every
one of those failures is silent until something downstream tries to resolve the reference.
Numbering in code makes the entire failure class unreachable.

The same reasoning drives the flat model-facing schemas. `ProductManagerOutput` is much
smaller than `PRDDocument`, because a small model producing deeply nested objects with enums
and frozensets fails constantly. The translation functions are the trust boundary, exactly as
`plan_to_tasks` is for the M2 planner.

**References that resolve to nothing are dropped, not propagated.** A UX spec claiming to
satisfy `REQ-042` when the PRD defines six requirements is hallucinating; the translation
drops it, and the requirement it was meant to cover then correctly reports as uncovered. The
alternative — passing it through — produces a dangling id that survives all the way to
implementation.

---

## 3. The authority hierarchy is executable

M4 extends the M2.1 hierarchy from five ranks to seven and makes it code rather than prose.
`edith.authority` defines the order, and `may_override(candidate, incumbent)` answers the
question directly. See `docs/INSTRUCTION_HIERARCHY.md` for the full table.

The load-bearing addition is that **approval confers authority**. The same ADR text is an
agent recommendation while it is a draft and an approved architecture decision after a human
accepts it. `Artifact` enforces this in a validator: claiming
`APPROVED_ARCHITECTURE_DECISION` authority on a `DRAFT` will not construct. Without it, an
agent mints authority by writing confidently.

Two rules are deliberately not what an ordered list would naively give you:

- Advisory levels (agent recommendation, repository content, external content) never
  override anything, including things below them.
- Equal levels do not override either. Two conflicting requirements is a contradiction to
  report, not a race.

---

## 4. Contradiction detection is a set intersection, not a judgement

M4.8's examples — "must work offline" versus "requires cloud-only service" — cannot be found
in prose. Three wordings of the same claim are unrelatable by rule, and asking an LLM
reintroduces exactly the judgement the milestone says not to depend on.

So the claim is made structural. `ProductProperty` is a closed vocabulary of 22 values;
requirements, UX specs, and architectures all declare from it; a contradiction is an
intersection against `CONTRADICTORY_PAIRS`. Implications are closed transitively first, so an
architecture declaring `CLOUD_DEPENDENT` is understood to require a network without anyone
remembering to tag both.

Three layers, in descending confidence:

| Layer | Basis | Blocking |
|---|---|---|
| Declared properties | closed vocabulary, exact | yes |
| Structural fields | endpoints, components, entities | yes |
| Prose hints | keyword match over free text | **never** |

The third layer exists because a document that discusses offline behaviour but declares no
property is under-specified, and that is worth surfacing. It never blocks, because a sentence
saying a product must *not* work offline contains the word "offline" just as clearly as one
saying it must — and a blocking finding must be one a human agrees with immediately.

The structural layer catches what a property tag alone would miss: an architecture can
declare `AUTHENTICATION_REQUIRED` and still expose every endpoint anonymously. The endpoints
are the ground truth.

---

## 5. Validation gates approval

`Artifact` refuses to be `APPROVED` while its validation state is anything but `VALID`, and
`UNVALIDATED` is the honest default — an unchecked artifact must not be indistinguishable
from one that passed.

Checks are deterministic and split by consequence:

- **Blocking**: a dangling element reference, a flow step that strands the user, a plan
  dependency cycle (Kahn's algorithm, same as the M2 task DAG), a task naming a component the
  architecture does not define.
- **Advisory**: a requirement with no acceptance criterion, a screen missing its loading or
  error state, a threat with no mitigation, a technology chosen with no rejected alternative.

The split matters. A backend requirement legitimately has no UX flow, and a constraint may
legitimately be verified by inspection rather than by a criterion. Blocking those would make
valid documents unapprovable; hiding them would make real gaps invisible.

---

## 6. Nothing approved is ever destroyed

A revision is a new row at `version + 1`, never an edit. The predecessor moves to
`SUPERSEDED` only when the successor is itself *approved* — a draft revision does not retire
an approved document, which is what stops half-finished work becoming the project's truth.

Project isolation is a SQL predicate, exactly as the memory store does it. Nothing above the
store layer is trusted to remember to filter.

---

## 7. Product agents cannot build what they design

| Agent | Tools | Write scope |
|---|---|---|
| `product_manager` | `filesystem.read`, `filesystem.search` | none — read-only |
| `ux_designer` | + `filesystem.write` | `design/**`, `docs/ux/**` |
| `architect` | + `filesystem.write` | `architecture/**`, `docs/adr/**` |

None has `shell.run`, `git.*`, or network access. A PM that could run commands is a PM that
could ship; an Architect that could edit source would make the design a formality. The
separation is enforced by the M1 gateway, not by prompt instruction, and a test asserts it
over the registry rather than per agent.

The implementation plan is a *document*, not an executable DAG. M4 produces it and stops.
Converting it into tasks the loop will run is a separate trust boundary, and an agent should
not be on both sides of one.

---

## 8. Memory stays off

M3.2 measured memory as not improving the 3B coding benchmark, and M4 does not relitigate
that. No product agent injects memory automatically; each accepts an optional
`prior_knowledge` field that only carries content when a caller explicitly supplies it, and
the M3.2 governor remains the only autonomous injection path. A test asserts that no
`PRIOR KNOWLEDGE` block appears in any product prompt during a default pipeline run.

Context cost is instrumented rather than budgeted. M4.11 warns against guessing sizes for
product artifacts, so `StageMetrics` records input characters, output characters, artifact
characters, model calls, attempts, duration, and elements produced, per stage. Those are the
measurements that would justify a budget; setting one now would be the same mistake M3 made
with memory.

---

## 9. Review is computed before it is opined

Every review finding is produced by a deterministic function over artifact structure. A model
critique, when one is requested, is stored in a separate `model_critique` field and labelled
as opinion.

The reason is legibility. A report mixing computed facts with model opinions produces
something where the reader cannot tell which is which — and the opinions are the ones that
get trusted. Verdicts are computed too: a review with a blocker is a `FAIL` whatever its
author thinks, the same principle the M2.1 Critic runs on.

---

## 10. Consequences

- The traceability chain `REQ -> UX/ARCH -> TASK` is recorded from the first document, so M7
  can build the impact graph without retrofitting identity onto documents written without it.
- `ProductService` is the whole surface a UI needs: agent roster with permissions, one method
  per stage, artifact inspection, project status — all plain serialisable data, no live
  handles into the engine.
- The seven-artifact list in the M4 spec is stored as **one** architecture document with
  views (`data_flow_view()`, `api_contract_view()`, …). Splitting them into independently
  versioned artifacts would let a data flow reference a component a newer architecture had
  removed, which is the dangling reference M4.6 forbids.
- **Measured, and negative:** the pipeline was run against the live
  `qwen2.5-coder:3b-instruct-q4_K_M`. The Product Manager **works** — one attempt, ~22s, a
  PRD that validates with five requirements and no blocking issues. The UX agent **fails**,
  six consecutive attempts across three runs, with `<root>: Input should be an object`: the
  model returns something that is not a JSON object at the top level.

  Two fixes were attempted and measured, neither sufficient. Flattening `flows -> steps ->
  next_steps` into a top-level `steps` list (the same shape the working PM schema uses) did
  not change the outcome. Making `product_name` optional — the model demonstrably omits the
  field it is asked to echo — did not either. A direct probe of the same prompt *did* return
  a well-formed object, so the failure is intermittent and specific to the
  `structured_generate` path, which appends a rendered schema instruction the probe did not
  send. The UX schema renders to ~4,900 bytes of JSON Schema; that instruction plus the PRD
  plausibly crowds an 8,192-token window.

  The Architect stage is therefore **unverified against a live model** — the pipeline never
  reaches it. Its schema is larger than the UX agent's, so the same failure should be
  assumed until measured.

  **The deterministic layer is unaffected.** Artifacts, validation, contradiction detection,
  review, storage, and the translation boundaries are all covered by 106 offline tests and
  do not depend on a model. What is unproven is whether a 3B model can drive the two larger
  stages, and the honest answer today is that one of them cannot.

  The next step is to split the UX and Architect stages into several smaller model calls —
  flows, then screens, then tokens — rather than one large structured output. That is the
  same lesson the M2 planner learned, applied one level further. It was not done here
  because it is a design change that deserves its own measurement, not a fix rushed in
  after a failing run.
- **Open:** the impact engine. Every M4 artifact exposes dependencies, affected components,
  and requirement references so M7 can build it; none of that graph is traversed yet.
