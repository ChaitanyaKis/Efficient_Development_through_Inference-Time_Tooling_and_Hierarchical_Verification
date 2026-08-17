# ADR 0005 — M3 Memory and Research

- **Status:** Accepted
- **Date:** 2026-08-13
- **Milestone:** M3

---

## 1. Memory is not chat history

A memory is a **claim plus its provenance**. `MemoryRecord.source_reference` is required by
the schema, so a fact with no traceable origin cannot be constructed, let alone stored.

The question the system must answer is not only *"what does Edith believe"* but *"why"*.
Supersession preserves the second: a changed decision writes a new record pointing at the
old one, the old one becomes `SUPERSEDED` rather than being overwritten, and
`history()` walks the chain backwards.

---

## 2. Sources are not equal

`MemorySource` carries a confidence baseline, and `to_record` **caps** confidence at that
baseline for untrusted sources. An agent cannot promote its own guess by asserting a high
number, because the ceiling is a property of where the claim came from, not of who is
making it.

| Source | Baseline | Auto-stored |
|---|---|---|
| `TEST_RESULT` | 0.95 | yes |
| `TOOL_OBSERVATION` | 0.90 | yes |
| `USER` | 0.90 | yes |
| `PROJECT_ARTIFACT` | 0.80 | yes |
| `EXTERNAL_RESEARCH` | 0.60 | no — needs approval |
| `AGENT_INFERENCE` | 0.40 | no — needs approval |
| `MODEL_SUGGESTION` | 0.25 | no — needs approval |

Deterministic evidence stores directly. Everything else is a *proposal*: it may be right,
but "may be right" is not what a knowledge base is for.

---

## 3. Isolation is a SQL predicate, not a convention

`visible_to(project_id)` builds its filter in the query. A caller cannot forget to scope a
read, because there is no unscoped read to call. The only cross-project path is
`MemoryScope.GLOBAL`, which the schema restricts to `ENGINEERING`, `FAILURE` and `DECISION`
— `PROJECT` and `TASK` memories are about one codebase by definition and are rejected at
construction if marked global.

---

## 4. Relevance gates retrieval; metadata only breaks ties

The first implementation scored importance, confidence and recency additively, which meant
an *irrelevant* memory still scored above zero and got injected into every prompt. That is
precisely how a memory system becomes a noise generator.

`LexicalRanker` now computes relevance first and returns zero if nothing ties the memory to
the query. Metadata is a tie-breaker among already-relevant memories and nothing more. The
same bug and the same fix appeared in the M2.1 Context Engine; it is worth stating as a
general rule: **a retrieval signal that does not mention the query cannot make something
relevant.**

---

## 5. Consolidation never rewrites

Duplicate detection is Jaccard similarity over titles and content; merging picks the
best-evidenced record as primary, transfers the combined recurrence count, and supersedes
the rest. Originals stay recoverable.

An LLM summariser that rewrote memories in place would destroy exactly the provenance that
makes them trustworthy, so it is not used here. If one is added later, the constraint stands:
the originals must survive.

---

## 6. Research: the provider retrieves, the model synthesises

`Claim.supported_by` has `min_length=1`. A claim with no evidence is unrepresentable, which
makes "the model recalled something from training" structurally distinct from research.

The model returns **source numbers**, never URLs — asked for a URL it will produce a
plausible one. `ground_claims` maps those indices back to sources that were actually
fetched and **discards** any claim whose citation does not resolve. Guessing which source
was meant would launder an invention into a citation.

Confidence follows the strongest supporting source's tier, not the model's conviction. Two
forum posts do not outweigh a specification.

---

## 7. Untrusted content, three defenses

Retrieved pages are hostile input. In order of importance:

1. **The Research Agent holds no tool gateway at all.** Its `AgentPermissions` are empty, so
   an instruction embedded in a page is addressing something with no ability to act. This is
   the real defense.
2. Content is **fenced and labelled** as untrusted data and only ever appears in a user
   message, never in a system prompt.
3. Instruction-shaped passages are **annotated, not deleted** — a page genuinely discussing
   prompt injection is legitimate research material, so it is quoted rather than censored,
   and the detection is surfaced.

Conflicts between sources are surfaced, never silently resolved. Research produces a
*recommendation*; an Architect makes the decision and an ADR records it.

---

## 8. Offline is a first-class state

`OfflineProvider` returns nothing and says why. `ResearchReport.unavailable_reason`
renders as `RESEARCH UNAVAILABLE`, with no summary, no recommendation, and zero confidence.
Fabricating a plausible answer when retrieval is impossible would be the worst available
behaviour, so it is impossible by construction.

Everything else in Edith — memory, coding, testing, verification — works with no network.

---

## 9. The memory experiment: a negative result

The measurable question M3 had to answer was whether engineering memory makes a weak local
model better at repairing code. On the `multi_repair` benchmark with
`qwen2.5-coder:3b-instruct-q4_K_M`:

| Arm | Success | Model calls | Repairs | Memory injected |
|---|---|---|---|---|
| baseline | 2/3 | 24.0 | 5.7 | 0 chars |
| memory | 0/3 | 25.0 | 6.0 | 1803 chars |

**Memory reduced the success rate.** The result is reported because it was measured, not
because it is flattering.

The plausible mechanism is context pressure: ~1800 characters of retrieved lessons compete
with the code itself inside an 8192-token window. On a 3B model that trade is not obviously
worth making, and this measurement suggests it is not.

Caveats stated plainly: this benchmark's variance across the project has been extreme
(0/4, 2/6, 2/3 for the same baseline configuration at different times), and n=3 per arm
cannot separate a real effect from that noise. What can be said is that **there is no
evidence memory helps this model on this task**, and some evidence it hurts.

The engineering conclusion is not "memory is useless" — it is that *injecting memory into
every coding prompt* is the wrong integration for a small-context model. Better candidates,
untested here: retrieving only after a failure, retrieving for the Debugger rather than the
Coder, or gating injection on a much higher relevance threshold.
