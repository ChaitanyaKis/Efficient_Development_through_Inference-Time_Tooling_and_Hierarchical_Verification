# Experiment 0002 — Requirement-aware targeted completion

- **Milestone:** M4.2
- **Date:** 2026-08-15
- **Status:** complete
- **Follows:** experiment 0001 (stage decomposition)

---

## Hypothesis

> Requirement-aware targeted completion can improve the sufficiency of decomposed artifacts
> without regenerating the entire artifact.

## Motivation

Experiment 0001 established that decomposition makes artifacts *valid*: 10/10 trials
produced schema-conforming, internally consistent, fully-referenced documents where the
monolithic path produced none. It also measured UX requirement coverage at **0.67** — the
specification was well-formed and silently omitted a third of what was asked for.

Validity checks cannot see that. A requirement nothing references is not a broken reference;
it is an absence, and absences do not fail schemas. So M4.2 asks a different question: can
the system detect the gap and close it *specifically*, without the shotgun of regenerating
the whole artifact and hoping the second attempt happens to mention what the first missed?

## Design

| | |
|---|---|
| **Independent variable** | completion strategy: none / detection only / targeted completion |
| **Dependent variables** | requirement coverage, critical coverage, missing, partial, model calls, runtime |
| **Model** | `qwen2.5-coder:3b-instruct-q4_K_M`, unchanged |
| **Brief** | the same "Stockroom" brief as experiment 0001, unchanged |
| **Trials** | 5 per arm, independent, none discarded |

### Arms

**A — baseline decomposed.** The M4.1 pipeline exactly.

**B — detection only.** Identical generation to A, plus the deterministic coverage matrix.
No correction. This arm exists to separate the benefit of *knowing* from the benefit of
*fixing*: any difference between A and B is noise, and any difference between B and C is
attributable to the completion pass rather than to measurement.

**C — targeted completion.** B, plus: for each gap, one narrow model call producing only the
missing element, validated, merged additively, and coverage recomputed from evidence.

### How coverage is computed

Deterministically, from element references — never from a model's opinion:

| state | meaning |
|---|---|
| `COVERED` | an element explicitly names the requirement id |
| `PARTIALLY_COVERED` | only a weak element names it (a screen with no flow; a decision with no component) |
| `MISSING` | nothing names it |
| `CONTRADICTED` | something names it *and* structurally conflicts with it |
| `NOT_APPLICABLE` | the requirement does not belong to this artifact kind |

A performance budget has no user flow, so it is `NOT_APPLICABLE` to a UX specification rather
than missing from it — forcing that mapping would manufacture a gap nobody should close.

### Controls against gaming

- Coverage requires an **element reference**. An overview that mentions "REQ-001" scores
  nothing; a test asserts this.
- A reference to a requirement the PRD never defined grants no coverage.
- The completion pass cannot mark its own success: the matrix is recomputed from the merged
  document, and a merge that did not close the gap leaves it open.
- Merges are **additive**. Existing element ids are never renumbered, so nothing already
  referenced by another artifact changes identity.
- The requirement a completion satisfies is attached **by the system**, not claimed by the
  model.

## Results

Five trials per arm. Every trial recorded; none discarded.

| arm | UX coverage | UX critical | architecture | missing | model calls | completion calls | runtime |
|---|---|---|---|---|---|---|---|
| A baseline | 0.667 | 0.667 | 1.000 | `REQ-002` | 11.0 | 0 | 51s |
| B detection only | 0.667 | 0.667 | 1.000 | `REQ-002` | 11.0 | 0 | 44s |
| C targeted completion | **1.000** | **1.000** | 1.000 | — | 12.0 | 1.0 | 47s |

Per-trial, with no variance in any arm:

```
A_baseline             ux: [0.667, 0.667, 0.667, 0.667, 0.667]   gaps closed: [0,0,0,0,0]
B_detection_only       ux: [0.667, 0.667, 0.667, 0.667, 0.667]   gaps closed: [0,0,0,0,0]
C_targeted_completion  ux: [1.0,   1.0,   1.0,   1.0,   1.0  ]   gaps closed: [1,1,1,1,1]
```

## Conclusion

**The hypothesis is supported.** Targeted completion raised UX requirement coverage from
0.667 to 1.000 in 5/5 trials, at a cost of **one additional model call** and no measurable
runtime penalty.

Four things are worth separating:

**The baseline is not noisy.** Arm A produced exactly 0.667 in all five trials, missing the
same requirement (`REQ-002`) every time. This is not a model that sometimes forgets — it is a
model that reliably under-covers, which is why a retry would not have helped and why the gap
was worth attacking specifically.

**Detection alone does nothing, as designed.** Arm B is identical to A on every measure. That
is the control working: the coverage matrix is a measurement instrument, and measuring
something does not change it. Any improvement in C is therefore attributable to the
completion pass rather than to the analysis.

**The correction is cheap.** 11 calls → 12. Compare the alternative that M4.2 item 4 rules
out: regenerating the whole UX specification costs 5 calls and, given the baseline's
consistency, would most likely have produced the same 0.667.

**Architecture needed no help.** Coverage was 1.000 in every arm, so the completion pass
correctly did nothing there — `complete_architecture_coverage` returns early when the
threshold is already met, spending zero calls. A correction mechanism that fires when there
is nothing to correct would be worse than none.

### What this does not show

Coverage is not correctness. A flow that names `REQ-002` covers it by this measure even if
the flow is a poor design. The experiment shows the requirement is now *addressed*; whether
it is addressed *well* is a further question, and the honest answer is that nothing here
measures it.

## Threats to validity

- **One brief, three requirements.** A coverage figure over three requirements moves in steps
  of 0.33; a larger brief would resolve the effect more finely.
- **One model, five trials.**
- **Coverage is not correctness.** A flow that names `REQ-003` covers it by this measure even
  if the flow is a poor design. Coverage measures whether the requirement was *addressed*,
  not whether it was addressed *well* — that is a further question this experiment does not
  answer.
- **The completion prompt is narrower than the pipeline prompt**, so part of any improvement
  may come from the narrower question rather than from targeting per se. Arm B controls for
  detection but not for prompt width.
