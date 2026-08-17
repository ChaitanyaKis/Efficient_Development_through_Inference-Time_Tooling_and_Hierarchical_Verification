# ADR 0010 — M4.2 Requirement Coverage and Targeted Completion

- **Status:** Accepted
- **Date:** 2026-08-15
- **Milestone:** M4.2
- **Builds on:** ADR 0009 (stage decomposition)
- **Experiment:** `docs/experiments/0002-targeted-completion.md`

---

## 1. Validity is not sufficiency

M4.1 shipped a pipeline that produces valid artifacts 10/10 times and measured UX
requirement coverage at 0.67. Both facts are true at once, and they are not in tension: a
specification can satisfy every schema, resolve every reference, and contradict nothing,
while never mentioning a third of what the user asked for.

Validation cannot catch that. A requirement nothing references is an *absence*, and absences
do not fail schemas — there is no dangling pointer to find. Coverage is therefore a separate
analysis with its own model, its own evidence, and its own gate.

## 2. Coverage is computed from evidence, never from opinion

`analyse_coverage` classifies each requirement against each artifact:

| state | basis |
|---|---|
| `COVERED` | an element explicitly names the requirement id |
| `PARTIALLY_COVERED` | only a weak element names it |
| `MISSING` | nothing names it |
| `CONTRADICTED` | something names it *and* structurally conflicts with it |
| `NOT_APPLICABLE` | the requirement does not belong to this artifact kind |

Two distinctions carry real weight.

**Strong versus weak evidence.** A flow covers a requirement; a screen alone does not. A
screen with no flow is somewhere the user can reach with no journey that reaches it. Likewise
a component covers; a decision alone is an intention nobody implements. Collapsing these into
"referenced / not referenced" would let an artifact score full coverage for acknowledging
requirements it never delivers.

**Not-applicable is not coverage and not a gap.** A performance budget has no user flow. It is
excluded from the UX fraction's numerator *and* denominator, so it neither counts against the
specification nor inflates it. Forcing the mapping would manufacture a gap nobody should
close — and the opposite, counting it as covered, would be the gaming M4.2 item 10 forbids.

`CoverageEntry` has a `critic_note` field for an advisory model opinion. It is stored beside
the computed state and can never change it. An LLM-only coverage judgement becoming
authoritative is exactly how a gap gets marked closed with nothing built.

## 3. The threshold is a policy, stated and testable

M4.2 item 6 forbids declaring 100% mandatory by default. Criticality derives from the
requirement's own priority — `MUST` is critical, `SHOULD` important, `COULD` optional — so the
bar is a property of what the user asked for rather than of what the system found convenient.
`WONT` is excluded entirely: an explicit decision not to build something is not a gap.

`CoverageThreshold` defaults to: every critical requirement covered, at least half of the
important ones, no overall floor. A project with different standards states them rather than
editing the engine.

## 4. Critical gaps block approval, through the existing gate

A gap becomes a `ValidationIssue`: `COVERAGE_GAP` (blocking) for critical requirements and
contradictions, `ADVISORY_COVERAGE_GAP` (non-blocking) otherwise. Those flow into the same
`ValidationOutcome` the M4 approval path already checks.

This is deliberate. There is no second gate to remember and no way to approve an artifact that
does not do what was asked — the mechanism that refuses a dangling reference refuses an
unaddressed critical requirement, for the same reason and in the same place.

A contradiction blocks at *any* criticality. An artifact that claims to address a requirement
while structurally conflicting with it is worse than one that omits it, because it looks
finished.

## 5. Targeted completion, not generic retry

The obvious response to 0.67 coverage is to re-run the specification. M4.2 item 4 rules it
out, and it would not work anyway: the second attempt has no more reason to mention the
missed requirement than the first did.

Targeted completion is a different shape:

    detect gap -> generate only what is missing -> validate -> merge -> re-check coverage

One narrow call, one requirement, a schema smaller than any pipeline stage. Four properties
make it safe:

- **Additive merges.** Ids continue from the highest existing element, so nothing already
  referenced by another artifact changes identity.
- **The system attaches the requirement.** The model produces content; Edith decides what it
  satisfies. That is what makes the resulting evidence trustworthy.
- **A failed merge is discarded**, not half-applied. An invalid artifact is worse than an
  incomplete one.
- **The recheck is authoritative.** Coverage is recomputed from the merged document. A pass
  that merged something which did not close the gap reports the gap as still open — a
  completion cannot declare its own success.

The attempt budget is bounded at four. An artifact with twelve gaps has a problem that twelve
extra model calls will not fix.

## 6. Results

Five trials per arm on the M4.1 brief and model:

| arm | UX coverage | critical | model calls | completion calls |
|---|---|---|---|---|
| A baseline | 0.667 | 0.667 | 11.0 | 0 |
| B detection only | 0.667 | 0.667 | 11.0 | 0 |
| C targeted completion | **1.000** | **1.000** | 12.0 | 1.0 |

No variance in any arm. The baseline missed the same requirement in all five trials, which is
why a retry would not have helped: this is reliable under-coverage, not noise. Detection-only
is identical to baseline on every measure — the control working as intended, so the
improvement in C is attributable to the correction rather than to the analysis.

Architecture coverage was 1.000 throughout, and the completion pass correctly spent zero
calls there. A correction that fires when there is nothing to correct would be worse than
none.

## 7. Consequences

- Coverage is part of what "done" means for a product artifact, enforced rather than reported.
- Targeted completion is available but **off by default** until the full experiment justifies
  the extra calls — the same discipline M3.2 applied to memory.
- **Open: coverage is not correctness.** A flow naming `REQ-003` covers it by this measure
  even if the design is poor. This milestone measures whether a requirement was *addressed*,
  not whether it was addressed *well*.
- **Open: prompt width is uncontrolled.** Arm B isolates detection from correction, but the
  completion prompt is also narrower than a pipeline prompt, so some of any improvement may
  come from asking a smaller question rather than from targeting specifically.
