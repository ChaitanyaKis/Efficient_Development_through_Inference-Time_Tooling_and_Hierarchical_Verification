# Experiment 0001 — Inference-time stage decomposition

- **Milestone:** M4.1
- **Date:** 2026-08-15
- **Status:** complete

---

## Hypothesis

> Decomposing large structured-generation tasks into smaller validated inference steps
> improves the reliability of a fixed 3B local coding model.

Stated as a falsifiable prediction: a UX specification generated through four small schemas
will validate more often than the same specification generated through one large schema, on
the same model, from the same brief.

## Motivation

M4 measured a monolithic UX generation call failing six consecutive times on
`qwen2.5-coder:3b-instruct-q4_K_M` with `<root>: Input should be an object` — the model
returning something that is not a JSON object at the top level. The Product Manager call,
whose schema is roughly a third the size, succeeded on its first attempt.

Two explanations fit that observation:

1. **Task difficulty.** Designing an interface is harder than writing requirements.
2. **Schema size.** The model cannot hold a 4,800-byte target structure while also reasoning
   about the content.

They make different predictions. If (1) is right, decomposition changes nothing — the model
still cannot do UX. If (2) is right, the same task in smaller pieces succeeds.

## Design

| | |
|---|---|
| **Independent variable** | generation strategy: monolithic vs decomposed |
| **Dependent variables** | validity, completeness, requirement coverage, runtime, model calls, context cost |
| **Model** | `qwen2.5-coder:3b-instruct-q4_K_M`, unchanged |
| **Hardware** | RTX 2060 6GB, 16GB RAM, sequential inference |
| **Brief** | the M4 "Stockroom" product brief, identical across every arm and trial |
| **Trials** | 5 per arm, independent, none discarded |

Four arms: `ux_monolithic`, `ux_decomposed`, `architect_monolithic`, `architect_decomposed`.

### Decomposition

**UX** — four stages. `flows` (which journeys exist), `steps` (run once *per flow*), `screens`
(screens and their states), `presentation` (components, accessibility, tokens).

**Architecture** — six stages. `components`, `data`, `api`, `decisions`, `threats`, `plan`.

Each downstream stage receives only what it needs — requirement lines rather than the whole
PRD, component *names* rather than the assembled architecture — so context cost does not
scale with the number of stages.

### What counts as success

A model returning valid JSON is not sufficient. A trial succeeds only if it produced valid
structure **and** valid references **and** requirement coverage **and** no blocking
contradictions **and** correct system-owned authority **and** every required stage **and** a
persisted artifact. Each gate is recorded separately, so well-formed nonsense is
distinguishable from a usable specification.

### Controls

- No prompt was changed between arms except as the decomposition itself requires.
- No validation was weakened. Stage schemas remain `extra="forbid"`, required lists remain
  non-empty, and decisions still require alternatives and consequences.
- Ids, versions, status, authority, and product identity remain system-owned in both arms. A
  test asserts no schema in either arm asks the model for them.

## Results

Five trials per arm. Every trial is recorded; none was discarded or re-rolled.

| arm | successful | schema-valid | complete | artifact valid | no blocking contradiction | req. coverage |
|---|---|---|---|---|---|---|
| `ux_monolithic` | **0/5** | 0/5 | 0/5 | 0/5 | 0/5 | 0.00 |
| `ux_decomposed` | **5/5** | 5/5 | 5/5 | 5/5 | 5/5 | 0.67 |
| `architect_monolithic` | **0/5** | 0/5 | 0/5 | 0/5 | 0/5 | 0.00 |
| `architect_decomposed` | **5/5** | 5/5 | 5/5 | 5/5 | 5/5 | 1.00 |

### Cost

| arm | model calls | mean runtime | largest schema | largest single prompt | total prompt |
|---|---|---|---|---|---|
| `ux_monolithic` | 1 | 132.1s | 4,800 B | 2,303 | 2,303 |
| `ux_decomposed` | 5 | **20.7s** | **1,614 B** | **1,068** | 4,902 |
| `architect_monolithic` | 1 | 99.1s | 6,968 B | 2,851 | 2,851 |
| `architect_decomposed` | 6 | **30.6s** | **2,138 B** | **1,629** | 6,404 |

Every monolithic failure classified as `RETRY_EXHAUSTED` — the model never produced a
schema-conforming object within its attempt budget, on any of ten trials.

## Conclusion

**The hypothesis is supported.** Explanation (2) holds and (1) is refuted: the same model,
on the same brief, produces a complete valid artifact in 10/10 decomposed trials and 0/10
monolithic ones. The failure was structural capacity, not task difficulty.

Three findings worth separating:

**Reliability: 0% → 100%.** Not a marginal improvement. On this model the monolithic path
does not work at all, and the decomposed path did not fail once.

**Decomposition is faster despite costing more calls.** 132s → 21s for UX, 99s → 31s for
architecture, at 5× and 6× the call count. This is a property of failure being expensive: the
monolithic arm spends three full attempts failing before it gives up. A strategy that costs
more calls and less time is unusual, and it is worth being clear that the saving comes from
not failing rather than from small calls being individually cheap.

**Total context roughly doubles; per-call context halves.** Decomposed UX sends 4,902
characters across five prompts where monolithic sends 2,303 in one. Per-call cost — the
number the model actually has to hold — drops from 2,303 to 1,068. That is the trade being
made, and it is the right way round for a model whose limit is per-call capacity.

### Quality, not just validity

Requirement coverage differs between the two decomposed agents: architecture covers 3/3
requirements, UX covers 2/3. So decomposition produced *valid and complete* specifications
that still miss a requirement about a third of the time. That is a real limitation and the
obvious next measurement — validity was the M4.1 question, and sufficiency is a different one.

## Conclusion

Explanation (2) is supported and (1) is refuted. The same model, given the same brief,
produces a complete and valid UX specification every time when the task is split into four
small schemas, and never when it is one large one. The failure was structural capacity, not
task difficulty.

The tradeoff is real and worth stating: decomposition costs 5 model calls instead of 1. It is
nonetheless **faster in wall-clock time**, because the monolithic path spends three full
attempts failing before it gives up. A strategy that costs more calls and less time is an
unusual shape, and it is a property of failure being expensive rather than of decomposition
being cheap.

## Threats to validity

- **One brief.** All trials use the same product. A brief with more requirements would
  produce longer stage prompts and might reintroduce the failure at a larger size.
- **One model.** Nothing here generalises to a larger model, which may handle the monolithic
  schema fine and gain nothing from decomposition.
- **Five trials.** Enough to separate 0/5 from 5/5 with confidence; not enough to distinguish
  4/5 from 5/5 in any arm where the result is less extreme.
- **Grammar unsupported.** The runtime declines to compile the schema into a decoding
  grammar and falls back to `format=json` on every call, in both arms. On a runtime that
  enforced the grammar, the monolithic arm might not fail this way at all.

## Reproduction

The decomposed path is the default:

```
edith product plan --project demo --idea "<brief>"
edith product plan --project demo --idea "<brief>" --monolithic   # the baseline arm
```

The experiment harness lives in `edith.product.experiment`; the arms are
`run_ux_monolithic_trial`, `run_ux_decomposed_trial`, `run_architect_monolithic_trial`, and
`run_architect_trial`. Every trial is recorded in the output, including failures.
