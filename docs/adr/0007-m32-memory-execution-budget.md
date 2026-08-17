# ADR 0007 — M3.2 Memory Execution Budget and Context Governor

- **Status:** Accepted
- **Date:** 2026-08-15
- **Milestone:** M3.2
- **Builds on:** ADR 0006 (M3.1 strategies), ADR 0005 (M3 memory subsystem)

---

## 1. The defect: a per-prompt limit is not a limit

M3.1 configured `max_chars = 1200` per prompt and observed it exactly. Executions still
injected up to 14,202 characters, because a repair loop retrieves again after every failure:

```
retrieve (1.2k) → repair → fail → retrieve (1.2k) → repair → fail → retrieve (1.2k) → …
```

Each individual prompt was compliant. The execution was not. On a 3B model with an
8,192-token window, the accumulated total is the number that competes with the code.

The unit of accounting is therefore the **execution** — the thing that actually has a
context cost — not the prompt and not the agent.

---

## 2. The budget belongs to the execution, and nothing can raise it

`ExecutionMemoryBudget` holds five ceilings (`max_total_chars`, `max_retrievals`,
`max_total_memories`, `max_chars_per_retrieval`, `max_memories_per_retrieval`) and the
consumption against them.

Three structural properties, each chosen so the guarantee does not depend on anyone
remembering it:

- **Counters are read-only properties** over private state. `budget.consumed_chars = 0`
  raises `AttributeError`. The only mutator is `record_injection`, which the governor calls.
- **Limits are a frozen dataclass.** A limit that can be reassigned mid-run is advisory.
- **Consumption only moves one way.** A negative charge is floored at zero, so
  "charge −1000 characters" cannot become a budget reset with extra steps.

`exhausted` is true when **any** dimension is spent, not all of them. A budget with
characters left but no retrievals left is spent, and pretending otherwise is how a ceiling
degrades into a suggestion.

---

## 3. One gate, and no path around it

Every autonomous injection goes through `MemoryGovernor.request(execution_id, query,
purpose, …)`. The signature is the enforcement: a caller states *where in the loop it is*
and receives what that position entitles it to, rather than passing retrieval parameters it
could inflate.

The bypass surface is closed structurally rather than by convention:

| Path | Why it cannot be taken |
|---|---|
| An agent retrieving directly | No module under `src/edith/agents/` imports `edith.memory` at all — asserted by a test |
| The orchestrator retrieving directly | `_retriever.retrieve` appears nowhere in `orchestrator.py` — asserted by a test |
| An agent editing its budget | No agent input or output schema names a budget; `AgentRequest` is `extra="forbid"` |
| One execution spending another's allowance | `request()` refuses an `execution_id` that is not the budget's |

`MemoryRetriever.retrieve` still exists as the low-level, unbudgeted path for the CLI and
for administration, and its docstring says so explicitly. That separation is item 11 of the
milestone: the unrestricted method is available to an operator and unreachable from the loop.

Project isolation is enforced in two places that do different jobs. The governor always
supplies the execution's project scope — a caller cannot pass a project id, so it cannot
ask for another project's memory — and never widens the result. The store then enforces the
predicate in SQL, so isolation does not depend on the governor being correct. Defence in
depth, not duplication: either layer alone would hold, and neither is trusted to.

---

## 4. Fail closed

An exhausted budget returns `GrantOutcome.BUDGET_EXHAUSTED` (wire value
`MEMORY_BUDGET_EXHAUSTED`) with empty text. It does not grow the budget, retry, borrow
against a later prompt, or raise.

Refusals cost nothing, so a run that hits the ceiling repeatedly does not keep paying for
the privilege. The count is reported as `budget_exhaustions` rather than swallowed — "the
budget was exhausted four times" is exactly the measurement this milestone exists to
produce, and it is invisible if only successful grants are counted.

The loop continues without memory. That is not a degraded mode: M3.1 measured the
no-memory arm as the *best* performing one, so a memoryless execution is the baseline, not
a casualty.

---

## 5. Nothing is sent twice

The budget keeps a ledger of every memory id already injected. A repeat is referenced by id
and first injection point rather than re-sent:

```
- (already provided earlier in this run at debugger: Ordering in assertions [mem_ab12…])
```

Re-spending several hundred characters to repeat a lesson the model has already been shown
is the purest form of the waste M3.1 found. Reference lines are charged too — cheap, but not
free, because free would be a leak.

---

## 6. Accounting is exact, not approximate

The governor renders the text first and charges `len(text)`. The pre-selection estimate is
computed from the render template itself, including the type label and every literal
character, so the estimate can never run under the truth. An estimate that under-counts
lets an injection overshoot the ceiling it was just checked against.

`memory_chars` in any M3.2 report therefore means literal characters that reached a prompt.

Every injection is recorded to `state.memory_injections`: memory ids, relevance scores,
titles, cost, reason, agent, retrieval point, and remaining budget.

**Ids and titles, never claim content.** The claims live in the memory store, which is where
a user inspects and deletes them; copying the text into the state database would create a
second copy with a second deletion path, and a memory the user deleted would go on living.
Titles are the deliberate exception — a resumed execution needs them to render its
"already provided earlier" reference line without re-reading a memory it must not re-send —
and a test asserts the claim body never appears in a ledger row.

The same rule governs logs: `memory.granted` records ids, scores, and costs. It does not
record what the memories say.

---

## 7. A restart continues the budget

`_build_governor` seeds the budget from `store.memory_consumption(execution_id)`. An
interrupted execution resumes with its allowance already partly spent and its ledger
intact, so it neither re-sends what the model saw before the crash nor receives a fresh
allowance.

Without this, "crash and retry" would be an unlimited memory supply — the same unbounded
accumulation, reached by a different route.

This required state schema v2. Because every migration so far is an additive
`CREATE TABLE IF NOT EXISTS`, an older database is now migrated forward on open rather than
rejected; only a *newer* schema than the build understands is refused.

---

## 8. Prioritisation under a tight budget

Ordering is `(-score, -confidence, type_rank, memory_id)`. The ranker has already weighted
failure-text and component relevance above task-title overlap (ADR 0006 §3), so score leads;
confidence then prefers better-evidenced claims, and `FAILURE` outranks `ENGINEERING`
outranks `PROJECT`. The id makes the order stable, so the experiment is reproducible.

The relevance gate remains mandatory and independent of the budget. Spare capacity is never
a reason to inject something irrelevant.

---

## 9. Experiment

See §10 for results. Three arms on `multi_repair` with `qwen2.5-coder:3b-instruct-q4_K_M`,
identical retry limits, sequential inference, same hardware:

- **A** — no memory (the M3.1 winner, and the control)
- **B** — debugger-only memory, **no** execution budget (the M3.1 behaviour)
- **C** — debugger-only memory, **with** the medium execution budget

B and C differ by one number. The governor runs in all three arms, so the comparison
measures the budget rather than two implementations.

A budget-size ablation (`small` / `medium` / `large`) is available via
`edith budget --ablation`.

---

## 10. Results

`multi_repair`, `qwen2.5-coder:3b-instruct-q4_K_M`, 3 runs per arm, sequential inference,
identical retry limits, same hardware.

### A/B/C — does the budget bound the cost?

| arm | pass | mean chars | peak chars | retrievals | exhaustions | model calls | repairs | false pos |
|---|---|---|---|---|---|---|---|---|
| A — no memory | 1/3 | 0 | 0 | 0.0 | 0 | 25.0 | 6.0 | 0 |
| B — debugger, unbudgeted | 0/3 | 4,376 | 4,626 | 6.0 | 0 | 25.0 | 6.0 | 0 |
| C — debugger, budgeted | 1/3 | 1,772 | 1,775 | 3.0 | 8 | 24.0 | 5.7 | 0 |

**The budget does what it was built to do.** Mean cost fell 4,376 → 1,772 and peak fell
4,626 → 1,775, against a configured ceiling of 2,400. Retrievals capped at exactly 3.
Eight exhaustion events fired across three runs, so fail-closed is not theoretical — it is
the path those runs actually took, and they completed anyway.

Peak matters more than mean here. An average hides the one execution that ran away; the
budget flattened the worst case by 62%, and B's peak sat only 250 characters above its own
mean because *every* unbudgeted run accumulated.

One nuance worth stating plainly: B is **not** the M3.1 behaviour reproduced exactly. The
governor runs in every arm, so B already has duplicate suppression. M3.1's unbudgeted
debugger-only arm averaged 6,099 characters; suppression alone brought that to 4,376, and
the execution ceiling brought it to 1,772. Two independent mechanisms, both contributing.

### Ablation — what does a tighter allowance cost or buy?

| budget | pass | mean chars | peak chars | retrievals | exhaustions | ceiling |
|---|---|---|---|---|---|---|
| small | **2/3** | 314 | 314 | 1.0 | 14 | 800 |
| medium | 1/3 | 1,706 | 1,767 | 3.0 | 9 | 2,400 |
| large | 1/3 | 3,595 | 3,915 | 5.3 | 0 | 4,800 |

Every preset held its ceiling. The `large` arm never exhausted, which is the correct
behaviour for a ceiling set above what the loop actually wants.

The medium budget was measured twice under different names — arm C above and `budget_medium`
here are the same configuration — and produced 1,772 and 1,706 mean characters, 1/3 and 1/3.
That reproducibility is the strongest evidence in this table that the accounting is sound.
Pooled, the medium budget is **2/6**.

### What may and may not be concluded

- **Supported.** The execution budget bounds memory cost, exactly and reproducibly, without
  breaking the loop. Every ceiling held; exhaustion never prevented completion; zero false
  positives in any arm, so the M2.1 integrity gate held throughout.
- **Suggestive.** Success is monotone in *less* injected memory: 2/3 at 314 characters,
  1/3 at 1,706, 1/3 at 3,595, 0/3 at 4,376. This is the same direction M3 and M3.1 found,
  now visible within a single controlled ablation.
- **Not supported.** That budgeted memory beats no memory. A=1/3 and C=1/3 pooled to 2/6;
  three runs cannot separate one success from two. The best-performing memory arm (`small`,
  314 characters) is also the arm closest to injecting nothing, which is not an argument for
  memory.

**M3.2 makes memory safe. It did not make memory useful.** The default strategy therefore
stays `none` — the budget is what makes any future decision to enable it survivable, not a
reason to enable it now.

---

## 11. Consequences

- Memory's cost per execution is bounded by construction and measurable after the fact.
- The default memory strategy remains `none` (ADR 0006): the budget makes memory *safe*,
  which is a different claim from making it *beneficial*, and only the second would justify
  changing the default.
- Context accounting is durable, so a repair loop's memory cost can be audited from state
  rather than inferred from logs.
- **Open:** token-level accounting. Characters are a proxy; the provider does not expose a
  tokeniser through the abstraction, and adding one for this alone would couple the memory
  layer to a specific runtime. Characters bound the real quantity monotonically, which is
  enough for a ceiling.
