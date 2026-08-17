# ADR 0011 — M5 Engineering Execution Layer

- **Status:** Accepted
- **Date:** 2026-08-15
- **Milestone:** M5

---

## 1. Boundaries are permissions, not prompts

Five roles — frontend, backend, database, devops, dependency — each declare repo-relative
write patterns and a set of *forbidden* areas. Those become `AgentPermissions`, and the M1
gateway enforces them.

A frontend agent asked to write a migration does not receive a polite refusal from its
system prompt. Its write fails at the policy layer, is classified `SECURITY_FAILURE`, and
never reaches disk. The prompt says what the role is for; the gateway decides what it can
reach; when they disagree the gateway wins.

Scope narrows twice. The **role** bounds what an agent may ever touch. The **task** bounds
what it may touch now: a backend task naming `src/backend/api.py` gets that file and its
directory, intersected with the role ceiling. A task naming a path outside its role is
refused before anything runs.

No engineering agent holds `shell.run` or any git tool. Generating code and deciding it works
stay in different hands, which is M2.1's separation applied to specialisation.

## 2. Specialised agents subclass the M2 coder

Only the prompt differs. The sanitiser, syntax gate, symbol-preservation check, gateway
write, rejection accounting and diff are all inherited.

This was a deliberate reversal during implementation: the agents originally emitted raw
`ModelEdits` and would have needed their own application pipeline. Five copies of that
pipeline is five chances to weaken a guarantee M2.1 established once. `EngineeringInput`
therefore *extends* `CoderInput` rather than replacing it, which is what lets the inherited
`_run` work untouched.

## 3. A task is not complete because code was generated

Completion requires, in order: the agent produced edits, the gateway accepted at least one,
the generated modules **import**, and the configured verification passes.

The import gate was added because the benchmark exposed a vacuous check. The model wrote
`collections.defaultdict` without importing `collections`; the syntax gate passed (it parses
fine) and the task suite passed (it never imported the new module), so the task was recorded
COMPLETE while the application was broken. That is the same failure shape as M2.1's missing
test runner: the check ran and said nothing about the artifact. For Python, importability is
the build, and it is now a gate.

Rejections are repaired within a bounded budget — same agent, same scope, plus the real
failure evidence — and only rejections. An agent that produced nothing at all will not be
helped by being shown its own absence.

## 4. Conflicts are detected before execution

Two tasks whose write scopes overlap, where neither depends on the other, are a
`TASK_CONFLICT`: the outcome depends on which ran last. A dependency between them removes
the conflict, because the DAG already serialises them.

M5 runs sequentially, so a conflict harms nothing today. It is still reported and the order
made deterministic, because "these two tasks are order-dependent and nobody said so" is a
defect in the plan that becomes a race the moment execution parallelises.

## 5. Dependency work is deterministic

The Dependency Agent's real work is M3.1's: imports come from the AST, declarations from the
manifests, installed versions from the interpreter. A model is not asked which packages a
project needs, because the parser already knows.

Only *discovered* imports are promoted into manifests. A package a model suggested is
representable (`MODEL_SUGGESTION`) and never installed unreviewed. Install artifacts are
generated text written through the gateway; nothing here executes an installer.

## 6. Results

Three trials per arm, live `qwen2.5-coder:3b`, same fixed plan:

| arm | tasks completed | runnable | model calls | repairs | blocked | scope violations |
|---|---|---|---|---|---|---|
| specialised | **6/6** | 1/3 | 2.0 | 1.0 | 0 | 0 |
| generic (one broad-scope coder) | 0/6 | 0/3 | 1.0 | 0 | 3 | 0 |

The generic arm failed `TASK-001` in all three trials, which blocked `TASK-002` every time.

**The comparison is confounded and must not be read as "specialisation wins".** The
specialised arm ran through the executor, which has a bounded repair loop; the benchmark's
generic arm does not. Repairs averaged 1.0 per trial, so repair contributed to the
difference. What the numbers support is that *the executor* — specialisation plus scoping
plus repair plus the import gate — completes tasks the single-shot generic path does not.
Isolating specialisation alone requires giving the generic arm the same repair loop, and that
measurement has not been made.

**Runnable in 1/3.** All six tasks completed and imported, and in two of three trials the
application still failed an independently-written acceptance test — the generated interface
did not match the specified one. Completing a task is not the same as building the right
thing, and the gap between 6/6 completed and 1/3 runnable is exactly that distance.

## 6b. M5.1 — hardening and a fair comparison

M5's comparison was confounded and said so. M5.1 removes the confound structurally rather
than by adjusting the benchmark: `EngineeringRole.GENERIC` is a *role*, so the control arm
runs through the identical executor, repair loop, import gate, verification, and gateway.
The only remaining difference between the arms is which agent — and therefore which prompt
and which scope — handles each task.

Its scope is computed as the union of the five specialised scopes rather than hand-listed, so
a control cannot silently fall behind when a specialised role gains a path. A test asserts
the union actually covers every role, and it caught a real gap the first time it ran.

Three further hardening changes:

**Workspace isolation.** `TaskWorkspace` records the workspace id, task id, base revision,
path, and owning execution; `WorkspaceLedger.may_write` answers the cross-workspace question
that the M1 path policy does not — a path inside *another task's* tree is refused even when
the writing agent's own permissions would allow the shape of it. Isolation is the existing
permission system applied to a narrower root, not a second one.

**Merge safety.** `may_merge` checks every condition explicitly: the task was verified, no
blocking issue remains, the workspace belongs to this task, the base revision is known, and
the workspace has not already been resolved. Nothing merges by last-write-wins, and a
refusal names the condition that failed.

**Explicit quality states.** `GENERATED → TASK_COMPLETE → VERIFIED → INTEGRATED`, plus
`REPAIR_EXHAUSTED` as a terminal outcome distinct from `REJECTED`. M5's most valuable finding
was that 6/6 tasks complete and 1/3 applications runnable are different numbers; collapsing
them into one boolean would have hidden it.

## 7. Consequences

- The plan → code path exists end to end and is gated by evidence at every step.
- **Open: quality beyond importability.** A module that imports and passes the project's own
  tests can still implement the wrong interface, which is exactly what the benchmark's
  acceptance failures were.
- **Open: no worktree isolation yet.** Tasks execute sequentially in one workspace. M1's
  `git.worktree` exists and the design does not have to change to use it, but M5 does not.
- **Open: frontend, devops and dependency agents are unexercised by the benchmark**, which is
  Python-only so that "is it runnable?" has an unambiguous answer on this hardware.
