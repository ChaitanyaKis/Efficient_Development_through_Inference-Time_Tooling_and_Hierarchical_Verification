# ADR 0003 — M2 Core Autonomous Loop decisions

- **Status:** Accepted
- **Date:** 2026-08-12
- **Milestone:** M2

M2 is where Edith stops being infrastructure and starts doing work. Most of these decisions
were forced by watching the real 3B model fail in specific, reproducible ways.

---

## 1. Evidence outranks the model's verdict

**Decision.** `adjudicate()` applies deterministic rules first: if the tests failed, the
verdict is FAIL regardless of what the Critic said. The Critic only decides cases the
evidence leaves open, and only a *high-severity* finding can overturn passing tests.

**Rationale.** A small model asked "did this work?" says yes far too often. Verification is
the gate; the Critic is a second opinion on questions verification cannot answer.

---

## 2. The verifier is a separate principal from the coder

**Decision.** Verification runs under `VERIFIER_PERMISSIONS` — `shell.run` plus read access,
no write scope. The coder can write but has no `shell.run`.

**Rationale.** Found by the loop failing on its first real run: verification was using the
coder's gateway and was correctly denied. The fix was not to grant the coder shell access —
it was to notice that the principal deciding whether work passed should never be the one
that wrote it.

---

## 3. Deterministic gates in front of every write

**Decision.** Before any file is written the coder checks, in order: content sanitisation
(strip context markers and code fences), syntax validity (`ast.parse`), and symbol
preservation (no existing top-level definition disappears unless declared in
`removes_definitions`).

**Rationale.** Each gate exists because the real model did the thing it prevents:

- it copied the context delimiter `--- FILE: calculator.py ---` into the file as line 1;
- it wrote `"""A tiny arithmetic library.""` — a docstring closed with two quotes;
- asked to *add* `multiply`, it returned a file with `add` and `multiply` and silently
  dropped `subtract`, producing an ImportError that looks nothing like the real mistake.

A prompt cannot reliably prevent any of these. A parser can. CLAUDE.md: prefer deterministic
tooling over LLM judgment.

---

## 4. Smallest edit primitive, not whole-file rewriting

**Decision.** `FileEdit.mode` offers `append`, `replace_function`, and `replace_file`. The
prompt asks for the narrowest one that does the job.

**Rationale.** Whole-file rewriting requires the model to faithfully reproduce every line it
is *not* changing, which is exactly what a 3B model is worst at. `append` for a new function
and `replace_function` for a bug fix ask it to emit only the new code. This single change
moved the feature benchmark from "failing after three attempts" to "passing on the first".

**Consequence.** `replace_function` locates the target by AST, so decorators and nested
bodies are handled exactly. The function name is *inferred from the content* when the model
omits it — observed rejecting three consecutive correct fixes over an empty `function_name`
while the debugger had diagnosed the bug perfectly each time.

---

## 5. The final gate, not per-task verification, decides success

**Decision.** Task-level verification drives the per-task repair loop, but the authoritative
judgement is a single verification after every task has run. A task that changed files but
left the suite red is *deferred* rather than failed while other tasks remain.

**Rationale.** Verification runs the project's whole test suite. In a three-task plan where
each task fixes one defect, the first two tasks *cannot* see green no matter how correct
they are — so they would fail, block their dependents, and doom a plan that was working.
This was not a hypothesis; it is what the multi-defect benchmark did.

---

## 6. The plan is a proposal, not an instruction

**Decision.** `plan_to_tasks()` re-expresses everything the model produced through the strict
`Task` schema: step numbers become opaque ids, dangling dependencies are dropped, and write
scope is *derived from the files the step named* rather than granted wholesale.

**Rationale.** The planner's output is untrusted input. A task that said it would touch
`calculator.py` gets a gateway that can write only `calculator.py`, which is how the loop
refused to edit a protected test file even when the model tried.

---

## 7. `python` means the interpreter Edith is running under

**Decision.** `resolve_executable` maps `python`/`python3` to `sys.executable` rather than
the first match on `PATH`.

**Rationale.** Edith runs from a virtualenv where pytest lives; `PATH` pointed at a system
interpreter without it. Every verification therefore failed with `No module named pytest`,
was classified `TEST_FAILURE`, and sent the Debugger hunting for a bug in code that had
never run. The runner-missing case is now classified `ENVIRONMENT_FAILURE` as well.

---

## 8. Security failures abort; nothing else is unbounded

**Decision.** `SECURITY_FAILURE` maps to `ABORT` and is never downgraded by attempt budgets.
Every other category resolves to RETRY, REPAIR, or ESCALATE, and both RETRY and REPAIR decay
to ESCALATE once the budget is spent.

**Rationale.** A denied path is a policy decision, not a transient fault. Retrying it is
indistinguishable from an agent probing the sandbox.

---

## 9. Workspaces live outside the kernel

**Decision.** Project workspaces default to `../Edith_Workspaces`, and `WorkspaceManager`
refuses — non-configurably — any workspace that is, contains, or lives inside the Edith
repository.

**Rationale.** M1 shipped `workspace_root: .`, which is right for an operator invoking a tool
by hand and catastrophic for an autonomous loop: a plausible-but-wrong plan would edit the
kernel currently executing it.

---

## 10. Benchmarks audit Edith from outside

**Decision.** The harness re-runs the project's own checks with `subprocess` directly, hashes
every protected file against the fixture, and treats Edith's self-reported verdict as the
*last* of three conditions rather than the first. Protected files are additionally made
unwritable through M1's path policy for the duration of the run.

**Rationale.** The `repair` benchmark caught the agent rewriting `assert subtract(5, 3) == 2`
into `== 8` — making the test match the bug and declaring victory. Edith's own view of that
run was PASS. A benchmark that trusts the system under test measures nothing.
