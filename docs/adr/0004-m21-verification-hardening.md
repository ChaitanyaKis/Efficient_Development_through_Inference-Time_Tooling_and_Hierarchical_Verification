# ADR 0004 — M2.1 Verification hardening

- **Status:** Accepted
- **Date:** 2026-08-12
- **Milestone:** M2.1

M2 shipped a verification architecture that reduced to *code + current test results*. It
failed in exactly the way that architecture has to fail. This milestone fixes the shape of
the problem, not the instance.

---

## 1. Why the Judge accepted a rewritten test

Asked to repair a broken `subtract`, the coding agent changed
`assert subtract(5, 3) == 2` into `== 8`. The suite went green. The Critic returned PASS.
Only the external harness caught it.

Tracing it precisely: `adjudicate()` checked, in order, whether verification could run,
whether any check failed, and whether files changed. All three were satisfied — the tests
genuinely passed — so it deferred to the Critic, which said PASS.

**There was no rule anywhere that compared the tests to what they used to be.** The Critic
received the diff and could in principle have noticed; a 3B model did not. The single defense
was model judgement, which is precisely the thing CLAUDE.md says must never be the last word.

The lesson is not "the Critic needs a better prompt". It is that **a system cannot verify
itself against a definition of correctness that it is also allowed to edit.**

---

## 2. Verification now takes four inputs, not two

```
CODE + ORIGINAL REQUIREMENTS + BASELINE TESTS + CURRENT TESTS + EXECUTION EVIDENCE
```

- **Original requirements** live on the `Task`, derived from the user request before any
  agent ran, and are never rewritten by an agent.
- **Baseline tests** come from git via `git.show` at a commit captured at execution start.
- **Current tests** are read from the workspace.
- **Execution evidence** is the real exit code and captured output.

`IntegrityChecker` compares baseline against current; `adjudicate()` consults the result
**before** it looks at whether the suite passed.

---

## 3. Ordering: integrity precedes results

A green suite is only meaningful once you know the tests are still the tests. So the gate
runs first, and a tampered suite fails *even when the tests pass and even when they also
fail* — the integrity violation is the more important fact either way.

---

## 4. Detection is AST-based, not textual

`compare_test_file` extracts every test function and the `ast.dump` of each assertion
expression. A changed expected value produces a different dump and therefore appears as a
removed assertion. This distinguishes:

| Change | Verdict |
|---|---|
| `== 2` → `== 8` | HIGH — assertion altered |
| assertion deleted | HIGH — assertion removed |
| test function deleted | CRITICAL |
| test file deleted | CRITICAL |
| `@pytest.mark.skip` added | HIGH — a skipped test cannot fail |
| test added | fine |
| comment/whitespace/reformat | fine |
| assertion added | fine |

Regexes would flag reformatting and miss a changed literal inside a rewritten line. An LLM
would be unreliable in both directions. The AST is exact.

---

## 5. Deletion stays possible, but never silent

`FileEdit.removes_definitions` and `IntegrityReport.justification` let an agent declare that
it is intentionally changing a test. A declared change is permitted and *still reported*.
Forbidding test edits outright would make Edith unable to fix a genuinely wrong test; the
requirement is visibility, not prohibition.

---

## 6. Defense in depth, deliberately

The tampering scenarios are caught at whichever layer reaches them first:

1. **Task scope** — a plan naming `calc.py` yields a gateway that cannot write tests at all.
2. **Coder symbol gate** — content dropping an existing definition is rejected before the
   write. This is what actually catches test *deletion*; the integrity check never fires.
3. **Integrity gate** — baseline comparison, catching what does reach disk.
4. **Critic** — now given the integrity report as evidence.
5. **Benchmark harness** — independent, outside the system under test.

The hardening suite asserts on *outcomes* rather than on which layer caught them, so
strengthening one layer does not break the tests for another.

---

## 7. Separate principals for separate authority

Three distinct permission sets now exist, and none is a superset of another by accident:

| Principal | May write | May run commands | May read history |
|---|---|---|---|
| coder | yes (task scope) | no | no |
| verifier | no | yes | yes |
| integrity | no | no | yes |

The agent that changes the system is not the one that runs the checks, and neither is the
one that establishes what the checks used to say.

---

## 8. Capability is reported, never assumed

`StructuredMode` (`native` / `json_mode` / `fallback` / `unknown`) is now part of
`ProviderHealth` and surfaced by `edith doctor`. M2 ran an entire milestone believing
decoding was schema-constrained while Ollama was rejecting every schema containing `$ref`.

Two fixes: `inline_schema_refs` resolves `$defs` so most schemas now genuinely compile, and
when a runtime still refuses, the provider records the degradation instead of hiding it. The
doctor's probe deliberately uses a *nested* schema — a flat one would report `native` on a
runtime that rejects everything the agents actually use.

---

## 9. Environment failures are not test failures

`python` now resolves to `sys.executable`, and output matching `No module named X` is
classified `ENVIRONMENT_FAILURE` with `ran=False`. Policy escalates that to a human rather
than sending the Debugger to hunt for a bug in code that never executed.

---

## 10. Context fails closed

`ContextBundle.degraded` marks a bundle that retrieved nothing, matched nothing, or returned
almost no content, with the reason recorded. The M2 `**/*` bug silently produced empty
bundles for every task and the loop proceeded regardless; an agent asked to edit code it was
never shown will invent something plausible, and the resulting failure looks like a reasoning
problem rather than the retrieval problem it is.

---

## 11. Task verification and project verification are different questions

Task-level verification drives the per-task repair loop. The authoritative judgement is a
single final gate after every task has run, because a whole-suite check cannot be satisfied
by a task that correctly fixes one of several defects. Intermediate tasks that changed files
but left the suite red are *deferred*, not failed.
