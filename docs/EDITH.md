# EDITH — Architecture, Security Model, and Evidence

This is the reference document for EDITH: what it is, how it is built, what has been measured,
and what has not. It is written to be checkable. Every capability claim below is labelled with
the evidence behind it, using five levels:

| Label | Meaning |
|---|---|
| **PROVEN** | Enforced by a mechanism and covered by a deterministic test that fails if the guarantee breaks |
| **MEASURED** | Observed in a live experiment with a stated sample size |
| **SUPPORTED** | A hypothesis the evidence favours, with the sample small enough that it could still be wrong |
| **NOT SUPPORTED** | A hypothesis that was tested and rejected |
| **UNKNOWN** | Not established. Named rather than assumed |

Nothing here is marketing. Where a mechanism exists but did not help, that is recorded as
plainly as the things that worked.

---

## 1. What EDITH is

A local-first, zero-API-cost autonomous product development platform. It takes a requirement,
plans work, writes code through a permission-bounded tool gateway, verifies it independently,
repairs what is genuinely repairable, and merges only what passed.

It runs entirely on one machine. No cloud LLM APIs, no paid services. The reference model is
`qwen2.5-coder:3b-instruct-q4_K_M` on a 6 GB GPU.

The governing principle, from `CLAUDE.md`:

> The model proposes. Tools execute. Tests verify. Critics challenge. Git records. The
> orchestrator decides.

---

## 2. Architecture inventory

| Subsystem | Purpose | Security boundary | Deterministic? | Default |
|---|---|---|---|---|
| Config (`config/`) | Typed, validated settings | Rejects unknown keys; bounds every budget | Yes | on |
| Model provider (`models/`) | The only seam to an LLM runtime | No filesystem or shell access | No | on |
| Tool gateway (`tools/gateway.py`) | Single route from agent to effect | Enforces every permission | Yes | on |
| Path policy (`tools/paths.py`) | Resolves and confines every path | Traversal, absolute, symlink escape | Yes | on |
| Permissions (`tools/permissions.py`) | Per-principal tool and path grants | The boundary itself | Yes | on |
| Process runner (`tools/process.py`) | argv-only execution, bounded output | `shell=False`, allowlisted argv[0], timeout, process-tree kill | Yes | on |
| Git tools (`tools/git.py`) | status, diff, log, commit, worktree | Only the executor principal holds mutation | Yes | on |
| Memory (`memory/`) | Three-layer recall over SQLite | Per-project isolation, budget governor | Mixed | **off** |
| Product layer (`product/`) | PRD, UX, architecture, coverage | Artifact authority, system-owned IDs | Mixed | on |
| Planning (`planning/`) | Task DAG, dependency order | Cycle detection | Yes | on |
| Engineering executor (`engineering/executor.py`) | Runs the plan | Task-scoped gateway per task | Yes | on |
| Workspace isolation (`engineering/isolation.py`) | One git worktree per task | Absolute roots, fail-closed, merge containment | Yes | on |
| Verification (`verification/runner.py`) | Runs the configured checks | Separate principal from the coder | Yes | on |
| Quality pipeline (`quality/`) | Deterministic gates, then adjudication | Model findings cannot self-authorise | Mixed | gates on |
| Security scanner (`quality/scanners.py`) | AST checks for injection, secrets, traversal | Blocking on CRITICAL | Yes | on |
| Model reviewers (`quality/agents.py`) | Advisory review, Judge | Read-only; no verdict authority | No | **off** |
| Test generation (`quality/testgen.py`, `testgate.py`) | Requirement-derived tests | Writes only `tests/generated/**` | Mixed | **off** |
| Boundary analysis (`requirements/boundaries.py`) | Reads thresholds out of requirement text | Imports no model, no judge, no tests | Yes | optional |
| CLI (`cli/`) | Human entry point | Holds `UNRESTRICTED`; never given to an agent | Yes | on |
| Observability (`observability/`) | Structured logs with secret redaction | Redactor on every sink | Yes | on |
| Benchmarks (`benchmarks/`) | Evaluation, never imported by production | Out of the runtime path | Yes | n/a |

**Principals.** `CODER` writes implementation. `TESTER` writes `tests/**`. `TESTGEN` writes
only `tests/generated/**` and holds no shell. `VERIFIER` runs checks and writes nothing.
`SECURITY` and `REVIEWER` read only. `JUDGE` holds read alone — no shell, no writes, no git.
No principal is a permission superset of another except one documented reduction
(`TESTGEN ⊂ TESTER`), which is asserted explicitly. **PROVEN.**

---

## 3. Security model

The boundary is the M1 tool gateway. An agent has no `subprocess`, no `open()`, no `pathlib`
writes, and no git of its own — its only capability is the gateway it was handed, bound to its
task's scope.

| Property | Status |
|---|---|
| Path traversal refused | **PROVEN** |
| Absolute paths outside the workspace refused | **PROVEN** |
| Relative task root refused (fail-closed) | **PROVEN** |
| Workspace isolation per task | **PROVEN** |
| Main workspace unchanged by a rejected task | **PROVEN** |
| Merge copies only declared files, re-checked inside the root | **PROVEN** |
| Out-of-scope write is terminal `SECURITY_FAILURE`, never repaired | **PROVEN** |
| Secrets redacted in logs | **PROVEN** |
| Model cannot assign itself authority | **PROVEN** — the fields do not exist on its schemas |
| Model cannot fabricate deterministic evidence | **PROVEN** — origin is overwritten at the gate |
| Judge cannot override a deterministic block | **PROVEN** |
| Fabricated citations discarded | **PROVEN** — quotes are checked against the source |
| Acceptance tests cannot influence implementation | **PROVEN** — separate process, after merge |
| Generated tests cannot become authoritative | **PROVEN** — provenance is system-assigned |
| Unknown configuration keys rejected | **PROVEN** |
| Budgets, retries and timeouts bounded | **PROVEN** |

**The `UNRESTRICTED` permission set exists** for the human CLI, where restricting a user who
already has a shell would be theatre. It is never handed to an agent — verified by inspection:
its only uses are in `cli/main.py`.

---

## 4. Execution lifecycle

```
plan
 → boundary validation (optional)
 → task DAG, dependency order
 → isolated git worktree per task
 → agent implementation via scoped gateway
 → import/build gate
 → configured tests, in the task workspace
 → test-integrity comparison (AST)
 → deterministic security scan
 → model review (optional, advisory)
 → deterministic adjudication
 → repair, only if genuinely repairable
 → merge, only if verified
 → independent acceptance
```

No step may skip a stronger authoritative step. The verdict is produced by a pure function over
the collected findings, never by a model.

---

## 5. Failure taxonomy and repair policy

Only these consume the coder's repair budget:

| Repairable | Not repairable |
|---|---|
| `CODE_FAILURE` | `TIMEOUT`, `ENVIRONMENT_FAILURE`, `DEPENDENCY_FAILURE` |
| `TEST_FAILURE` | `SECURITY_FAILURE`, `CONFIGURATION_ERROR`, `TOOL_ERROR` |
| `BUILD_ERROR` | `MODEL_ERROR`, `REQUIREMENT_FAILURE`, `ARCHITECTURE_FAILURE`, `UNKNOWN` |
| `VALIDATION_FAILURE` | |

**Why it matters.** Spending budget on an environment fault both wastes the attempt and
misattributes the failure to the agent's work. The mirror case matters equally: a refused edit
carrying an actionable reason *is* repairable, and denying it wasted a task that one round
would fix (M11 → M12).

A refused edit enters repair only when it is inside the agent's authority, has structured
evidence, involved no policy denial, and the budget permits. A set containing even one denial
is not partly repairable. **PROVEN.**

**Naming gap (UNKNOWN):** the taxonomy has no distinct `MERGE_FAILURE` or `WORKSPACE_FAILURE`;
both surface as `TOOL_ERROR`. Behaviour is correct (neither is repairable); only the label is
coarse.

---

## 6. Research findings

Every hypothesis tested from M3 onward, including the ones that failed.

| # | Hypothesis | Experiment | Result | Verdict |
|---|---|---|---|---|
| 1 | Memory injection improves output | 4 batches, 3 strategies | `none` best 4/6; `always` **0/12** | **NOT SUPPORTED** |
| 2 | A memory budget improves success | M3.2 A/B/C | Bounds cost (4,626→1,775 peak); no success change | **NOT SUPPORTED** (cost only) |
| 3 | Monolithic artifact generation works | M4.1, 10 trials | Monolithic 0/10; decomposed 10/10, also faster | **SUPPORTED** (decompose) |
| 4 | Targeted completion closes coverage gaps | M4.2 | 0.667 → 1.000 for one extra call | **SUPPORTED** |
| 5 | Specialisation beats a generic coder | M5.1 fair rerun | 10/10 vs 10/10 complete; 5/5 vs 0/5 runnable | **SUPPORTED**, residual prompt confound |
| 6 | Model quality review improves acceptance | M6.1, M6.2, M7 — three runs | 0 findings in 36 runs; +41% runtime | **NOT SUPPORTED** |
| 7 | Requirement-derived tests reduce false PASS | M8, 72 runs | False PASS 3→0, but blocked 32/36 | **NOT SUPPORTED** |
| 8 | A scaffold gate makes them usable | M9, 72 runs | Harm removed; false PASS 3→3 | **NOT SUPPORTED** |
| 9 | Boundary analysis removes boundary defects | M10, 72 runs | 30/36→33/36; false PASS 3→0; 0 false blocks | **SUPPORTED** |
| 10 | The remaining failures were semantic | M11, 15 runs | 0 semantic failures; a tool-contract error | **NOT SUPPORTED** — the premise was wrong |
| 11 | Actionable rejections should be repairable | M12, 5+72 runs | SEM-003 0/5→5/5; benchmark 33/36→36/36 | **SUPPORTED** |

**The pattern worth stating:** every mechanism that asked a 3B model to check another 3B
model's work failed to help. Every mechanism that read the requirement or the evidence
deterministically did help. Four generations of model-based verification were beaten by a
regex-and-AST reader of requirement text.

---

## 7. Configuration and defaults

Defaults are measurements, not caution:

```yaml
memory:                   off     # helped nothing; `always` scored 0/12
model_quality_review:     false   # 0 findings across 36 runs, +41% runtime
requirement_derived_testing: off  # blocked 32/36 ungated; no gain gated
boundary_analysis:        optional, supported by M10
max_repair_attempts:      2       # bounded, validated
```

Invalid configuration fails closed: unknown keys are rejected, budgets and timeouts are
range-validated, and no experimental feature becomes default merely because it exists.

---

## 8. Model support

Model identity is configuration. `build_provider(config, profile)` returns a `ModelProvider`;
**no agent imports a concrete provider** — verified by inspection. Swapping the local model is
a config change, and a larger model receives no additional authority: the gateway, principals,
verification and adjudication are identical regardless of what is behind the seam.

Tested with one model only (`qwen2.5-coder:3b-instruct-q4_K_M`). Behaviour with any other model
is **UNKNOWN**.

---

## 9. Reproducibility

```bash
# 1. prerequisites: Python 3.13, git, Ollama
ollama pull qwen2.5-coder:3b-instruct-q4_K_M

# 2. install
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"

# 3. check the environment
.venv/Scripts/python -m edith doctor

# 4. tests
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check src tests benchmarks
.venv/Scripts/python -m mypy --strict src

# 5. benchmarks (each prints its own summary)
.venv/Scripts/python benchmarks/run_production.py 3   # independent, M13
.venv/Scripts/python benchmarks/run_boundary.py 3     # M10 boundary A/B
.venv/Scripts/python benchmarks/run_gated.py 3        # M9 scaffold gate
```

Benchmarks need Ollama running with the model loaded. Each writes its own temporary workspaces
and cleans up after itself; none depends on undocumented local state.

---

## 10. Known limitations

- **One model, one machine.** Every number comes from `qwen2.5-coder:3b` on a 6 GB RTX 2060.
- **Small samples.** The largest benchmark is 72 runs. No statistical significance is claimed
  anywhere in this document.
- **Python only.** The syntax gate, import gate, symbol-preservation check and boundary case
  expansion are Python-specific. Other languages fall through unchecked.
- **Single-file tasks dominate.** Multi-file coordination is exercised by one benchmark task.
- **Unexercised agents.** Frontend, DevOps, Dependency, UX and Architect roles exist and are
  unit-tested but are not covered by an end-to-end benchmark.
- **Mode selection is unstable and unexplained (UNKNOWN).** The coder sometimes chooses an edit
  mode that cannot apply. Repair recovers it every observed time; the cause is not isolated.
- **Sequential execution.** One inference at a time, by hardware constraint.
- **No performance tuning.** Runtimes are recorded, not optimised.

---

## 11. What EDITH can do (evidence-backed)

- Take a requirement, plan it, implement it in an isolated git worktree, verify it, repair a
  genuinely repairable failure, and merge only verified work. **MEASURED**
- Refuse every write outside an agent's scope, at the policy layer rather than by prompt.
  **PROVEN**
- Detect numeric thresholds in requirement prose, state the operator and the neighbouring
  cases, and refuse to guess when the wording is ambiguous. **PROVEN + SUPPORTED**
- Detect command injection, hardcoded credentials, unsafe deserialisation, path traversal and
  credential logging by AST, and block on them. **PROVEN**
- Prevent a model — reviewer or judge — from approving its own output or overriding a
  deterministic block. **PROVEN**
- Survive process restart with durable state. **PROVEN**

## 12. What EDITH cannot guarantee

- That generated code is correct. Verification is evidence, not proof.
- That a passing task meets an unstated requirement.
- Any behaviour with a model other than the one tested.
- Correctness for non-Python targets.
- Detection of semantic defects with no lexical or executable signal.
- That a requirement's *intent* was understood — only that stated conditions were satisfied.
- Performance characteristics under load, concurrency, or large repositories.

---

## 13. Independent benchmark (M13)

Twelve tasks written after M10–M12 and never used to tune them. 3 trials × 12 tasks × 2 arms
= 72 runs. Arm A is production defaults; arm B adds M10 boundary analysis.

| Metric | A (defaults) | B (+ boundary) |
|---|---:|---:|
| runs | 36 | 36 |
| completed | 33 | 33 |
| **accepted** | **24** | **24** |
| false PASS (excl. ambiguous) | 6 | 6 |
| false PASS (incl. ambiguous) | 9 | 9 |
| repairs | 4 | 6 |
| security failures | 0 | 0 |
| merge failures | 0 | 0 |
| boundaries stated | 0 | 9 |
| mean / worst runtime | 4.5s / 12.9s | 4.4s / 6.0s |

By area, both arms identical: crud 3/3, database 3/3, auth 3/3, file_io 3/3, business_rule 3/3,
boundary 3/3, multi_file 3/3, dependency 3/3 — and **api 0/3, transform 0/3, error_handling
0/3, ambiguous 0/3**.

**This is the most important number in the document.** On the benchmark that shaped M10–M12,
EDITH reaches 36/36. On tasks it has never seen, it reaches **24/36**. The difference is the
cost of having tuned against one benchmark, and it is why the tuned figure must never be quoted
on its own.

### What each failure actually was

| Task | Behaviour | Classification |
|---|---|---|
| PRD-002 (api) | Generated code did not import; repaired twice; refused | **Correct rejection** — the import gate worked. Not a false pass |
| PRD-008 (transform) | `slugify('a  --  b')` → `'a------b'`, runs not collapsed | Genuine semantic failure |
| PRD-009 (error handling) | Raised on `k=a=b` instead of splitting on the first `=` | Genuine semantic failure |
| PRD-012 (ambiguous) | `shorten('abcdefghij', 5)` → 8 characters | Violated the stated limit |

**On PRD-012 and the ambiguity exclusion.** It is labelled ambiguous because the requirement
does not say whether to add an ellipsis. It does say the result must fit within the limit, and
the implementation broke that — so counting it as ambiguity is generous to EDITH. The honest
false-pass figure is between 6 and 9 of 36, and the higher number is the defensible one.

### The finding this overturns

M11 concluded, from the tuned benchmark, that no semantic failures remained. That was true of
those twelve tasks and **false in general**: PRD-008 and PRD-009 are exactly the wrong-operation
and wrong-condition failures M11 looked for and did not find. Boundary analysis (M10) does not
help here — it fires on nine boundaries and changes nothing, because these defects have no
numeric threshold to read.

**Semantic correctness on unseen tasks is the open problem.** No mechanism built across M6–M12
addresses it, and four of them were measured not to.
