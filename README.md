# Project Edith

A local-first, zero-API-cost autonomous product development agent platform.

Everything runs on your machine. No cloud LLM APIs, no paid services, no hidden network
calls — the model endpoint is validated to be loopback unless you explicitly opt out.

**Status: M5 complete.** See [Milestones](#milestones).

---

## Hardware target

Edith is designed for a constrained machine, not a workstation:

| Resource | Target |
|---|---|
| GPU | NVIDIA RTX 2060, 6 GB VRAM |
| CPU | Intel i7 10th gen |
| RAM | 16 GB (often < 5 GB actually free) |
| OS | Windows |
| API budget | $0 |

Every default is chosen against that budget: one inference at a time, a 3B quantized model,
8k context, and a VRAM fit check before the model is loaded.

---

## Quick start

```powershell
# 1. Install Ollama (once)
winget install Ollama.Ollama

# 2. Pull the default model (~1.9 GB)
ollama pull qwen2.5-coder:3b-instruct-q4_K_M

# 3. Set up the Python environment
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 4. Check the environment
.venv\Scripts\edith.exe doctor

# 5. Prove the kernel works end to end against the real model
.venv\Scripts\edith.exe selftest
```

---

## Commands

| Command | Purpose |
|---|---|
| `edith doctor` | Diagnose Python, Git, RAM, disk, GPU, model fit, and the model runtime. `--offline` skips the live probe; `--json` for machines. |
| `edith config` | Show the fully-resolved configuration after file + env merging. |
| `edith agents` | List registered agents with capabilities and declared permissions. |
| `edith tools` | List registered tools with their access mode and whether they spawn a process. |
| `edith tool <name>` | Invoke one tool directly with `--args` JSON. Operator-only; agents never get this path. |
| `edith run <agent>` | Invoke one agent with a JSON payload; prints a structured response. |
| `edith selftest` | End-to-end kernel check against the live model. Exits non-zero on failure. |
| `edith execute <request>` | Run the full autonomous loop: plan → implement → verify → adjudicate → repair. |
| `edith benchmark [id]` | Run a benchmark and report measured evidence. |
| `edith memory <action>` | Inspect, search, and delete stored memories. |
| `edith research <action>` | Run the research workflow; `--offline` uses the local cache only. |
| `edith environment` | Report what a project requires and what is missing. `--write` generates the manifest and install scripts (it never installs). |
| `edith strategies` | Compare every memory strategy on the same benchmark. |
| `edith budget` | Measure whether an execution memory budget keeps memory from becoming harmful. `--ablation` compares budget sizes. |
| `edith product <action>` | Drive the product pipeline: `create` a PRD, `plan` the whole chain, `status`, `agents`, `approve`. `--complete-gaps` runs a targeted pass over uncovered requirements. Produces the plan; never executes it. |
| `edith requirements` | Inspect requirements and which have acceptance criteria. |
| `edith coverage` | Show which requirements each artifact actually addresses, with the element ids as evidence. Exits non-zero on a blocking gap. |
| `edith ux` | Inspect the UX specification, flows, and screen states. |
| `edith architecture <action>` | `inspect` the design, list `decisions` (ADRs), or show the `plan`. |
| `edith experiment` | Measure memory against a no-memory control arm. |
| `edith version` | Print the version. |

Exit codes: `0` success, `1` operational failure, `2` configuration/usage error — so every
command is usable as a gate in a script.

Commands for future milestones (`project`, `task`) are deliberately absent rather than
stubbed. A command that prints "not implemented" is worse than one that does not exist.

---

## Configuration

All tunables live in `config/`. Source code never hard-codes a model name, timeout, host,
or context length.

| File | Contents |
|---|---|
| `config/system.yaml` | Project name, state dir, logging, resource thresholds |
| `config/models.yaml` | Provider, endpoint, retry policy, model profiles |
| `config/agents.yaml` | Per-agent defaults and overrides |
| `config/tools.yaml` | Workspace root, protected paths, shell allowlist, git policy |
| `config/orchestration.yaml` | Retry limits, verification profile, context budget, memory strategy and execution memory budget |

Any value can be overridden by environment variable using `EDITH__` with `__` as the
nesting separator:

```powershell
$env:EDITH__SYSTEM__LOGGING__LEVEL = "DEBUG"
$env:EDITH__MODELS__OLLAMA__TIMEOUT_SECONDS = "300"
```

Precedence: schema defaults → YAML files → environment → programmatic overrides.

### Model profiles

| Profile | Model | Est. VRAM | Notes |
|---|---|---|---|
| `default` | `qwen2.5-coder:3b-instruct-q4_K_M` | ~2.8 GB | Fits the 2060 with >3 GB headroom |
| `fast` | `qwen2.5-coder:1.5b-instruct-q4_K_M` | ~1.5 GB | Low-latency routing/classification |
| `large` | `qwen2.5-coder:7b-instruct-q4_K_M` | ~5.8 GB | **Opt-in only** — does not fit alongside a desktop |

`estimated_vram_mb` covers weights + KV cache + compute buffers + CUDA context. `edith
doctor` compares it against actually-free VRAM and warns before you load something that
will spill into system RAM.

### Memory

Memory is off by the default *strategy*, not missing. Measured over six runs per arm on
`multi_repair`, no retrieval strategy beat the no-memory control and always-inject scored
0/6 (ADR 0006), so `orchestration.memory.strategy` defaults to `none`. The subsystem is
fully built and one config line away; `edith strategies` re-runs the comparison.

When a strategy is enabled, every injection passes through the Memory Governor, which
charges it against an **execution-wide** budget (`orchestration.memory.budget`). Per-prompt
limits did not bound total cost — a repair loop retrieves again after every failure — so
the ceiling belongs to the execution. An exhausted budget injects nothing and says so; the
loop continues without memory. `edith budget` measures whether that budgeting helps.

---

## Architecture

Imports flow strictly downward. Nothing below a layer may import from above it.

```
                  cli  /  diagnostics
                          |
                       agents          contract, registry, lifecycle
                       /     \
                  models      tools    gateway, permissions, workspace
                       \     /
                       schemas         provider-neutral domain types
                          |
        config   observability   errors   system
```

### The tool gateway

Agents never hold a `Tool`. They hold a `ToolGateway` bound to their own declared
`AgentPermissions`, so there is no path from agent code to an unauthorized operation.

```
Agent -> ToolGateway -> PermissionEngine -> Tool -> Workspace -> filesystem / process
```

| Tool | Access | Notes |
|---|---|---|
| `filesystem.read` | read | UTF-8 text, optional line range |
| `filesystem.search` | read | Path glob and/or content regex |
| `filesystem.write` | write | Refuses to overwrite unless asked |
| `filesystem.patch` | read+write | Exact match; requires a unique hit by default |
| `shell.run` | read | argv list only, allowlisted executable, timeout |
| `git.status` `git.diff` `git.log` | read | Structured output |
| `git.branch` `git.commit` `git.worktree` | write | Prefix + protected-branch policy |

Three properties are structural rather than best-effort:

1. **Paths are normalized and resolved *before* authorization.** Authorizing the raw string
   is the classic traversal bug.
2. **A tool cannot construct a path except through `Workspace`.** It is handed neither the
   workspace root nor the permission set.
3. **`shell.run` takes argv, never a command string.** Command injection is not mitigated,
   it is absent — nothing ever parses a shell string.

Denials are classified `SECURITY_FAILURE`, never retried, and logged with the agent and
reason. `ToolResult.denied` distinguishes a policy refusal from an execution failure.

### The model seam

`ModelProvider` is the only place that knows an LLM exists:

```
ModelProvider
├── generate()             free-form text
├── stream()               incremental chunks
├── structured_generate()  validated Pydantic object, with bounded repair
├── supports_tools()
└── health_check()         structured, never raises for expected failures
```

`structured_generate()` is implemented once on the base class, so every provider gets the
same JSON extraction, schema validation, and repair loop. A model *claiming* its output is
correct never counts — only successful Pydantic validation returns.

### The agent contract

`Agent.execute()` is `final` in spirit: subclasses implement `_run()`, and the base class
owns validate-input → run → validate-output → structured response. An agent therefore
cannot emit unvalidated output or swallow a failure, because it does not own that code path.

Every outcome is an `AgentResponse` with an explicit status:

- `SUCCESS` — output validated against the agent's `output_schema`
- `REJECTED` — input or output failed validation (retrying will not help)
- `FAILURE` — execution failed, with a `FailureCategory`

`execute()` never raises.

---

## Development

```powershell
.venv\Scripts\python.exe -m pytest -m "not integration"   # hermetic unit tests
.venv\Scripts\python.exe -m pytest -m integration         # requires Ollama + the model
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m mypy
```

Unit tests never touch a real model runtime — HTTP is mocked with `respx` and the agent
layers run against a `FakeProvider`. That a fake can substitute for `OllamaProvider` with no
other change is the practical proof that the abstraction holds.

Integration tests **skip** (visibly) when the runtime or model is absent. A skip is never
reported as a pass.

---

## Milestones

- **M0 — Local Agent Kernel** ✅ config, logging, schemas, model provider abstraction,
  Ollama provider, agent contract, registry, CLI, doctor, test infrastructure
- **M1 — Tool Kernel** ✅ tool gateway, permission engine, path policy, filesystem/shell/git
  tools, audit logging
- **M2 — Core Autonomous Loop** ✅ orchestrator, planner, context engine, coder, verification
  runner, critic, debugger, bounded retries, state persistence
- **M2.1 — Verification Hardening** ✅ AST test-integrity gate, environment classification,
  multi-task verification semantics, structured-output capability reporting
- **M3 — Memory + Research** ✅ SQLite memory with provenance and project isolation,
  relevance-based retrieval, research subsystem, measurable memory experiment
- **M3.1 — Targeted Memory + Environment** ✅ memory strategies measured rather than assumed,
  four-way failure classification, dependency discovery, reproducible install artifacts
- **M3.2 — Memory Execution Budget** ✅ execution-wide memory budget, central governor with
  no bypass path, fail-closed exhaustion, duplicate suppression, durable context accounting
- **M4 — Product Development Layer** ✅ versioned artifacts, PRD/UX/architecture agents,
  executable authority hierarchy, deterministic contradiction detection, implementation plan
- **M4.1 — Stage Decomposition** ✅ large structured generations split into small validated
  stages with deterministic assembly and partial-failure isolation. Measured on the live 3B
  model: monolithic UX 0/5, decomposed 5/5. See
  [experiment 0001](docs/experiments/0001-stage-decomposition.md).
- **M4.2 — Requirement Coverage** ✅ evidence-based coverage states, structured gaps, an
  explicit threshold, critical gaps blocking approval, and targeted completion that closes a
  gap without regenerating the artifact. See
  [experiment 0002](docs/experiments/0002-targeted-completion.md).
- **M5 — Engineering Execution Layer** ✅ five specialised agents (frontend, backend,
  database, devops, dependency) with structurally enforced scopes, task ownership, cross-agent
  conflict detection, an importability gate, bounded repair, and a live application benchmark
- **M6** — quality team · **M7** — full workflow · **M8** — benchmark + hardening

See `claude.md` for the full specification and engineering invariants.
