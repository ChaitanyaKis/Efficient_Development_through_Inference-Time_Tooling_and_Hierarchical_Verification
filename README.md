# Project Edith

A local-first, zero-API-cost autonomous product development agent platform.

Everything runs on your machine. No cloud LLM APIs, no paid services, no hidden network
calls — the model endpoint is validated to be loopback unless you explicitly opt out.

**Status: M0 (Local Agent Kernel) complete.** See [Milestones](#milestones).

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
| `edith run <agent>` | Invoke one agent with a JSON payload; prints a structured response. |
| `edith selftest` | End-to-end kernel check against the live model. Exits non-zero on failure. |
| `edith version` | Print the version. |

Exit codes: `0` success, `1` operational failure, `2` configuration/usage error — so every
command is usable as a gate in a script.

Commands for future milestones (`project`, `memory`, `task`, `benchmark`) are deliberately
absent rather than stubbed. A command that prints "not implemented" is worse than one that
does not exist.

---

## Configuration

All tunables live in `config/`. Source code never hard-codes a model name, timeout, host,
or context length.

| File | Contents |
|---|---|
| `config/system.yaml` | Project name, state dir, logging, resource thresholds |
| `config/models.yaml` | Provider, endpoint, retry policy, model profiles |
| `config/agents.yaml` | Per-agent defaults and overrides |

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

---

## Architecture

Imports flow strictly downward. Nothing below a layer may import from above it.

```
                  cli  /  diagnostics
                          |
                       agents          contract, registry, lifecycle
                          |
                       models          ModelProvider -> OllamaProvider
                          |
                       schemas         provider-neutral domain types
                          |
        config   observability   errors   system
```

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
- **M1 — Tool Kernel** — filesystem/shell/Git tools, permission engine, tool gateway
- **M2 — Core Autonomous Loop** — orchestrator, planner, context, coder, testing, critic, debugger
- **M3** — memory + research · **M4** — PM/UX/architect · **M5** — engineering team
- **M6** — quality team · **M7** — full workflow · **M8** — benchmark + hardening

See `claude.md` for the full specification and engineering invariants.
