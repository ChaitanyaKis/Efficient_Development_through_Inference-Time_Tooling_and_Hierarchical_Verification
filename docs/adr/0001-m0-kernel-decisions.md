# ADR 0001 — M0 Local Agent Kernel decisions

- **Status:** Accepted
- **Date:** 2026-08-11
- **Milestone:** M0

This records the decisions made while building the kernel, so later milestones inherit the
reasoning rather than rediscovering it.

---

## 1. Default model: `qwen2.5-coder:3b-instruct-q4_K_M`

**Context.** The target GPU is an RTX 2060 with 6144 MiB total and ~5955 MiB free. System
RAM is 16 GB but frequently under 5 GB actually free, so anything that does not fit in VRAM
degrades badly — it spills to a system that has no headroom either.

**Decision.** Default to the 3B Q4_K_M coder model at 8k context.

**Alternatives.**

| Option | Estimated VRAM | Verdict |
|---|---|---|
| 7B Q4_K_M @ 8k | ~4.7 GB weights + ~0.47 GB KV + ~0.3 GB buffers + ~0.3 GB CUDA ≈ **5.8 GB** | Rejected — within noise of 5.955 GB free; spills the moment the desktop compositor takes VRAM |
| **3B Q4_K_M @ 8k** | ~1.9 + ~0.30 + ~0.25 + ~0.3 ≈ **2.8 GB** | **Chosen** — >3 GB headroom |
| 1.5B Q4_K_M @ 4k | ≈ 1.5 GB | Kept as the `fast` profile for routing/classification |

**Rationale.** For the kernel, reliable JSON-schema conformance matters more than raw
reasoning depth; the 3B model holds structured output well and leaves room to grow context
later. The 7B profile is retained in `models.yaml` as opt-in so the tradeoff stays visible
rather than being quietly forgotten.

**Consequences.** Reasoning-heavy agents in M2+ may find 3B limiting. The mitigation is
already in place: profiles are per-agent, so a future Planner or Critic can request `large`
and `edith doctor --profile large` will report the VRAM cost honestly.

---

## 2. Structured generation lives on the base class, not per provider

**Context.** Agents must receive validated objects, never text. Ollama supports constrained
decoding via a JSON Schema in `format`; other runtimes may not.

**Decision.** `ModelProvider.structured_generate()` is concrete on the abstract base,
implemented in terms of the abstract `_generate_raw()`. It passes the schema down as a hint,
then independently extracts JSON, validates with Pydantic, and runs a **bounded** repair
loop feeding the validation error back to the model.

**Rationale.** If each provider implemented this, each would get the edge cases subtly
wrong. More importantly, native constrained decoding is a *performance* optimisation, not a
correctness guarantee — local validation is the actual gate. A model claiming its output is
correct never counts.

**Consequences.** A new provider implements four small methods and inherits correct
structured generation. The repair loop costs at most `max_repair_attempts` extra inferences,
which is bounded and configurable.

---

## 3. The agent lifecycle is owned by the base class

**Context.** CLAUDE.md invariants: agents must not claim success without evidence, must not
hide failures, and must have explicit validated I/O.

**Decision.** `Agent.execute()` is fixed: bind trace context → validate input → run with
bounded retry → validate output → return a structured `AgentResponse`. Subclasses implement
only `_run()`.

**Rationale.** Making this a *convention* would mean every future agent author can violate
it. Making it structural means they cannot: a subclass does not own the code path that
builds the response, so it cannot emit unvalidated output or swallow an exception.
`execute()` never raises.

**Consequences.** Agents needing a genuinely different lifecycle must override `execute()`
explicitly and visibly, which is exactly when it should be reviewed.

---

## 4. Sequential inference by default

**Decision.** `max_concurrent_inferences: 1`. The provider is synchronous; no async layer.

**Rationale.** Two concurrent generations on a 6 GB GPU either OOM or force layer offload.
CLAUDE.md also forbids unnecessary async complexity. Concurrency can be introduced later
behind the config value if resource detection ever justifies it.

---

## 5. Loopback-only endpoint, enforced by validation

**Decision.** `OllamaProviderConfig` rejects a non-loopback host unless `allow_remote: true`
is set explicitly.

**Rationale.** "No hidden cloud dependency" is an invariant, and a config typo should not be
able to ship prompts containing proprietary source code to a third party. Making the escape
hatch explicit means enabling it is a reviewable act.

---

## 6. Scope discipline: no stubs for future milestones

**Decision.** M0 ships exactly one agent (`echo`, the kernel canary) and only the CLI
commands the kernel can actually support. No placeholder Planner, Coder, or Critic; no
`project`/`memory`/`task` commands.

**Rationale.** CLAUDE.md invariant 16 — do not over-engineer infrastructure before
validating the core loop. A stub that prints "not implemented" is a liability: it looks like
progress, invites premature coupling, and has to be deleted later.

**Consequences.** The kernel is small enough to be fully tested, and M1/M2 extend it by
addition rather than by rewriting speculative interfaces.
