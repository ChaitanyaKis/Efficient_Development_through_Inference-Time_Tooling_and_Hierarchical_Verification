# Edith benchmarks

Deterministic tasks used to measure whether Edith actually works, run against the real
local model. Each benchmark is a small, self-contained repository plus a request and
machine-checkable acceptance criteria.

The benchmark repositories live in `benchmarks/fixtures/` as *templates*. A run copies a
template into a scratch workspace, initializes git there, and executes against the copy, so
a run never mutates the fixture and every run starts from an identical state.

## Benchmarks

| Id | Scenario | What it proves |
|---|---|---|
| `feature` | `calculator` is missing `multiply`; a test for it exists and fails | Plan → context → code → test → judge on a real request |
| `repair` | `subtract` has a seeded bug (`a + b`); its test fails | Detect → classify → debug → fix → re-verify → PASS |

The `repair` benchmark is the important one. Any system can look successful when the first
attempt works; the question M2 has to answer is whether Edith recovers from a failure it
did not expect.

**The seeded defect** lives in `calculator_bug/calculator.py`: `subtract` returns `a + b`.
It is documented *here* rather than in the file, because the agent reads that file. An
earlier version carried a "do not fix this, it is deliberate" comment for human
maintainers; the model read it, obeyed it, and declined to repair the bug through three
attempts. Never annotate a fixture with instructions you do not want an agent to follow.

## Running

```powershell
edith benchmark --list
edith benchmark feature
edith benchmark repair
edith benchmark --all --json
```

Requires a healthy Ollama and the configured model (`edith doctor`). These call a real
model, so they are slow and not part of the unit suite. The `benchmark` pytest marker runs
them from pytest:

```powershell
pytest -m benchmark -v
```

## Acceptance

A benchmark passes only when **all** of the following hold, checked against the workspace
rather than reported by an agent:

1. The execution reached `RELEASE` with verdict `PASS`.
2. The benchmark's own verification command passes in the resulting workspace.
3. Every file the benchmark declares as protected is byte-identical to the fixture.

Point 3 exists so a "successful" run that achieved its goal by deleting the failing test is
recorded as the failure it is.
