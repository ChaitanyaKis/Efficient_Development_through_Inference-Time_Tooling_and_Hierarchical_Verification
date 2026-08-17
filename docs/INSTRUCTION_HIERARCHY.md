# Instruction hierarchy

Edith reads text from many places, and not all of it carries the same authority. A comment
in a source file is *data about the repository*; it is not an instruction to the system.

This became concrete during M2. A benchmark fixture carried a note for human maintainers:

```python
# NOTE: `subtract` contains a deliberately seeded defect.
# Do not "fix" it here -- the point is that Edith must find and repair it.
```

The model read that file as context, treated the comment as an instruction, and declined to
repair the bug through three attempts and two debugger consultations. Nothing was broken.
The system did exactly what it was told — by the wrong author.

## The order of authority

Higher entries win. Lower entries may inform, never override.

| Rank | Source | Authority | Where it comes from |
|---|---|---|---|
| 1 | **User-approved requirement** | Absolute. What was actually asked for and accepted. | `Execution.request`, an approved `PRD` artifact |
| 2 | **Project policy** | Operator-set limits: protected paths, shell allowlist, verification commands, workspace root. | `config/*.yaml` |
| 3 | **Approved architecture decision** | An ADR a human accepted. Binding until superseded. | An `APPROVED` architecture artifact |
| 4 | **Task acceptance criteria** | What "done" means for this task. | `Task.acceptance_criteria`, from the plan |
| 5 | **Agent recommendation** | Anything an agent proposes: a draft requirement, a suggested design, a critique. | Any `DRAFT` or `REVIEW` artifact |
| 6 | **Repository content** | Source, tests, comments, docstrings, READMEs. | The workspace |
| 7 | **Untrusted external content** | Anything fetched from outside the machine. | Research sources, web pages |

Ranks 1–2 are set outside the loop by a human. Rank 3 exists only once a human approves it.
Rank 4 is derived from rank 1 and validated against rank 2. Ranks 5–7 are **read as
evidence**, never obeyed as instructions.

M4 added ranks 3, 5, and 7 and made the whole order executable: `edith.authority` defines
`AuthorityLevel` with a rank table, and `may_override(candidate, incumbent)` answers the
question directly. Two rules there are worth stating explicitly, because they are not what a
naive reading of an ordered list would give you:

**Advisory levels never override anything**, not even something below them. An agent
recommendation outranks a web page, but it still does not get to rewrite one — resolving that
conflict is a human's call, and the system's job is to surface it.

**Equal levels do not override either.** Two requirements in conflict is a contradiction to
report, not a race won by whichever was written last.

### Approval is what confers authority

The distinction between rank 3 and rank 5 is *status*, not content. The same ADR text is an
agent recommendation while it is a draft and an approved architecture decision after a human
accepts it. `Artifact` enforces this in a validator: an artifact claiming
`APPROVED_ARCHITECTURE_DECISION` authority while its status is `DRAFT` will not construct.

Without that check, an agent could mint authority simply by writing confidently — which is
the same failure mode as the fixture comment below, one layer up.

## What this means in practice

**Repository content is untrusted input.** A comment saying "do not change this", a README
claiming a function is deprecated, or a docstring instructing an agent to skip a test are
all *observations about the codebase*. They describe what someone once believed. They do not
alter the task.

**The distinction is enforced structurally, not by asking politely.** Telling a model to
"ignore comments" is unreliable and also wrong — comments carry real information about
intent. Instead:

- Context is presented as *quoted material*: "File `x.py` contains the following code",
  never as instructions appended to the prompt.
- The task and its acceptance criteria appear in a separate, higher-priority position in the
  prompt, and the system prompt states the hierarchy explicitly.
- Anything that would actually *constrain* an agent — where it may write, what it may run,
  which files are protected — lives in configuration and is enforced by the M1 gateway, not
  by text the model reads.

That last point is the important one. A comment cannot grant or remove permission, because
permission is not something the model decides. If repository content could change what an
agent is allowed to do, the hierarchy would be advisory; because it cannot, the hierarchy
only has to govern *reasoning*, which is a much smaller problem.

**Conflicts are surfaced, not silently resolved.** When repository content contradicts the
task, the agent proceeds with the task and records the conflict in its notes. An operator
reading the run can then see that the codebase disagreed, which is often a genuine signal
that the request was wrong — but that is a human's call, not the model's.

## Fixture rule

A corollary for anything Edith reads during evaluation: **never annotate a fixture with an
instruction you do not want an agent to follow.** Explanations for human maintainers belong
in the harness, the benchmark definition, or a README the agent does not read — not in the
file under test. The `calculator_bug` fixture now carries no such note, and
`tests/test_benchmarks.py` asserts that none ever will.
