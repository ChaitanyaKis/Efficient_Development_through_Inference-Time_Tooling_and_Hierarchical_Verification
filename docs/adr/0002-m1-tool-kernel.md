# ADR 0002 — M1 Tool Kernel decisions

- **Status:** Accepted
- **Date:** 2026-08-11
- **Milestone:** M1

M1 gives agents their first contact with the outside world. Every decision here is a
security decision.

---

## 1. Capabilities, not argument scanning

**Context.** The gateway must ensure a tool only touches paths the calling agent is allowed
to touch.

**Rejected.** Have the gateway inspect a tool's arguments for path-shaped fields and
authorize those. This fails for tools with a variable number of paths
(`filesystem.search` walks a whole tree), and it depends on every future tool author
remembering to annotate their fields. A forgotten annotation is a silent hole.

**Decision.** A tool receives a `Workspace` already bound to the caller's permissions. The
*only* way to obtain an absolute path is `resolve_read()` / `resolve_write()`, each of
which normalizes, resolves, containment-checks, and authorizes before returning.

**Consequence.** A tool author who forgets to authorize simply has no path to open. The
tool is handed neither the workspace root nor the permission set, so it cannot widen its
own scope. Security is structural rather than conventional.

---

## 2. Normalize and resolve *before* authorizing

**Decision.** `PathPolicy.resolve()` runs syntactic rejection → join → `resolve()` →
containment → symlink check → protected-list check, in that order. Authorization happens
afterwards, on the resolved relative path.

**Rationale.** Authorizing the raw string is the classic traversal bug: `src/../../etc/passwd`
passes a `startswith("src/")` check and then resolves somewhere else entirely.

**Consequence.** Order is load-bearing and documented as such in the module docstring.

---

## 3. `shell.run` takes argv, never a command string

**Decision.** The input schema has `argv: list[str]`. There is no `command: str` field, and
`shell=False` always.

**Rationale.** This does not *mitigate* command injection, it removes the category. There is
no string for `;`, `&&`, backticks, or `$(...)` to be interpreted in, because nothing ever
parses one. Escaping and blocklists are the alternatives, and both are historically leaky.

**Consequence.** Agents cannot use pipes or redirection. That is acceptable: an agent
needing to chain commands issues two tool calls, which is also more auditable.

---

## 4. Environment built from an allowlist

**Decision.** The child environment is constructed from scratch from a configured allowlist,
plus `PATH`.

**Rationale.** The developer's shell almost certainly holds credentials. Inheriting the
parent environment would hand every one of them to any program an agent runs.

---

## 5. Curated git surface, not a `git` passthrough

**Decision.** Eleven specific tools rather than a generic `git <args>` tool. Refs and branch
names are validated against a conservative pattern, path lists are always preceded by `--`,
new branches must carry the `agent/` prefix, and protected branches cannot be deleted.

**Rationale.** A passthrough gives agents `reset --hard`, `push --force`, and `clean -fdx`.
A ref beginning with `-` is read by git as an option, so `--upload-pack=...` in a ref field
is remote code execution. Neither risk is worth the convenience.

**Consequence.** New git capabilities require a new tool. That is the intended friction.

---

## 6. Windows reparse points are the real escape vector

**Context.** A symlink check is the obvious containment guard. On Windows it is not enough.

**Finding, verified on the target machine.** A directory **junction** requires no elevation
to create, `Path.is_symlink()` returns `False` for it, and `Path.resolve()` follows it out
of the tree. A symlink-only guard therefore misses the easiest escape available to an
unprivileged attacker.

**Decision.** Containment is checked against the **resolved** path everywhere, and
`PathPolicy.contains()` is the single helper that does it. `filesystem.search` re-checks
containment per discovered entry, because it walks the tree itself rather than resolving one
caller-supplied path.

**Consequence.** Junction-escape tests run for real on Windows rather than being skipped;
they caught an actual escape in `filesystem.search` during M1 development, where a junction
let content search read files outside the workspace.

---

## 7. Denials are security events, not errors

**Decision.** `PermissionDeniedError` is classified `SECURITY_FAILURE`, is never retryable,
and is logged at WARNING with the reason and the agent name. `ToolResult.denied`
distinguishes a policy refusal from an execution failure.

**Rationale.** "Agent tried to read `.env`" and "file not found" are different events. Only
one of them should draw attention during audit review.

---

## 8. Search omits what it cannot show

**Decision.** Files outside the agent's read scope are silently absent from search results
rather than reported as forbidden.

**Rationale.** "You may not read `secrets/prod.key`" confirms the file exists. Absence
leaks nothing. The search *base directory* is exempt from the scope check (it is only a
traversal root, and every returned file is still filtered), otherwise an agent scoped to
`src/**` could not search at all.

---

## 9. Non-zero exit is data, not an exception

**Decision.** `shell.run` returns `ok=True` with `exit_code=1` when a command runs and
fails. The tool call itself only fails if the command could not be run.

**Rationale.** A failing test suite is exactly the evidence the Testing and Debugging agents
(M2) exist to consume. Raising would force every caller to unwrap an exception to reach the
information it actually wanted.
