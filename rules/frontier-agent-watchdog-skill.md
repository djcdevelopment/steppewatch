# Frontier Agent & Subagent Watchdog Ruleset (Drop-In Starter Skill)

> Drop this into `CLAUDE.md`, `AGENTS.md`, `.claude/skills/watchdog-checkpoints/SKILL.md`, or your Hermes agent skill/instructions.  
> **Goal:** Eliminate silent planning loops, hallucinated file/line claims, and runaway subagents while giving your local machine's zero-token watcher clean signals to supervise execution and log JEPA residual episodes.

---

## 1. Declare Intent Before Editing (First Turn / Subagent Launch)

Before making tool calls or editing files, emit one compact `INTENT` line naming the concrete scope and exit condition:

```text
INTENT: <1-sentence goal> | TARGETS: <exact files/modules expected> | DONE_WHEN: <verifiable test/command>
```

Keep work strictly inside `TARGETS`. If you discover you must touch files outside `TARGETS`, emit a new `SCOPE_EXPANSION:` line stating *why* before editing them.

## 2. Anti-Stall & Anti-Paraphrase Rule (Progress or Escalate)

Never spend more than **2 consecutive turns** re-reading files or restating a plan without either:
- Executing a concrete edit, command, or test targeting a *new* symbol/file/error, **or**
- Emitting an explicit `STUCK:` block:

```text
STUCK: <exact blocker or missing fact> | TRIED: <concrete files/commands already checked> | NEED: <specific human decision or missing input>
```

Reporting `STUCK:` in 15 seconds is strictly better than burning 5 polite turns repeating the same analysis.

## 3. Grounded Claims Only (Zero Assumed Absences)

- Never claim a test, guard, symbol, or config field is *"missing"* without running a direct search (`rg` / `grep` / symbol read) against the target file in the current session.
- Never invent or approximate line numbers from memory. When citing code, cite exact `path:line` only when verified from direct line-numbered output.
- Separate **verified facts** (directly observed via tool output) from **hypotheses** (inferred, not yet executed).

## 4. Compact End-of-Task Receipt (Feeds Local JEPA Residual Episodes)

When finishing a task or subagent slice, end with a 3-line structured receipt so both humans and local watchdogs can compare `intended_outcome` vs. `end_state` at zero extra token cost:

```text
END_STATE: <pass | partial | stuck> | TOUCHED: <comma-separated files actually modified, or none>
VERIFIED: <exact build/test command and pass/fail count>
DELTA_NOTES: <any unexpected failure, scope change, or trap worth remembering for next time; or "none">
```

## 5. Subagent Supervision Contract (When Spawning Parallel Workers)

When launching subagents:
- Give each subagent disjoint `TARGETS` so parallel workers do not collide on the same files.
- Require every subagent to follow Rules 1–4 above.
- If a local watchdog raises an advisory flag (`thrash_detected = true` -> `advisory_inspect_error_loop` or `advisory_inspect_loop`), inspect the subagent's last 5 steps before intervening; never auto-terminate an agent from a heuristic score alone.
