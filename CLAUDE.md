# Project workflow

This project uses three subagents (defined in `.claude/agents/`):

- `developer`: implements features, fixes, and refactors.
- `unit-tester`: writes and runs unit tests. Only edits test files.
- `reviewer`: read-only code review. Reports findings, never edits.

## Standard flow for any non-trivial change

1. **Implement**: delegate to `developer`.
2. **Test**: delegate to `unit-tester` with the developer's summary and the list of changed files.
3. **Review**: delegate to `reviewer` on the resulting diff (`git diff`). `reviewer` and `unit-tester` may run in parallel since neither edits production code.
4. **Fix loop**: if `reviewer` returns **Request changes**, or a test exposes a real bug, send the findings to `developer`, then re-run `unit-tester` and `reviewer`. Cap at **2 fix rounds**; if still not approved, stop and report what remains to the user.
5. **Report**: summarize what changed, test results, and the reviewer's verdict.

Run the whole flow with `/ship <task>`.

## Rules

- Trivial edits (typos, comments, one-line config) don't need the full flow.
- When delegating, pass context explicitly: subagents don't share reasoning, only the working tree. Include the goal, changed files, and prior findings.
- Never commit or push as part of the flow unless the user asks.
- Report failures, skipped steps, and tests that couldn't run plainly.
